// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr std::array<char, 4> kLnueMagic = {'L', 'N', 'U', 'E'};
constexpr std::array<char, 4> kLnueChunkMagic = {'C', 'H', 'N', 'K'};

void WriteRaw(std::ofstream* out, const void* data, std::size_t size) {
  out->write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(size));
}

void WriteU32(std::ofstream* out, uint32_t value) {
  WriteRaw(out, &value, sizeof(value));
}

void WriteU64(std::ofstream* out, uint64_t value) {
  WriteRaw(out, &value, sizeof(value));
}

void WriteHeader(std::ofstream* out) {
  WriteRaw(out, kLnueMagic.data(), kLnueMagic.size());
  WriteU32(out, kLnueFormatVersion);
  WriteU32(out, kLnueEndianMarker);
  WriteU32(out, kLnueSchemaId);
  WriteU32(out, nnue::kNnueFeatureDim);
  uint32_t flags = kLnueFlagHasRunFeatures | kLnueFlagDualHead |
                   kLnueFlagHasGameId | kLnueFlagHasPly;
  WriteU32(out, flags);
  WriteU32(out, kLnueFeatureIndexBytes);
  std::array<uint32_t, 9> reserved{};
  WriteRaw(out, reserved.data(), reserved.size() * sizeof(uint32_t));
}

void WriteChunkHeader(std::ofstream* out, uint32_t uncompressed_bytes,
                      uint32_t compressed_bytes, uint32_t num_samples) {
  WriteRaw(out, kLnueChunkMagic.data(), kLnueChunkMagic.size());
  WriteU32(out, uncompressed_bytes);
  WriteU32(out, compressed_bytes);
  WriteU32(out, num_samples);
  std::array<uint32_t, 3> reserved{};
  WriteRaw(out, reserved.data(), reserved.size() * sizeof(uint32_t));
}

uint32_t ComputeOffsets(const std::vector<SelfPlaySample>& samples,
                        int start, int count,
                        std::vector<uint32_t>* offsets,
                        std::vector<uint16_t>* indices) {
  offsets->resize(static_cast<size_t>(count) + 1);
  (*offsets)[0] = 0;
  size_t total = 0;
  for (int i = 0; i < count; ++i) {
    total += samples[start + i].active_features.size();
    (*offsets)[static_cast<size_t>(i) + 1] =
        static_cast<uint32_t>(total);
  }
  indices->reserve(total);
  for (int i = 0; i < count; ++i) {
    for (int idx : samples[start + i].active_features) {
      SPIEL_CHECK_GE(idx, 0);
      SPIEL_CHECK_LT(idx, nnue::kNnueFeatureDim);
      indices->push_back(static_cast<uint16_t>(idx));
    }
  }
  return static_cast<uint32_t>(total);
}

void FillColumnar(const std::vector<SelfPlaySample>& samples, int start,
                  int count, std::vector<float>* v_search,
                  std::vector<int8_t>* outcome, std::vector<float>* target,
                  std::vector<uint64_t>* game_id,
                  std::vector<uint16_t>* ply) {
  v_search->resize(count);
  outcome->resize(count);
  target->resize(count);
  game_id->resize(count);
  ply->resize(count);
  for (int i = 0; i < count; ++i) {
    const SelfPlaySample& sample = samples[start + i];
    (*v_search)[i] = static_cast<float>(sample.search_value);
    double out = sample.outcome_value;
    SPIEL_CHECK_TRUE(out == -2.0 || out == -1.0 || out == 1.0 || out == 2.0);
    (*outcome)[i] = static_cast<int8_t>(out);
    (*target)[i] = static_cast<float>(sample.target_value);
    (*game_id)[i] = sample.game_id;
    (*ply)[i] = sample.ply;
  }
}

}  // namespace

bool WriteLnueShard(const std::string& path, const SelfPlayBatch& batch,
                    const LnueShardConfig& config) {
  if (nnue::kNnueFeatureDim > 0xFFFF) {
    SpielFatalError("LNUE requires feature dimension <= 65535.");
  }

  std::string out_path = path;
  if (config.write_tmp) {
    out_path = path + ".tmp";
  }

  std::ofstream out(out_path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    return false;
  }

  WriteHeader(&out);

  int total_samples = static_cast<int>(batch.samples.size());
  int samples_per_chunk = std::max(1, config.samples_per_chunk);
  for (int start = 0; start < total_samples; start += samples_per_chunk) {
    int count = std::min(samples_per_chunk, total_samples - start);
    std::vector<uint32_t> offsets;
    std::vector<uint16_t> indices;
    uint32_t total_indices =
        ComputeOffsets(batch.samples, start, count, &offsets, &indices);

    std::vector<float> v_search;
    std::vector<int8_t> outcome;
    std::vector<float> target;
    std::vector<uint64_t> game_id;
    std::vector<uint16_t> ply;
    FillColumnar(batch.samples, start, count, &v_search, &outcome, &target,
                 &game_id, &ply);

    uint32_t bytes_offsets =
        static_cast<uint32_t>(offsets.size() * sizeof(uint32_t));
    uint32_t bytes_indices =
        static_cast<uint32_t>(indices.size() * sizeof(uint16_t));
    uint32_t bytes_search =
        static_cast<uint32_t>(v_search.size() * sizeof(float));
    uint32_t bytes_outcome =
        static_cast<uint32_t>(outcome.size() * sizeof(int8_t));
    uint32_t bytes_target =
        static_cast<uint32_t>(target.size() * sizeof(float));
    uint32_t bytes_game_id =
        static_cast<uint32_t>(game_id.size() * sizeof(uint64_t));
    uint32_t bytes_ply =
        static_cast<uint32_t>(ply.size() * sizeof(uint16_t));

    uint32_t uncompressed_bytes = bytes_offsets + bytes_indices + bytes_search +
                                  bytes_outcome + bytes_target + bytes_game_id +
                                  bytes_ply;

    WriteChunkHeader(&out, uncompressed_bytes, /*compressed_bytes=*/0, count);
    WriteRaw(&out, offsets.data(), bytes_offsets);
    WriteRaw(&out, indices.data(), bytes_indices);
    WriteRaw(&out, v_search.data(), bytes_search);
    WriteRaw(&out, outcome.data(), bytes_outcome);
    WriteRaw(&out, target.data(), bytes_target);
    WriteRaw(&out, game_id.data(), bytes_game_id);
    WriteRaw(&out, ply.data(), bytes_ply);
  }

  out.close();
  if (config.write_tmp) {
    std::remove(path.c_str());
    std::rename(out_path.c_str(), path.c_str());
  }
  return true;
}

}  // namespace long_narde
}  // namespace open_spiel
