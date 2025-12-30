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
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr std::array<char, 4> kLnueMagic = {'L', 'N', 'U', 'E'};
constexpr std::array<char, 4> kLnueChunkMagic = {'C', 'H', 'N', 'K'};
constexpr std::array<char, 4> kLnueTrajectoryMagic = {'L', 'N', 'U', 'T'};
constexpr uint32_t kLnueTrajectoryVersion = 1;

void WriteRaw(std::ofstream* out, const void* data, std::size_t size) {
  out->write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(size));
}

void AppendRaw(std::string* out, const void* data, std::size_t size) {
  out->append(reinterpret_cast<const char*>(data),
              static_cast<std::string::size_type>(size));
}

void WriteU32(std::ofstream* out, uint32_t value) {
  WriteRaw(out, &value, sizeof(value));
}

void WriteU64(std::ofstream* out, uint64_t value) {
  WriteRaw(out, &value, sizeof(value));
}

void WriteHeader(std::ofstream* out, const LnueShardConfig& config) {
  WriteRaw(out, kLnueMagic.data(), kLnueMagic.size());
  WriteU32(out, kLnueFormatVersion);
  WriteU32(out, kLnueEndianMarker);
  WriteU32(out, kLnueSchemaId);
  WriteU32(out, nnue::kNnueFeatureDim);
  uint32_t flags = kLnueFlagHasRunFeatures | kLnueFlagDualHead |
                   kLnueFlagHasGameId | kLnueFlagHasPly |
                   kLnueFlagHasPipDelta | kLnueFlagHasMobility;
  WriteU32(out, flags);
  WriteU32(out, kLnueFeatureIndexBytes);
  WriteU64(out, config.seed);
  WriteU32(out, config.shard_id);
  WriteU32(out, config.run_block_threshold);
  std::array<uint32_t, 5> reserved{};
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

bool SerializeTrajectoryInternal(const std::vector<SelfPlaySample>& samples,
                                 const LnueShardConfig& config,
                                 std::string* out) {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  if (samples.empty()) {
    return true;
  }
  std::vector<uint32_t> offsets;
  std::vector<uint16_t> indices;
  uint32_t total_indices =
      ComputeOffsets(samples, 0, static_cast<int>(samples.size()), &offsets,
                     &indices);

  std::vector<float> v_search;
  std::vector<int8_t> outcome;
  std::vector<float> target;
  std::vector<uint64_t> game_id;
  std::vector<uint16_t> ply;
  FillColumnar(samples, 0, samples.size(), &v_search, &outcome, &target,
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

  LnueTrajectoryHeader header{};
  std::memcpy(header.magic, kLnueTrajectoryMagic.data(), sizeof(header.magic));
  header.version = kLnueTrajectoryVersion;
  header.endian = kLnueEndianMarker;
  header.schema_id = kLnueSchemaId;
  header.feature_dim = nnue::kNnueFeatureDim;
  header.flags = kLnueFlagHasRunFeatures | kLnueFlagDualHead |
                 kLnueFlagHasGameId | kLnueFlagHasPly |
                 kLnueFlagHasPipDelta | kLnueFlagHasMobility;
  header.feature_index_bytes = kLnueFeatureIndexBytes;
  header.num_samples = static_cast<uint32_t>(samples.size());
  header.bytes_offsets = bytes_offsets;
  header.bytes_indices = bytes_indices;
  header.bytes_search = bytes_search;
  header.bytes_outcome = bytes_outcome;
  header.bytes_target = bytes_target;
  header.bytes_game_id = bytes_game_id;
  header.bytes_ply = bytes_ply;

  AppendRaw(out, &header, sizeof(header));
  AppendRaw(out, offsets.data(), bytes_offsets);
  AppendRaw(out, indices.data(), bytes_indices);
  AppendRaw(out, v_search.data(), bytes_search);
  AppendRaw(out, outcome.data(), bytes_outcome);
  AppendRaw(out, target.data(), bytes_target);
  AppendRaw(out, game_id.data(), bytes_game_id);
  AppendRaw(out, ply.data(), bytes_ply);
  return static_cast<uint32_t>(indices.size()) == total_indices;
}

bool ReadExact(const char* data, std::size_t size, std::size_t* offset,
               void* dst, std::size_t bytes) {
  if (offset == nullptr || dst == nullptr) {
    return false;
  }
  if (*offset + bytes > size) {
    return false;
  }
  std::memcpy(dst, data + *offset, bytes);
  *offset += bytes;
  return true;
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

  WriteHeader(&out, config);

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

bool SerializeLnueTrajectory(const std::vector<SelfPlaySample>& samples,
                             const LnueShardConfig& config,
                             std::string* out) {
  if (nnue::kNnueFeatureDim > 0xFFFF) {
    return false;
  }
  if (config.run_block_threshold != 1) {
    return false;
  }
  return SerializeTrajectoryInternal(samples, config, out);
}

bool DeserializeLnueTrajectory(const std::string& payload,
                               std::vector<SelfPlaySample>* samples) {
  if (samples == nullptr) {
    return false;
  }
  samples->clear();
  if (payload.empty()) {
    return true;
  }
  std::size_t offset = 0;
  LnueTrajectoryHeader header{};
  if (!ReadExact(payload.data(), payload.size(), &offset, &header,
                 sizeof(header))) {
    return false;
  }
  if (std::memcmp(header.magic, kLnueTrajectoryMagic.data(),
                  kLnueTrajectoryMagic.size()) != 0) {
    return false;
  }
  if (header.version != kLnueTrajectoryVersion ||
      header.endian != kLnueEndianMarker ||
      header.schema_id != kLnueSchemaId ||
      header.feature_dim != nnue::kNnueFeatureDim ||
      header.feature_index_bytes != kLnueFeatureIndexBytes) {
    return false;
  }
  uint32_t count = header.num_samples;
  std::vector<uint32_t> offsets(count + 1);
  std::vector<uint16_t> indices(header.bytes_indices / sizeof(uint16_t));
  std::vector<float> v_search(count);
  std::vector<int8_t> outcome(count);
  std::vector<float> target(count);
  std::vector<uint64_t> game_id(count);
  std::vector<uint16_t> ply(count);

  if (!ReadExact(payload.data(), payload.size(), &offset, offsets.data(),
                 header.bytes_offsets) ||
      !ReadExact(payload.data(), payload.size(), &offset, indices.data(),
                 header.bytes_indices) ||
      !ReadExact(payload.data(), payload.size(), &offset, v_search.data(),
                 header.bytes_search) ||
      !ReadExact(payload.data(), payload.size(), &offset, outcome.data(),
                 header.bytes_outcome) ||
      !ReadExact(payload.data(), payload.size(), &offset, target.data(),
                 header.bytes_target) ||
      !ReadExact(payload.data(), payload.size(), &offset, game_id.data(),
                 header.bytes_game_id) ||
      !ReadExact(payload.data(), payload.size(), &offset, ply.data(),
                 header.bytes_ply)) {
    return false;
  }
  samples->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t start = offsets[i];
    uint32_t end = offsets[i + 1];
    if (end < start || end > indices.size()) {
      return false;
    }
    SelfPlaySample sample;
    sample.active_features.reserve(end - start);
    for (uint32_t j = start; j < end; ++j) {
      sample.active_features.push_back(indices[j]);
    }
    sample.search_value = v_search[i];
    sample.outcome_value = outcome[i];
    sample.target_value = target[i];
    sample.game_id = game_id[i];
    sample.ply = ply[i];
    samples->push_back(std::move(sample));
  }
  return true;
}

LnueStreamWriter::~LnueStreamWriter() { Close(); }

bool LnueStreamWriter::Open(const std::string& path,
                            const LnueShardConfig& config) {
  Close();
  if (nnue::kNnueFeatureDim > 0xFFFF) {
    return false;
  }
  config_ = config;
  path_ = path;
  tmp_path_.clear();
  std::string out_path = path;
  if (config.write_tmp) {
    tmp_path_ = path + ".tmp";
    out_path = tmp_path_;
  }
  out_ = new std::ofstream(out_path, std::ios::binary | std::ios::out);
  if (!out_->is_open()) {
    delete out_;
    out_ = nullptr;
    path_.clear();
    tmp_path_.clear();
    return false;
  }
  WriteHeader(out_, config_);
  buffer_.clear();
  return true;
}

bool LnueStreamWriter::AddSamples(const std::vector<SelfPlaySample>& samples) {
  if (out_ == nullptr) {
    return false;
  }
  if (samples.empty()) {
    return true;
  }
  buffer_.insert(buffer_.end(), samples.begin(), samples.end());
  int chunk = std::max(1, config_.samples_per_chunk);
  while (static_cast<int>(buffer_.size()) >= chunk) {
    if (!FlushChunk(chunk)) {
      return false;
    }
  }
  return true;
}

bool LnueStreamWriter::Flush() {
  if (out_ == nullptr) {
    return false;
  }
  if (buffer_.empty()) {
    return true;
  }
  return FlushChunk(static_cast<int>(buffer_.size()));
}

void LnueStreamWriter::Close() {
  if (out_ != nullptr) {
    if (!buffer_.empty()) {
      FlushChunk(static_cast<int>(buffer_.size()));
    }
    out_->close();
    delete out_;
    out_ = nullptr;
    if (!tmp_path_.empty() && !path_.empty()) {
      std::remove(path_.c_str());
      std::rename(tmp_path_.c_str(), path_.c_str());
    }
  }
  buffer_.clear();
  path_.clear();
  tmp_path_.clear();
}

bool LnueStreamWriter::FlushChunk(int count) {
  if (out_ == nullptr || count <= 0) {
    return false;
  }
  count = std::min(count, static_cast<int>(buffer_.size()));
  std::vector<uint32_t> offsets;
  std::vector<uint16_t> indices;
  ComputeOffsets(buffer_, 0, count, &offsets, &indices);

  std::vector<float> v_search;
  std::vector<int8_t> outcome;
  std::vector<float> target;
  std::vector<uint64_t> game_id;
  std::vector<uint16_t> ply;
  FillColumnar(buffer_, 0, count, &v_search, &outcome, &target, &game_id, &ply);

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

  WriteChunkHeader(out_, uncompressed_bytes, /*compressed_bytes=*/0, count);
  WriteRaw(out_, offsets.data(), bytes_offsets);
  WriteRaw(out_, indices.data(), bytes_indices);
  WriteRaw(out_, v_search.data(), bytes_search);
  WriteRaw(out_, outcome.data(), bytes_outcome);
  WriteRaw(out_, target.data(), bytes_target);
  WriteRaw(out_, game_id.data(), bytes_game_id);
  WriteRaw(out_, ply.data(), bytes_ply);

  buffer_.erase(buffer_.begin(), buffer_.begin() + count);
  return true;
}

}  // namespace long_narde
}  // namespace open_spiel
