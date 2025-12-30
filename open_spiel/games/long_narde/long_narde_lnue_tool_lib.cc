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

#include "open_spiel/games/long_narde/long_narde_lnue_tool_lib.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr std::array<char, 4> kLnueMagic = {'L', 'N', 'U', 'E'};
constexpr std::array<char, 4> kLnueChunkMagic = {'C', 'H', 'N', 'K'};

struct LnueHeader {
  char magic[4];
  uint32_t version;
  uint32_t endian;
  uint32_t schema_id;
  uint32_t feature_dim;
  uint32_t flags;
  uint32_t feature_index_bytes;
  uint64_t seed;
  uint32_t shard_id;
  uint32_t run_block_threshold;
  uint32_t reserved[5];
};

struct LnueChunkHeader {
  char magic[4];
  uint32_t uncompressed_bytes;
  uint32_t compressed_bytes;
  uint32_t num_samples;
  uint32_t reserved[3];
};

struct OutputState {
  LnueStreamWriter writer;
  LnueShardConfig config;
  std::string out_dir;
  std::string out_prefix;
  int games_per_shard = 0;
  int shard_index = 0;
  int games_in_shard = 0;
  int start_index = 0;
  int out_shards = 0;
  bool limit_output = false;
  bool open = false;
  bool limit_reached = false;
  uint64_t total_games = 0;
};

std::string Basename(const std::string& path) {
  std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

bool ReadExact(std::ifstream* in, void* data, std::size_t size) {
  in->read(reinterpret_cast<char*>(data),
           static_cast<std::streamsize>(size));
  return in->good() && in->gcount() == static_cast<std::streamsize>(size);
}

bool ReadLnueHeader(std::ifstream* in, LnueHeader* header) {
  if (!ReadExact(in, header->magic, sizeof(header->magic))) {
    return false;
  }
  if (!ReadExact(in, &header->version, sizeof(header->version)) ||
      !ReadExact(in, &header->endian, sizeof(header->endian)) ||
      !ReadExact(in, &header->schema_id, sizeof(header->schema_id)) ||
      !ReadExact(in, &header->feature_dim, sizeof(header->feature_dim)) ||
      !ReadExact(in, &header->flags, sizeof(header->flags)) ||
      !ReadExact(in, &header->feature_index_bytes,
                 sizeof(header->feature_index_bytes)) ||
      !ReadExact(in, &header->seed, sizeof(header->seed)) ||
      !ReadExact(in, &header->shard_id, sizeof(header->shard_id)) ||
      !ReadExact(in, &header->run_block_threshold,
                 sizeof(header->run_block_threshold)) ||
      !ReadExact(in, header->reserved, sizeof(header->reserved))) {
    return false;
  }
  if (std::memcmp(header->magic, kLnueMagic.data(), kLnueMagic.size()) != 0) {
    return false;
  }
  if (header->version != kLnueFormatVersion ||
      header->endian != kLnueEndianMarker ||
      header->feature_index_bytes != kLnueFeatureIndexBytes ||
      header->feature_dim != nnue::kNnueFeatureDim) {
    return false;
  }
  return true;
}

bool ReadLnueChunkHeader(std::ifstream* in, LnueChunkHeader* header) {
  if (!ReadExact(in, header->magic, sizeof(header->magic))) {
    return false;
  }
  if (!ReadExact(in, &header->uncompressed_bytes,
                 sizeof(header->uncompressed_bytes)) ||
      !ReadExact(in, &header->compressed_bytes,
                 sizeof(header->compressed_bytes)) ||
      !ReadExact(in, &header->num_samples, sizeof(header->num_samples)) ||
      !ReadExact(in, header->reserved, sizeof(header->reserved))) {
    return false;
  }
  if (std::memcmp(header->magic, kLnueChunkMagic.data(),
                  kLnueChunkMagic.size()) != 0) {
    return false;
  }
  return true;
}

bool ReadChunkSamples(std::ifstream* in, const LnueChunkHeader& chunk,
                      std::vector<SelfPlaySample>* samples_out) {
  if (chunk.compressed_bytes != 0 || chunk.num_samples == 0) {
    return false;
  }
  std::vector<uint32_t> offsets(chunk.num_samples + 1);
  if (!ReadExact(in, offsets.data(),
                 offsets.size() * sizeof(uint32_t))) {
    return false;
  }
  uint32_t indices_count = offsets.back();
  std::vector<uint16_t> indices(indices_count);
  if (!ReadExact(in, indices.data(),
                 indices.size() * sizeof(uint16_t))) {
    return false;
  }
  std::vector<float> v_search(chunk.num_samples);
  std::vector<int8_t> outcome(chunk.num_samples);
  std::vector<float> target(chunk.num_samples);
  std::vector<uint64_t> game_id(chunk.num_samples);
  std::vector<uint16_t> ply(chunk.num_samples);
  std::vector<float> mobility(chunk.num_samples);
  if (!ReadExact(in, v_search.data(),
                 v_search.size() * sizeof(float)) ||
      !ReadExact(in, outcome.data(),
                 outcome.size() * sizeof(int8_t)) ||
      !ReadExact(in, target.data(),
                 target.size() * sizeof(float)) ||
      !ReadExact(in, game_id.data(),
                 game_id.size() * sizeof(uint64_t)) ||
      !ReadExact(in, ply.data(),
                 ply.size() * sizeof(uint16_t)) ||
      !ReadExact(in, mobility.data(),
                 mobility.size() * sizeof(float))) {
    return false;
  }
  samples_out->reserve(samples_out->size() + chunk.num_samples);
  for (uint32_t i = 0; i < chunk.num_samples; ++i) {
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
    sample.mobility = mobility[i];
    samples_out->push_back(std::move(sample));
  }
  return true;
}

std::string ShardPath(const OutputState& state) {
  std::ostringstream oss;
  oss << state.out_dir << "/" << state.out_prefix << "_"
      << std::setw(4) << std::setfill('0')
      << (state.start_index + state.shard_index) << ".lnue";
  return oss.str();
}

bool OpenNextShard(OutputState* state) {
  if (state == nullptr) {
    return false;
  }
  if (state->limit_output &&
      state->shard_index >= state->out_shards) {
    state->limit_reached = true;
    return false;
  }
  state->config.shard_id =
      static_cast<uint32_t>(state->start_index + state->shard_index);
  if (!state->writer.Open(ShardPath(*state), state->config)) {
    return false;
  }
  state->open = true;
  state->games_in_shard = 0;
  return true;
}

bool CloseShard(OutputState* state) {
  if (state == nullptr || !state->open) {
    return true;
  }
  state->writer.Close();
  state->open = false;
  state->shard_index += 1;
  if (state->limit_output &&
      state->shard_index >= state->out_shards) {
    state->limit_reached = true;
  }
  return true;
}

bool FlushGame(OutputState* state,
               std::vector<SelfPlaySample>* game_samples) {
  if (state == nullptr || game_samples == nullptr ||
      game_samples->empty()) {
    return true;
  }
  if (!state->open) {
    if (!OpenNextShard(state)) {
      return false;
    }
  }
  uint64_t game_index = state->total_games;
  uint64_t game_id =
      (state->config.seed << 32) ^ static_cast<uint64_t>(game_index);
  for (auto& sample : *game_samples) {
    sample.game_id = game_id;
  }
  if (!state->writer.AddSamples(*game_samples)) {
    return false;
  }
  state->games_in_shard += 1;
  state->total_games += 1;
  game_samples->clear();
  if (state->games_in_shard >= state->games_per_shard) {
    return CloseShard(state);
  }
  return true;
}

}  // namespace

bool ParseShardName(const std::string& path, std::string* prefix, int* index) {
  std::string name = Basename(path);
  const std::string ext = ".lnue";
  if (name.size() > ext.size() &&
      name.compare(name.size() - ext.size(), ext.size(), ext) == 0) {
    name = name.substr(0, name.size() - ext.size());
  }
  std::size_t underscore = name.rfind('_');
  if (underscore == std::string::npos || underscore + 1 >= name.size()) {
    return false;
  }
  std::string digits = name.substr(underscore + 1);
  if (digits.empty()) {
    return false;
  }
  for (char c : digits) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  if (prefix != nullptr) {
    *prefix = name.substr(0, underscore);
  }
  if (index != nullptr) {
    *index = std::stoi(digits);
  }
  return true;
}

bool ProcessInputs(const std::vector<std::string>& inputs,
                   const ToolConfig& config) {
  if (!file::Mkdirs(config.out_dir)) {
    std::cerr << "Failed to create output dir: " << config.out_dir << "\n";
    return false;
  }
  OutputState out_state;
  out_state.out_dir = config.out_dir;
  out_state.out_prefix = config.out_prefix;
  out_state.games_per_shard = config.games_per_shard;
  out_state.start_index = config.start_index;
  out_state.out_shards = config.out_shards;
  out_state.limit_output = config.limit_output;
  out_state.config.samples_per_chunk = config.samples_per_chunk;
  out_state.config.write_tmp = true;

  bool header_initialized = false;
  for (const std::string& path : inputs) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Failed to open: " << path << "\n";
      return false;
    }
    LnueHeader header{};
    if (!ReadLnueHeader(&in, &header)) {
      std::cerr << "Invalid LNUE header: " << path << "\n";
      return false;
    }
    uint32_t expected_flags =
        kLnueFlagHasRunFeatures | kLnueFlagDualHead | kLnueFlagHasGameId |
        kLnueFlagHasPly | kLnueFlagHasPipDelta | kLnueFlagHasMobility;
    if (header.schema_id != kLnueSchemaId ||
        header.flags != expected_flags) {
      std::cerr << "Unsupported LNUE flags in: " << path << "\n";
      return false;
    }
    if (!header_initialized) {
      out_state.config.seed = header.seed;
      out_state.config.run_block_threshold = header.run_block_threshold;
      header_initialized = true;
    }

    std::vector<SelfPlaySample> chunk_samples;
    std::vector<SelfPlaySample> game_samples;
    bool has_game = false;
    uint16_t last_ply = 0;
    while (true) {
      LnueChunkHeader chunk{};
      if (!ReadLnueChunkHeader(&in, &chunk)) {
        break;
      }
      chunk_samples.clear();
      if (!ReadChunkSamples(&in, chunk, &chunk_samples)) {
        std::cerr << "Failed to read chunk from: " << path << "\n";
        return false;
      }
      for (const auto& sample : chunk_samples) {
        bool new_game = false;
        if (!has_game) {
          new_game = true;
        } else if (sample.ply == 0 || sample.ply <= last_ply) {
          new_game = true;
        }
        if (new_game) {
          if (!FlushGame(&out_state, &game_samples)) {
            return false;
          }
          if (out_state.limit_reached) {
            break;
          }
          has_game = true;
        }
        last_ply = sample.ply;
        game_samples.push_back(sample);
      }
      if (out_state.limit_reached) {
        break;
      }
    }
    if (!FlushGame(&out_state, &game_samples)) {
      return false;
    }
    if (out_state.limit_reached) {
      break;
    }
  }
  if (out_state.open) {
    out_state.writer.Close();
    out_state.open = false;
    out_state.shard_index += 1;
  }
  if (config.limit_output &&
      out_state.shard_index < config.out_shards) {
    std::cerr << "Warning: only wrote " << out_state.shard_index
              << " of " << config.out_shards
              << " requested shard(s).\n";
  }

  if (out_state.total_games <=
      static_cast<uint64_t>(config.games_per_shard)) {
    std::cerr << "Warning: input games (" << out_state.total_games
              << ") <= games_per_shard (" << config.games_per_shard
              << ").\n";
  }
  std::cout << "Wrote " << out_state.total_games
            << " games into " << out_state.shard_index
            << " shard(s).\n";
  return true;
}

}  // namespace long_narde
}  // namespace open_spiel
