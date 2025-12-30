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

#include <sys/stat.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr std::array<char, 4> kLnueMagic = {'L', 'N', 'U', 'E'};
constexpr std::array<char, 4> kLnueChunkMagic = {'C', 'H', 'N', 'K'};

bool GetFileSize(const std::string& path, std::uintmax_t* size) {
  if (size == nullptr) {
    return false;
  }
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return false;
  }
  *size = static_cast<std::uintmax_t>(info.st_size);
  return true;
}

bool TruncateFile(const std::string& path, std::uintmax_t size) {
#ifdef _WIN32
  int fd = _open(path.c_str(), _O_RDWR | _O_BINARY);
  if (fd < 0) {
    return false;
  }
  int result = _chsize_s(fd, static_cast<size_t>(size));
  _close(fd);
  return result == 0;
#else
  return truncate(path.c_str(), static_cast<off_t>(size)) == 0;
#endif
}

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
  if (header->version != kLnueFormatVersion) {
    return false;
  }
  if (header->endian != kLnueEndianMarker) {
    return false;
  }
  if (header->feature_index_bytes != kLnueFeatureIndexBytes) {
    return false;
  }
  if (header->feature_dim != nnue::kNnueFeatureDim) {
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

bool ScanLnueFile(const std::string& path, const LnueShardConfig& config,
                  LnueResumeState* state) {
  if (state == nullptr) {
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  LnueHeader header{};
  if (!ReadLnueHeader(&in, &header)) {
    return false;
  }
  uint32_t expected_flags =
      kLnueFlagHasRunFeatures | kLnueFlagDualHead | kLnueFlagHasGameId |
      kLnueFlagHasPly | kLnueFlagHasPipDelta | kLnueFlagHasMobility;
  if (header.schema_id != kLnueSchemaId ||
      header.flags != expected_flags) {
    return false;
  }
  if (header.seed != config.seed || header.shard_id != config.shard_id ||
      header.run_block_threshold != config.run_block_threshold) {
    return false;
  }

  state->seed = header.seed;
  state->shard_id = header.shard_id;
  state->run_block_threshold = header.run_block_threshold;
  state->games_done = 0;
  state->samples_done = 0;
  state->valid_bytes = static_cast<std::uintmax_t>(in.tellg());

  uint64_t max_game = 0;
  bool has_game = false;
  while (true) {
    LnueChunkHeader chunk{};
    if (!ReadLnueChunkHeader(&in, &chunk)) {
      break;
    }
    if (chunk.compressed_bytes != 0 || chunk.num_samples == 0) {
      break;
    }
    std::vector<uint32_t> offsets(chunk.num_samples + 1);
    if (!ReadExact(&in, offsets.data(),
                   offsets.size() * sizeof(uint32_t))) {
      break;
    }
    uint32_t indices_count = offsets.back();
    std::size_t bytes_indices =
        static_cast<std::size_t>(indices_count) * kLnueFeatureIndexBytes;
    in.seekg(static_cast<std::streamoff>(bytes_indices), std::ios::cur);
    if (!in.good()) {
      break;
    }
    std::size_t bytes_search =
        static_cast<std::size_t>(chunk.num_samples) * sizeof(float);
    std::size_t bytes_outcome =
        static_cast<std::size_t>(chunk.num_samples) * sizeof(int8_t);
    std::size_t bytes_target =
        static_cast<std::size_t>(chunk.num_samples) * sizeof(float);
    std::size_t bytes_mobility =
        static_cast<std::size_t>(chunk.num_samples) * sizeof(float);
    in.seekg(static_cast<std::streamoff>(bytes_search + bytes_outcome +
                                         bytes_target),
             std::ios::cur);
    if (!in.good()) {
      break;
    }
    std::vector<uint64_t> game_id(chunk.num_samples);
    if (!ReadExact(&in, game_id.data(),
                   game_id.size() * sizeof(uint64_t))) {
      break;
    }
    for (uint64_t id : game_id) {
      uint32_t low = static_cast<uint32_t>(id & 0xFFFFFFFFu);
      if (!has_game || low > max_game) {
        max_game = low;
        has_game = true;
      }
    }
    std::size_t bytes_ply =
        static_cast<std::size_t>(chunk.num_samples) * sizeof(uint16_t);
    in.seekg(static_cast<std::streamoff>(bytes_ply + bytes_mobility),
             std::ios::cur);
    if (!in.good()) {
      break;
    }
    std::size_t bytes_offsets =
        offsets.size() * sizeof(uint32_t);
    std::size_t uncompressed =
        bytes_offsets + bytes_indices + bytes_search + bytes_outcome +
        bytes_target + game_id.size() * sizeof(uint64_t) + bytes_ply +
        bytes_mobility;
    if (uncompressed != chunk.uncompressed_bytes) {
      break;
    }
    state->samples_done += chunk.num_samples;
    state->valid_bytes = static_cast<std::uintmax_t>(in.tellg());
  }

  if (has_game) {
    state->games_done = static_cast<uint64_t>(max_game + 1);
  }
  return true;
}

}  // namespace

bool LnueStreamWriter::OpenAppend(const std::string& path,
                                  const LnueShardConfig& config,
                                  LnueResumeState* state) {
  Close();
  LnueResumeState local{};
  if (!ScanLnueFile(path, config, &local)) {
    return false;
  }
  std::error_code ec;
  std::uintmax_t size = 0;
  if (!GetFileSize(path, &size)) {
    return false;
  }
  if (local.valid_bytes < size) {
    if (!TruncateFile(path, local.valid_bytes)) {
      return false;
    }
  }
  config_ = config;
  config_.write_tmp = false;
  path_ = path;
  tmp_path_.clear();
  out_ = new std::ofstream(path, std::ios::binary | std::ios::out |
                                  std::ios::app);
  if (!out_->is_open()) {
    delete out_;
    out_ = nullptr;
    path_.clear();
    return false;
  }
  buffer_.clear();
  if (state != nullptr) {
    *state = local;
  }
  return true;
}

}  // namespace long_narde
}  // namespace open_spiel
