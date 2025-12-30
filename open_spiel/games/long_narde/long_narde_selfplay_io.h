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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_IO_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_IO_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_selfplay.h"

namespace open_spiel {
namespace long_narde {

constexpr uint32_t kLnueFormatVersion = 3;
constexpr uint32_t kLnueEndianMarker = 0x01020304u;
constexpr uint32_t kLnueSchemaId = 2;
constexpr uint32_t kLnueFlagHasRunFeatures = 1u << 0;
constexpr uint32_t kLnueFlagDualHead = 1u << 1;
constexpr uint32_t kLnueFlagHasGameId = 1u << 2;
constexpr uint32_t kLnueFlagHasPly = 1u << 3;
constexpr uint32_t kLnueFlagHasPipDelta = 1u << 4;
constexpr uint32_t kLnueFlagHasMobility = 1u << 5;
constexpr uint32_t kLnueFeatureIndexBytes = 2;

struct LnueShardConfig {
  int samples_per_chunk = 4096;
  bool write_tmp = true;
  uint64_t seed = 0;
  uint32_t shard_id = 0;
  uint32_t run_block_threshold = 1;
};

struct LnueTrajectoryHeader {
  char magic[4];
  uint32_t version;
  uint32_t endian;
  uint32_t schema_id;
  uint32_t feature_dim;
  uint32_t flags;
  uint32_t feature_index_bytes;
  uint32_t num_samples;
  uint32_t bytes_offsets;
  uint32_t bytes_indices;
  uint32_t bytes_search;
  uint32_t bytes_outcome;
  uint32_t bytes_target;
  uint32_t bytes_game_id;
  uint32_t bytes_ply;
};

struct LnueResumeState {
  uint64_t seed = 0;
  uint32_t shard_id = 0;
  uint32_t run_block_threshold = 0;
  uint64_t games_done = 0;
  uint64_t samples_done = 0;
  std::uintmax_t valid_bytes = 0;
};

bool WriteLnueShard(const std::string& path, const SelfPlayBatch& batch,
                    const LnueShardConfig& config);

bool SerializeLnueTrajectory(const std::vector<SelfPlaySample>& samples,
                             const LnueShardConfig& config,
                             std::string* out);
bool DeserializeLnueTrajectory(const std::string& payload,
                               std::vector<SelfPlaySample>* samples);

class LnueStreamWriter {
 public:
  LnueStreamWriter() = default;
  ~LnueStreamWriter();

  bool Open(const std::string& path, const LnueShardConfig& config);
  bool OpenAppend(const std::string& path, const LnueShardConfig& config,
                  LnueResumeState* state);
  bool AddSamples(const std::vector<SelfPlaySample>& samples);
  bool Flush();
  void Close();
  int samples_in_buffer() const { return static_cast<int>(buffer_.size()); }

 private:
  bool FlushChunk(int count);

  std::ofstream* out_ = nullptr;
  LnueShardConfig config_{};
  std::string path_;
  std::string tmp_path_;
  std::vector<SelfPlaySample> buffer_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_IO_H_
