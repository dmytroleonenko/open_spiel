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
#include <string>

#include "open_spiel/games/long_narde/long_narde_selfplay.h"

namespace open_spiel {
namespace long_narde {

constexpr uint32_t kLnueFormatVersion = 1;
constexpr uint32_t kLnueEndianMarker = 0x01020304u;
constexpr uint32_t kLnueSchemaId = 1;
constexpr uint32_t kLnueFlagHasRunFeatures = 1u << 0;
constexpr uint32_t kLnueFlagDualHead = 1u << 1;
constexpr uint32_t kLnueFlagHasGameId = 1u << 2;
constexpr uint32_t kLnueFlagHasPly = 1u << 3;
constexpr uint32_t kLnueFeatureIndexBytes = 2;

struct LnueShardConfig {
  int samples_per_chunk = 4096;
  bool write_tmp = true;
};

bool WriteLnueShard(const std::string& path, const SelfPlayBatch& batch,
                    const LnueShardConfig& config);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_IO_H_
