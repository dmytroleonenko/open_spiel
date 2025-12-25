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

// Portions of the NNUE inference layout are adapted from the Cerebrum project
// (MIT License, Copyright 2020-2025 David Carteau).

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"

namespace open_spiel {
namespace long_narde {
namespace nnue {

constexpr int kNnueBucketCount = 16;
constexpr int kNnuePointsWithOff = kNumPoints + 1;
constexpr int kNnueBaseFeaturesPerSide = kNnuePointsWithOff * kNnueBucketCount;
constexpr int kNnueBaseFeatures = kNnueBaseFeaturesPerSide * 2;
constexpr int kNnueRunMin = 2;
constexpr int kNnueRunMax = 6;
constexpr int kNnueRunCount2 = kNumPoints - 2 + 1;
constexpr int kNnueRunCount3 = kNumPoints - 3 + 1;
constexpr int kNnueRunCount4 = kNumPoints - 4 + 1;
constexpr int kNnueRunCount5 = kNumPoints - 5 + 1;
constexpr int kNnueRunCount6 = kNumPoints - 6 + 1;
constexpr int kNnueRunFeaturesPerSide = kNnueRunCount2 + kNnueRunCount3 +
kNnueRunCount4 + kNnueRunCount5 + kNnueRunCount6;
constexpr int kNnueRunFeatures = kNnueRunFeaturesPerSide * 2;
constexpr int kNnueFeatureDim = kNnueBaseFeatures + kNnueRunFeatures;
constexpr int kNnueL1 = 256;
constexpr int kNnueL2 = 32;
constexpr int kNnueL3 = 2;
static_assert(kNnueL1 % 16 == 0, "kNnueL1 must be multiple of 16.");
static_assert(kNnueL1 % 32 == 0, "kNnueL1 must be multiple of 32.");
constexpr uint32_t kNnueRunBlockThreshold = 1;
constexpr int kNnueFileVersion = 1;
constexpr uint32_t kNnueEndianMarker = 0x01020304u;
constexpr uint32_t kNnueQuantInt8 = 1;
constexpr uint32_t kNnueQuantInt16 = 2;

struct NnueEval {
  float p_win = 0.0f;
  float p_mars = 0.0f;
  float ev = 0.0f;
};

struct NnueRawOutput {
  int32_t logit_win = 0;
  int32_t logit_mars = 0;
};

struct NnueFileHeader {
  char magic[4];
  uint32_t version;
  uint32_t endian;
  uint32_t feature_dim;
  uint32_t l1;
  uint32_t l2;
  uint32_t l3;
  uint32_t run_features_enabled;
  uint32_t run_block_threshold;
  uint32_t w0_type;
  uint32_t w1_type;
  uint32_t w2_type;
  uint32_t b0_type;
  uint32_t b1_type;
  uint32_t b2_type;
  uint32_t payload_size;
  uint32_t payload_hash;
};

struct NnueNetwork {
  char name[64];
  char author[64];
  std::array<int16_t, kNnueFeatureDim * kNnueL1> w0;
  std::array<int16_t, kNnueL1> b0;
  std::array<int8_t, kNnueL1 * kNnueL2> w1;
  std::array<int8_t, kNnueL2> b1;
  std::array<int8_t, kNnueL2 * kNnueL3> w2;
  std::array<int8_t, kNnueL3> b2;
};

class NnueModel {
 public:
  // Loads a headered NNUE file or a raw NnueNetwork blob.
  bool Load(const std::string& path);
  bool IsLoaded() const { return loaded_; }
  const NnueNetwork& network() const { return network_; }

 private:
  NnueNetwork network_{};
  bool loaded_ = false;
};

class NnueEvaluator {
 public:
  explicit NnueEvaluator(const NnueModel* model) : model_(model) {}
  NnueEval EvaluateState(const LongNardeState& state) const;

 private:
  const NnueModel* model_;
};

void CollectActiveFeatureIndices(const LongNardeState& state,
                                 std::vector<int>* out);
NnueRawOutput EvaluateRawFromFeatures(const NnueNetwork& network,
                                      const std::vector<int>& active_features);

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_H_
