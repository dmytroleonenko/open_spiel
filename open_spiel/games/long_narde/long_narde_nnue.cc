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
#include "open_spiel/games/long_narde/long_narde_nnue.h"

#include "open_spiel/games/long_narde/long_narde_nnue_features.h"
#include "open_spiel/games/long_narde/long_narde_nnue_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace open_spiel {
namespace long_narde {
namespace nnue {
namespace {
constexpr int kSigmoidTableSize = 4096;
constexpr float kSigmoidMin = -8.0f;
constexpr float kSigmoidMax = 8.0f;

float SigmoidApprox(float x) {
  static const std::array<float, kSigmoidTableSize> table = []() {
    std::array<float, kSigmoidTableSize> vals{};
    for (int i = 0; i < kSigmoidTableSize; ++i) {
      float t =
          static_cast<float>(i) / static_cast<float>(kSigmoidTableSize - 1);
      float v = kSigmoidMin + t * (kSigmoidMax - kSigmoidMin);
      vals[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return vals;
  }();
  if (x <= kSigmoidMin) {
    return table.front();
  }
  if (x >= kSigmoidMax) {
    return table.back();
  }
  float pos = (x - kSigmoidMin) / (kSigmoidMax - kSigmoidMin);
  float idx = pos * static_cast<float>(kSigmoidTableSize - 1);
  int i = static_cast<int>(idx);
  float frac = idx - static_cast<float>(i);
  return table[i] + frac * (table[i + 1] - table[i]);
}
uint32_t BlockedBits(const std::array<int, kNumPoints>& row) {
  uint32_t bits = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    if (row[i] >= 1) {
      bits |= (1u << i);
    }
  }
  return bits;
}
const std::array<uint32_t, kNnueRunFeaturesPerSide>& RunMasks() {
  static const std::array<uint32_t, kNnueRunFeaturesPerSide> masks = []() {
    std::array<uint32_t, kNnueRunFeaturesPerSide> out{};
    int idx = 0;
    for (int len = kNnueRunMin; len <= kNnueRunMax; ++len) {
      int limit = kNumPoints - len;
      for (int start = 0; start <= limit; ++start) {
        uint32_t mask = 0;
        for (int i = 0; i < len; ++i) {
          mask |= (1u << (start + i));
        }
        out[idx++] = mask;
      }
    }
    return out;
  }();
  return masks;
}

void CollectActiveFeaturesInternal(const LongNardeState& state,
                                   NnueActiveFeatures* active) {
  active->count = 0;
  const auto& board = state.board();
  for (int side = 0; side < kNumPlayers; ++side) {
    int base_offset = side * kNnueBaseFeaturesPerSide;
    int total = 0;
    for (int point = 0; point < kNumPoints; ++point) {
      int count = board[side][point];
      total += count;
      int bucket = std::min(std::max(count, 0), kNnueBucketCount - 1);
      int idx = base_offset + point * kNnueBucketCount + bucket;
      active->indices[active->count++] = idx;
    }
    int off = kNumCheckersPerPlayer - total;
    int off_bucket = std::min(std::max(off, 0), kNnueBucketCount - 1);
    int idx = base_offset + kNumPoints * kNnueBucketCount + off_bucket;
    active->indices[active->count++] = idx;
  }
  const auto& masks = RunMasks();
  for (int side = 0; side < kNumPlayers; ++side) {
    uint32_t blocked = BlockedBits(board[side]);
    int run_offset = kNnueBaseFeatures + side * kNnueRunFeaturesPerSide;
    for (int i = 0; i < kNnueRunFeaturesPerSide; ++i) {
      if ((blocked & masks[i]) == masks[i]) {
        active->indices[active->count++] = run_offset + i;
      }
    }
  }
  int pip_bucket = PipDeltaBucketFromBoard(board);
  active->indices[active->count++] = kNnuePipDeltaOffset + pip_bucket;
  int mobility_bucket = MobilityBucketFromBoard(board);
  internal::Board opp_board = board;
  internal::FlipBoard(&opp_board);
  int opp_mobility_bucket = MobilityBucketFromBoard(opp_board);
  active->indices[active->count++] = kNnueMobilityOffset + mobility_bucket;
  active->indices[active->count++] =
      kNnueOppMobilityOffset + opp_mobility_bucket;
}
void BuildAccumulatorInternal(const NnueNetwork& net,
                              const NnueActiveFeatures& active,
                              std::array<int16_t, kNnueL1>* acc_out) {
  std::memcpy(acc_out->data(), net.b0.data(), kNnueL1 * sizeof(int16_t));
  AddRowFn add_row = GetAddRowKernel();
  for (int i = 0; i < active.count; ++i) {
    int feature = active.indices[i];
    const int16_t* row = &net.w0[feature * kNnueL1];
    add_row(row, acc_out->data());
  }
}
void UpdateAccumulatorInternal(const NnueNetwork& net,
                               const NnueActiveFeatures& old_active,
                               const NnueActiveFeatures& new_active,
                               std::array<int16_t, kNnueL1>* acc_out) {
  AddRowFn add_row = GetAddRowKernel();
  SubRowFn sub_row = GetSubRowKernel();
  int i = 0;
  int j = 0;
  while (i < old_active.count && j < new_active.count) {
    int old_idx = old_active.indices[i];
    int new_idx = new_active.indices[j];
    if (old_idx == new_idx) {
      ++i;
      ++j;
      continue;
    }
    if (old_idx < new_idx) {
      const int16_t* row = &net.w0[old_idx * kNnueL1];
      sub_row(row, acc_out->data());
      ++i;
    } else {
      const int16_t* row = &net.w0[new_idx * kNnueL1];
      add_row(row, acc_out->data());
      ++j;
    }
  }
  for (; i < old_active.count; ++i) {
    int idx = old_active.indices[i];
    const int16_t* row = &net.w0[idx * kNnueL1];
    sub_row(row, acc_out->data());
  }
  for (; j < new_active.count; ++j) {
    int idx = new_active.indices[j];
    const int16_t* row = &net.w0[idx * kNnueL1];
    add_row(row, acc_out->data());
  }
}

int32_t ComputeOutput(const int8_t* input, const int8_t* weights,
                      const int8_t* bias, int in_dim) {
  int32_t sum = bias[0] * kNnueFactor;
  for (int i = 0; i < in_dim; ++i) {
    sum += static_cast<int32_t>(input[i]) * weights[i];
  }
  return sum / kNnueFactor;
}

}  // namespace

void CollectActiveFeatures(const LongNardeState& state,
                           NnueActiveFeatures* active) {
  CollectActiveFeaturesInternal(state, active);
}

void BuildAccumulator(const NnueNetwork& net,
                      const NnueActiveFeatures& active,
                      std::array<int16_t, kNnueL1>* acc_out) {
  BuildAccumulatorInternal(net, active, acc_out);
}

void UpdateAccumulator(const NnueNetwork& net,
                       const NnueActiveFeatures& old_active,
                       const NnueActiveFeatures& new_active,
                       std::array<int16_t, kNnueL1>* acc_out) {
  UpdateAccumulatorInternal(net, old_active, new_active, acc_out);
}

void ApplyAccumulatorDelta(const NnueNetwork& net,
                           const NnueActiveFeatures& remove,
                           const NnueActiveFeatures& add,
                           std::array<int16_t, kNnueL1>* acc_out) {
  AddRowFn add_row = GetAddRowKernel();
  SubRowFn sub_row = GetSubRowKernel();
  for (int i = 0; i < remove.count; ++i) {
    int idx = remove.indices[i];
    const int16_t* row = &net.w0[idx * kNnueL1];
    sub_row(row, acc_out->data());
  }
  for (int i = 0; i < add.count; ++i) {
    int idx = add.indices[i];
    const int16_t* row = &net.w0[idx * kNnueL1];
    add_row(row, acc_out->data());
  }
}

NnueEval NnueEvaluator::EvaluateState(const LongNardeState& state) const {
  NnueEval eval;
  if (model_ == nullptr || !model_->IsLoaded()) {
    return eval;
  }
  NnueActiveFeatures active;
  CollectActiveFeatures(state, &active);
  std::array<int16_t, kNnueL1> acc;
  BuildAccumulator(model_->network(), active, &acc);
  return EvaluateFromAccumulator(model_->network(), acc);
}
void CollectActiveFeatureIndices(const LongNardeState& state,
                                 std::vector<int>* out) {
  SPIEL_CHECK_TRUE(out != nullptr);
  NnueActiveFeatures active;
  CollectActiveFeatures(state, &active);
  out->assign(active.indices.begin(), active.indices.begin() + active.count);
}
NnueEval EvaluateFromAccumulator(const NnueNetwork& network,
                                 const std::array<int16_t, kNnueL1>& acc) {
  NnueEval eval;
  std::array<int8_t, kNnueL1> l1;
  for (int i = 0; i < kNnueL1; ++i) {
    l1[i] = ClampAcc(acc[i]);
  }

  std::array<int8_t, kNnueL2> l2;
  ComputeLayerFn compute_layer = GetComputeLayerKernel();
  compute_layer(l1.data(), network.w1.data(), network.b1.data(), kNnueL1,
                kNnueL2, l2.data());
  int32_t out_win =
      ComputeOutput(l2.data(), network.w2.data(), &network.b2[0], kNnueL2);
  int32_t out_mars = ComputeOutput(l2.data(), network.w2.data() + kNnueL2,
                                   &network.b2[1], kNnueL2);
  int32_t out_opp_mars =
      ComputeOutput(l2.data(), network.w2.data() + 2 * kNnueL2,
                    &network.b2[2], kNnueL2);
  int32_t out_opp_mobility =
      ComputeOutput(l2.data(), network.w2.data() + 3 * kNnueL2,
                    &network.b2[3], kNnueL2);

  float logit_win = static_cast<float>(out_win);
  float logit_mars = static_cast<float>(out_mars);
  float logit_opp_mars = static_cast<float>(out_opp_mars);
  float logit_opp_mobility = static_cast<float>(out_opp_mobility);
  eval.p_win = SigmoidApprox(logit_win);
  eval.p_mars = SigmoidApprox(logit_mars);
  eval.p_opp_mars = SigmoidApprox(logit_opp_mars);
  eval.p_opp_mobility = SigmoidApprox(logit_opp_mobility);
  eval.ev =
      eval.p_win + eval.p_mars - (1.0f - eval.p_win) - eval.p_opp_mars;
  return eval;
}
NnueRawOutput EvaluateRawFromFeatures(
    const NnueNetwork& network, const std::vector<int>& active_features) {
  NnueRawOutput output;
  NnueActiveFeatures active;
  active.count = 0;
  for (int idx : active_features) {
    SPIEL_CHECK_GE(idx, 0);
    SPIEL_CHECK_LT(idx, kNnueFeatureDim);
    SPIEL_CHECK_LT(active.count, kNnueMaxActiveFeatures);
    active.indices[active.count++] = idx;
  }
  std::array<int16_t, kNnueL1> acc;
  BuildAccumulator(network, active, &acc);
  std::array<int8_t, kNnueL1> l1;
  for (int i = 0; i < kNnueL1; ++i) {
    l1[i] = ClampAcc(acc[i]);
  }
  std::array<int8_t, kNnueL2> l2;
  ComputeLayerFn compute_layer = GetComputeLayerKernel();
  compute_layer(l1.data(), network.w1.data(), network.b1.data(), kNnueL1,
                kNnueL2, l2.data());

  output.logit_win =
      ComputeOutput(l2.data(), network.w2.data(), &network.b2[0], kNnueL2);
  output.logit_mars = ComputeOutput(l2.data(), network.w2.data() + kNnueL2,
                                    &network.b2[1], kNnueL2);
  output.logit_opp_mars =
      ComputeOutput(l2.data(), network.w2.data() + 2 * kNnueL2,
                    &network.b2[2], kNnueL2);
  output.logit_opp_mobility =
      ComputeOutput(l2.data(), network.w2.data() + 3 * kNnueL2,
                    &network.b2[3], kNnueL2);
  return output;
}

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel
