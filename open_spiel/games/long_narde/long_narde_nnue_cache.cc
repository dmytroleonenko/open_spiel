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

#include "open_spiel/games/long_narde/long_narde_nnue_cache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_internal.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr int kMaxRunsPerPoint = 20;

struct RunIndexList {
  std::array<std::array<uint8_t, kMaxRunsPerPoint>, kNumPoints> runs{};
  std::array<uint8_t, kNumPoints> counts{};
};

int BucketForCount(int count) {
  if (count < 0) {
    return 0;
  }
  if (count >= nnue::kNnueBucketCount) {
    return nnue::kNnueBucketCount - 1;
  }
  return count;
}

int BaseFeatureIndex(int side, int point, int count) {
  int bucket = BucketForCount(count);
  int base_offset = side * nnue::kNnueBaseFeaturesPerSide;
  return base_offset + point * nnue::kNnueBucketCount + bucket;
}

int OffFeatureIndex(int side, int off_count) {
  int bucket = BucketForCount(off_count);
  int base_offset = side * nnue::kNnueBaseFeaturesPerSide;
  return base_offset + kNumPoints * nnue::kNnueBucketCount + bucket;
}

void AddFeature(int idx, nnue::NnueActiveFeatures* feats) {
  SPIEL_CHECK_LT(feats->count, nnue::kNnueMaxActiveFeatures);
  feats->indices[feats->count++] = idx;
}

const std::array<uint32_t, nnue::kNnueRunFeaturesPerSide>& RunMasks();

void BuildActiveFromCache(const nnue::NnueCache& cache,
                          nnue::NnueActiveFeatures* active) {
  active->count = 0;
  for (int side = 0; side < kNumPlayers; ++side) {
    for (int point = 0; point < kNumPoints; ++point) {
      AddFeature(BaseFeatureIndex(side, point, cache.counts[side][point]),
                 active);
    }
    AddFeature(OffFeatureIndex(side, cache.off[side]), active);
  }

  const auto& masks = RunMasks();
  for (int side = 0; side < kNumPlayers; ++side) {
    uint32_t blocked = cache.blocked_bits[side];
    int run_offset = nnue::kNnueBaseFeatures +
                     side * nnue::kNnueRunFeaturesPerSide;
    for (int i = 0; i < nnue::kNnueRunFeaturesPerSide; ++i) {
      if ((blocked & masks[i]) == masks[i]) {
        AddFeature(run_offset + i, active);
      }
    }
  }
}

void RebuildAccumulatorFromCache(const nnue::NnueNetwork& net,
                                 const nnue::NnueCache& cache,
                                 std::array<int16_t, nnue::kNnueL1>* acc_out) {
  nnue::NnueActiveFeatures active;
  BuildActiveFromCache(cache, &active);
  nnue::BuildAccumulator(net, active, acc_out);
}

const std::array<uint32_t, nnue::kNnueRunFeaturesPerSide>& RunMasks() {
  static const std::array<uint32_t, nnue::kNnueRunFeaturesPerSide> masks =
      []() {
        std::array<uint32_t, nnue::kNnueRunFeaturesPerSide> out{};
        int idx = 0;
        for (int len = nnue::kNnueRunMin; len <= nnue::kNnueRunMax; ++len) {
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

const RunIndexList& RunsByPoint() {
  static const RunIndexList list = []() {
    RunIndexList out{};
    int idx = 0;
    for (int len = nnue::kNnueRunMin; len <= nnue::kNnueRunMax; ++len) {
      int limit = kNumPoints - len;
      for (int start = 0; start <= limit; ++start) {
        for (int i = 0; i < len; ++i) {
          int point = start + i;
          uint8_t pos = out.counts[point]++;
          out.runs[point][pos] = static_cast<uint8_t>(idx);
        }
        ++idx;
      }
    }
    return out;
  }();
  return list;
}

uint32_t BlockedBits(const std::array<uint8_t, kNumPoints>& row) {
  uint32_t bits = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    if (row[i] >= 1) {
      bits |= (1u << i);
    }
  }
  return bits;
}

void InitCacheFromState(const LongNardeState& state,
                        const nnue::NnueNetwork& net,
                        nnue::NnueCache* cache) {
  nnue::NnueActiveFeatures active;
  nnue::CollectActiveFeatures(state, &active);
  nnue::BuildAccumulator(net, active, &cache->acc);

  const auto& board = state.board();
  for (int side = 0; side < kNumPlayers; ++side) {
    int total = 0;
    for (int point = 0; point < kNumPoints; ++point) {
      int count = board[side][point];
      cache->counts[side][point] = static_cast<uint8_t>(count);
      total += count;
    }
    cache->off[side] =
        static_cast<uint8_t>(kNumCheckersPerPlayer - total);
    cache->blocked_bits[side] = BlockedBits(cache->counts[side]);
  }
}

void UpdateRunFeaturesForSide(int side, uint32_t old_bits, uint32_t new_bits,
                              nnue::NnueActiveFeatures* remove,
                              nnue::NnueActiveFeatures* add) {
  const auto& masks = RunMasks();
  int run_offset = nnue::kNnueBaseFeatures +
                   side * nnue::kNnueRunFeaturesPerSide;
  for (int i = 0; i < nnue::kNnueRunFeaturesPerSide; ++i) {
    uint32_t mask = masks[i];
    bool old_active = (old_bits & mask) == mask;
    bool new_active = (new_bits & mask) == mask;
    if (old_active == new_active) {
      continue;
    }
    int feature = run_offset + i;
    if (old_active) {
      AddFeature(feature, remove);
    } else {
      AddFeature(feature, add);
    }
  }
}

void UpdateRunFeaturesForPoints(uint32_t old_bits, uint32_t new_bits,
                                const std::array<int, kNumPoints>& points,
                                int point_count,
                                nnue::NnueActiveFeatures* remove,
                                nnue::NnueActiveFeatures* add) {
  const auto& masks = RunMasks();
  const auto& runs_by_point = RunsByPoint();
  std::array<uint8_t, nnue::kNnueRunFeaturesPerSide> touched{};
  std::array<int, kNumPoints * kMaxRunsPerPoint> runs{};
  int runs_count = 0;
  for (int i = 0; i < point_count; ++i) {
    int point = points[i];
    int count = runs_by_point.counts[point];
    for (int i = 0; i < count; ++i) {
      int run_idx = runs_by_point.runs[point][i];
      if (touched[run_idx] != 0) {
        continue;
      }
      touched[run_idx] = 1;
      runs[runs_count++] = run_idx;
    }
  }
  for (int i = 0; i < runs_count; ++i) {
    int run_idx = runs[i];
    uint32_t mask = masks[run_idx];
    bool old_active = (old_bits & mask) == mask;
    bool new_active = (new_bits & mask) == mask;
    if (old_active == new_active) {
      continue;
    }
    int feature = nnue::kNnueBaseFeatures + run_idx;
    if (old_active) {
      AddFeature(feature, remove);
    } else {
      AddFeature(feature, add);
    }
  }
}

void UpdateCacheNoFlip(const std::array<int8_t, kNumPoints>& delta,
                       int off_delta, nnue::NnueCache* cache,
                       nnue::NnueActiveFeatures* remove,
                       nnue::NnueActiveFeatures* add) {
  uint32_t old_bits = cache->blocked_bits[0];
  uint32_t new_bits = old_bits;
  std::array<int, kNumPoints> changed{};
  int changed_count = 0;
  for (int point = 0; point < kNumPoints; ++point) {
    int delta_count = delta[point];
    if (delta_count == 0) {
      continue;
    }
    int old_count = cache->counts[0][point];
    int new_count = old_count + delta_count;
    cache->counts[0][point] = static_cast<uint8_t>(new_count);
    int old_feat = BaseFeatureIndex(0, point, old_count);
    int new_feat = BaseFeatureIndex(0, point, new_count);
    if (old_feat != new_feat) {
      AddFeature(old_feat, remove);
      AddFeature(new_feat, add);
    }
    bool old_blocked = old_count >= 1;
    bool new_blocked = new_count >= 1;
    if (old_blocked != new_blocked) {
      if (new_blocked) {
        new_bits |= (1u << point);
      } else {
        new_bits &= ~(1u << point);
      }
      changed[changed_count++] = point;
    }
  }

  if (off_delta != 0) {
    int old_off = cache->off[0];
    int new_off = old_off + off_delta;
    cache->off[0] = static_cast<uint8_t>(new_off);
    int old_feat = OffFeatureIndex(0, old_off);
    int new_feat = OffFeatureIndex(0, new_off);
    if (old_feat != new_feat) {
      AddFeature(old_feat, remove);
      AddFeature(new_feat, add);
    }
  }

  if (changed_count > 0) {
    UpdateRunFeaturesForPoints(old_bits, new_bits, changed, changed_count,
                               remove, add);
  }

  cache->blocked_bits[0] = new_bits;
}

void UpdateCacheFlipRebuild(const std::array<int8_t, kNumPoints>& delta,
                            int off_delta, const nnue::NnueNetwork& net,
                            nnue::NnueCache* cache) {
  auto counts_after = cache->counts;
  auto off_after = cache->off;
  for (int point = 0; point < kNumPoints; ++point) {
    int delta_count = delta[point];
    if (delta_count == 0) {
      continue;
    }
    int new_count = counts_after[0][point] + delta_count;
    counts_after[0][point] = static_cast<uint8_t>(new_count);
  }
  if (off_delta != 0) {
    off_after[0] = static_cast<uint8_t>(off_after[0] + off_delta);
  }

  std::array<std::array<uint8_t, kNumPoints>, 2> new_counts{};
  for (int point = 0; point < kNumPoints; ++point) {
    int src = (point + 12) % kNumPoints;
    new_counts[0][point] = counts_after[1][src];
    new_counts[1][point] = counts_after[0][src];
  }
  std::array<uint8_t, 2> new_off{
      static_cast<uint8_t>(off_after[1]),
      static_cast<uint8_t>(off_after[0]),
  };

  cache->counts = new_counts;
  cache->off = new_off;
  cache->blocked_bits[0] = BlockedBits(new_counts[0]);
  cache->blocked_bits[1] = BlockedBits(new_counts[1]);
  RebuildAccumulatorFromCache(net, *cache, &cache->acc);
}

}  // namespace

void NnueCacheStack::Reset(const LongNardeState& state,
                           const nnue::NnueNetwork* network) {
  stack_.clear();
  network_ = network;
  if (network_ == nullptr) {
    return;
  }
  nnue::NnueCache root;
  InitCacheFromState(state, *network_, &root);
  stack_.push_back(root);
}

void NnueCacheStack::Clear() {
  stack_.clear();
  network_ = nullptr;
}

void NnueCacheStack::PushChance() {
  if (network_ == nullptr || stack_.empty()) {
    return;
  }
  stack_.push_back(stack_.back());
}

void NnueCacheStack::PushAction(const LongNardeState& state, Action action,
                                const std::array<int, 2>& dice,
                                Player prev_player) {
  if (network_ == nullptr || stack_.empty()) {
    return;
  }
  nnue::NnueCache child = stack_.back();
  nnue::NnueActiveFeatures remove;
  nnue::NnueActiveFeatures add;
  remove.count = 0;
  add.count = 0;

  internal::DecodedAction decoded = internal::DecodeAction(action);
  int d_min = std::min(dice[0], dice[1]);
  int d_max = std::max(dice[0], dice[1]);
  int d1 = (decoded.order == 0) ? d_max : d_min;
  int d2 = (decoded.order == 0) ? d_min : d_max;

  std::array<int8_t, kNumPoints> delta{};
  int off_delta = 0;
  auto apply = [&](int src, int die) {
    if (src == kActionPassSrc) {
      return;
    }
    int src_read = std::min(src, kNumPoints - 1);
    delta[src_read] -= 1;
    int target = src + die;
    if (target < kNumPoints) {
      delta[target] += 1;
    } else {
      off_delta += 1;
    }
  };

  apply(decoded.src1, d1);
  apply(decoded.src2, d2);

  bool flipped = state.current_player_id() != prev_player;
  if (flipped) {
    UpdateCacheFlipRebuild(delta, off_delta, *network_, &child);
  } else {
    UpdateCacheNoFlip(delta, off_delta, &child, &remove, &add);
    nnue::ApplyAccumulatorDelta(*network_, remove, add, &child.acc);
  }
  stack_.push_back(std::move(child));
}

void NnueCacheStack::Pop() {
  if (network_ == nullptr || stack_.empty()) {
    return;
  }
  stack_.pop_back();
}

const nnue::NnueCache* NnueCacheStack::Current() const {
  if (network_ == nullptr || stack_.empty()) {
    return nullptr;
  }
  return &stack_.back();
}

}  // namespace long_narde
}  // namespace open_spiel
