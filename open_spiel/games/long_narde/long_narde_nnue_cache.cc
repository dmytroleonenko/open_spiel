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

#include "open_spiel/games/long_narde/long_narde_internal.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr int kRunWords = 2;
constexpr int kMaxDeltaPoints = 4;

constexpr int MaxRunUpdatesPerPoint() {
  return (nnue::kNnueRunMax * (nnue::kNnueRunMax + 1)) / 2 - 1;
}

constexpr int kMaxRunUpdatesPerPoint = MaxRunUpdatesPerPoint();

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

struct PointDeltaList {
  std::array<int, kMaxDeltaPoints> points{};
  std::array<int8_t, kMaxDeltaPoints> deltas{};
  int count = 0;
};

struct RunUpdateEntry {
  uint16_t index = 0;
  uint32_t mask = 0;
};

struct RunUpdateList {
  std::array<RunUpdateEntry, kMaxRunUpdatesPerPoint> entries{};
  uint8_t count = 0;
};

void AddPointDelta(PointDeltaList* list, int point, int8_t delta) {
  if (delta == 0) {
    return;
  }
  for (int i = 0; i < list->count; ++i) {
    if (list->points[i] == point) {
      list->deltas[i] =
          static_cast<int8_t>(list->deltas[i] + delta);
      return;
    }
  }
  SPIEL_CHECK_LT(list->count, kMaxDeltaPoints);
  list->points[list->count] = point;
  list->deltas[list->count] = delta;
  ++list->count;
}

void SetRunBit(std::array<uint64_t, kRunWords>* bits, int index) {
  int word = index / 64;
  int bit = index % 64;
  (*bits)[word] |= (uint64_t{1} << bit);
}

void SetRunBitValue(std::array<uint64_t, kRunWords>* bits, int index,
                    bool value) {
  int word = index / 64;
  int bit = index % 64;
  uint64_t mask = (uint64_t{1} << bit);
  if (value) {
    (*bits)[word] |= mask;
  } else {
    (*bits)[word] &= ~mask;
  }
}

const std::array<int, nnue::kNnueRunMax + 2>& RunOffsets() {
  static const std::array<int, nnue::kNnueRunMax + 2> offsets = []() {
    std::array<int, nnue::kNnueRunMax + 2> out{};
    int offset = 0;
    for (int len = nnue::kNnueRunMin; len <= nnue::kNnueRunMax + 1; ++len) {
      out[len] = offset;
      if (len <= nnue::kNnueRunMax) {
        offset += kNumPoints - len + 1;
      }
    }
    return out;
  }();
  return offsets;
}

const std::array<RunUpdateList, kNumPoints>& RunUpdateLists() {
  static const std::array<RunUpdateList, kNumPoints> lists = []() {
    std::array<RunUpdateList, kNumPoints> out{};
    const auto& offsets = RunOffsets();
    for (int point = 0; point < kNumPoints; ++point) {
      RunUpdateList list;
      int count = 0;
      for (int len = nnue::kNnueRunMin; len <= nnue::kNnueRunMax; ++len) {
        int start_min = std::max(0, point - (len - 1));
        int start_max = std::min(point, kNumPoints - len);
        if (start_min > start_max) {
          continue;
        }
        uint32_t mask_base = (uint32_t{1} << len) - 1u;
        int offset = offsets[len];
        for (int start = start_min; start <= start_max; ++start) {
          SPIEL_CHECK_LT(count, kMaxRunUpdatesPerPoint);
          list.entries[count++] = {static_cast<uint16_t>(offset + start),
                                   mask_base << start};
        }
      }
      list.count = static_cast<uint8_t>(count);
      out[point] = list;
    }
    return out;
  }();
  return lists;
}

std::array<uint64_t, kRunWords> ComputeRunBits(uint32_t blocked) {
  std::array<uint64_t, kRunWords> out{};
  int offset = 0;
  for (int len = nnue::kNnueRunMin; len <= nnue::kNnueRunMax; ++len) {
    uint32_t run = blocked;
    for (int shift = 1; shift < len; ++shift) {
      run &= (blocked >> shift);
    }
    int max_start = kNumPoints - len;
    while (run != 0u) {
      int bit = __builtin_ctz(run);
      if (bit > max_start) {
        break;
      }
      SetRunBit(&out, offset + bit);
      run &= run - 1;
    }
    offset += (max_start + 1);
  }
  return out;
}

void UpdateRunBitsIncremental(uint32_t old_blocked, uint32_t new_blocked,
                              std::array<uint64_t, kRunWords>* run_bits) {
  uint32_t changed = old_blocked ^ new_blocked;
  if (changed == 0u) {
    return;
  }
  std::array<uint8_t, nnue::kNnueRunFeaturesPerSide> touched{};
  const auto& lists = RunUpdateLists();
  while (changed != 0u) {
    int point = __builtin_ctz(changed);
    changed &= changed - 1;
    const RunUpdateList& list = lists[point];
    for (int i = 0; i < list.count; ++i) {
      int index = list.entries[i].index;
      if (touched[index]) {
        continue;
      }
      touched[index] = 1;
      uint32_t mask = list.entries[i].mask;
      bool active = (new_blocked & mask) == mask;
      SetRunBitValue(run_bits, index, active);
    }
  }
}

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

  for (int side = 0; side < kNumPlayers; ++side) {
    const auto& runs = cache.run_bits[side];
    int run_offset = nnue::kNnueBaseFeatures +
                     side * nnue::kNnueRunFeaturesPerSide;
    for (int word = 0; word < kRunWords; ++word) {
      uint64_t bits = runs[word];
      while (bits != 0u) {
        int bit = __builtin_ctzll(bits);
        int index = word * 64 + bit;
        if (index >= nnue::kNnueRunFeaturesPerSide) {
          break;
        }
        AddFeature(run_offset + index, active);
        bits &= bits - 1;
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
    cache->run_bits[side] = ComputeRunBits(cache->blocked_bits[side]);
  }
}

void UpdateRunFeaturesDelta(
    int run_offset, const std::array<uint64_t, kRunWords>& old_bits,
    const std::array<uint64_t, kRunWords>& new_bits,
    nnue::NnueActiveFeatures* remove, nnue::NnueActiveFeatures* add) {
  for (int word = 0; word < kRunWords; ++word) {
    uint64_t diff = old_bits[word] ^ new_bits[word];
    while (diff != 0u) {
      int bit = __builtin_ctzll(diff);
      int index = word * 64 + bit;
      if (index >= nnue::kNnueRunFeaturesPerSide) {
        break;
      }
      int feature = run_offset + index;
      if (new_bits[word] & (uint64_t{1} << bit)) {
        AddFeature(feature, add);
      } else {
        AddFeature(feature, remove);
      }
      diff &= diff - 1;
    }
  }
}

void UpdateCacheNoFlipSide(int side,
                           const PointDeltaList& delta,
                           int off_delta, nnue::NnueCache* cache,
                           nnue::NnueActiveFeatures* remove,
                           nnue::NnueActiveFeatures* add) {
  uint32_t old_bits = cache->blocked_bits[side];
  uint32_t new_bits = old_bits;
  auto old_run_bits = cache->run_bits[side];
  for (int i = 0; i < delta.count; ++i) {
    int point = delta.points[i];
    int delta_count = delta.deltas[i];
    int old_count = cache->counts[side][point];
    int new_count = old_count + delta_count;
    cache->counts[side][point] = static_cast<uint8_t>(new_count);
    int old_feat = BaseFeatureIndex(side, point, old_count);
    int new_feat = BaseFeatureIndex(side, point, new_count);
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
    }
  }

  if (off_delta != 0) {
    int old_off = cache->off[side];
    int new_off = old_off + off_delta;
    cache->off[side] = static_cast<uint8_t>(new_off);
    int old_feat = OffFeatureIndex(side, old_off);
    int new_feat = OffFeatureIndex(side, new_off);
    if (old_feat != new_feat) {
      AddFeature(old_feat, remove);
      AddFeature(new_feat, add);
    }
  }

  cache->blocked_bits[side] = new_bits;
  UpdateRunBitsIncremental(old_bits, new_bits, &cache->run_bits[side]);
  if (cache->run_bits[side] != old_run_bits) {
    int run_offset =
        nnue::kNnueBaseFeatures + side * nnue::kNnueRunFeaturesPerSide;
    UpdateRunFeaturesDelta(run_offset, old_run_bits, cache->run_bits[side],
                           remove, add);
  }
}

void FlipCache(const nnue::NnueNetwork& net, const nnue::NnueCache& src,
               nnue::NnueCache* dst) {
  for (int point = 0; point < kNumPoints; ++point) {
    int rot = (point + 12) % kNumPoints;
    dst->counts[0][point] = src.counts[1][rot];
    dst->counts[1][point] = src.counts[0][rot];
  }
  dst->off[0] = src.off[1];
  dst->off[1] = src.off[0];
  dst->blocked_bits[0] = BlockedBits(dst->counts[0]);
  dst->blocked_bits[1] = BlockedBits(dst->counts[1]);
  dst->run_bits[0] = ComputeRunBits(dst->blocked_bits[0]);
  dst->run_bits[1] = ComputeRunBits(dst->blocked_bits[1]);
  RebuildAccumulatorFromCache(net, *dst, &dst->acc);
}

}  // namespace

void NnueCacheStack::Reset(const LongNardeState& state,
                           const nnue::NnueNetwork* network) {
  stack_.clear();
  network_ = network;
  if (network_ == nullptr) {
    return;
  }
  CachePair root;
  InitCacheFromState(state, *network_, &root.cur);
  FlipCache(*network_, root.cur, &root.opp);
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
  CachePair child = stack_.back();
  nnue::NnueActiveFeatures remove;
  nnue::NnueActiveFeatures add;
  remove.count = 0;
  add.count = 0;

  internal::DecodedAction decoded = internal::DecodeAction(action);
  int d_min = std::min(dice[0], dice[1]);
  int d_max = std::max(dice[0], dice[1]);
  int d1 = (decoded.order == 0) ? d_max : d_min;
  int d2 = (decoded.order == 0) ? d_min : d_max;

  PointDeltaList delta;
  PointDeltaList delta_opp;
  int off_delta = 0;
  auto add_delta = [&](int point, int8_t delta_count) {
    AddPointDelta(&delta, point, delta_count);
    int rot = (point + 12) % kNumPoints;
    AddPointDelta(&delta_opp, rot, delta_count);
  };
  auto apply = [&](int src, int die) {
    if (src == kActionPassSrc) {
      return;
    }
    int src_read = std::min(src, kNumPoints - 1);
    add_delta(src_read, -1);
    int target = src + die;
    if (target < kNumPoints) {
      add_delta(target, 1);
    } else {
      off_delta += 1;
    }
  };

  apply(decoded.src1, d1);
  apply(decoded.src2, d2);

  bool flipped = state.current_player_id() != prev_player;

  UpdateCacheNoFlipSide(0, delta, off_delta, &child.cur, &remove, &add);
  nnue::ApplyAccumulatorDelta(*network_, remove, add, &child.cur.acc);

  remove.count = 0;
  add.count = 0;
  UpdateCacheNoFlipSide(1, delta_opp, off_delta, &child.opp, &remove, &add);
  nnue::ApplyAccumulatorDelta(*network_, remove, add, &child.opp.acc);

  if (flipped) {
    std::swap(child.cur, child.opp);
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
  return &stack_.back().cur;
}

}  // namespace long_narde
}  // namespace open_spiel
