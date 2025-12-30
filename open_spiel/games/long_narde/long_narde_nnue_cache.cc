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
#include "open_spiel/games/long_narde/long_narde_nnue_features.h"
#include "open_spiel/games/long_narde/long_narde_nnue_kernels.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

constexpr int kRunWords = 2;
constexpr int kMaxDeltaPoints = 4;
constexpr int kRunOffset2 = 0;
constexpr int kRunOffset3 = kRunOffset2 + nnue::kNnueRunCount2;
constexpr int kRunOffset4 = kRunOffset3 + nnue::kNnueRunCount3;
constexpr int kRunOffset5 = kRunOffset4 + nnue::kNnueRunCount4;
constexpr int kRunOffset6 = kRunOffset5 + nnue::kNnueRunCount5;

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

struct RunStarts {
  uint32_t run2 = 0;
  uint32_t run3 = 0;
  uint32_t run4 = 0;
  uint32_t run5 = 0;
  uint32_t run6 = 0;
};

RunStarts ComputeRunStarts(uint32_t blocked) {
  RunStarts out;
  out.run2 = blocked & (blocked >> 1);
  out.run3 = out.run2 & (blocked >> 2);
  out.run4 = out.run3 & (blocked >> 3);
  out.run5 = out.run4 & (blocked >> 4);
  out.run6 = out.run5 & (blocked >> 5);
  return out;
}

void UpdateRunStarts(int run_offset, int index_offset, uint32_t old_run,
                     uint32_t new_run, const int16_t* w0,
                     std::array<int16_t, nnue::kNnueL1>* acc,
                     std::array<uint64_t, kRunWords>* run_bits,
                     nnue::AddRowFn add_row, nnue::SubRowFn sub_row) {
  uint32_t diff = old_run ^ new_run;
  while (diff != 0u) {
    int bit = __builtin_ctz(diff);
    int index = index_offset + bit;
    const int16_t* row = &w0[(run_offset + index) * nnue::kNnueL1];
    if (new_run & (uint32_t{1} << bit)) {
      add_row(row, acc->data());
      SetRunBitValue(run_bits, index, true);
    } else {
      sub_row(row, acc->data());
      SetRunBitValue(run_bits, index, false);
    }
    diff &= diff - 1;
  }
}

internal::Board BoardFromCounts(
    const std::array<std::array<uint8_t, kNumPoints>, 2>& counts) {
  internal::Board board{};
  for (int side = 0; side < kNumPlayers; ++side) {
    for (int point = 0; point < kNumPoints; ++point) {
      board[side][point] = counts[side][point];
    }
  }
  return board;
}

void SetExtraBuckets(nnue::NnueCache* cache) {
  internal::Board board = BoardFromCounts(cache->counts);
  cache->pip_delta_bucket = nnue::PipDeltaBucketFromBoard(board);
  cache->mobility_bucket = nnue::MobilityBucketFromBoard(board);
  internal::Board opp_board = board;
  internal::FlipBoard(&opp_board);
  cache->opp_mobility_bucket = nnue::MobilityBucketFromBoard(opp_board);
}

void UpdateExtraFeatures(const nnue::NnueNetwork& net, nnue::NnueCache* cache,
                         nnue::AddRowFn add_row, nnue::SubRowFn sub_row) {
  internal::Board board = BoardFromCounts(cache->counts);
  int pip_bucket = nnue::PipDeltaBucketFromBoard(board);
  int mobility_bucket = nnue::MobilityBucketFromBoard(board);
  internal::Board opp_board = board;
  internal::FlipBoard(&opp_board);
  int opp_mobility_bucket = nnue::MobilityBucketFromBoard(opp_board);
  auto update_bucket = [&](int* cached, int offset, int next) {
    if (*cached == next) {
      return;
    }
    const int16_t* row_old = &net.w0[(offset + *cached) * nnue::kNnueL1];
    const int16_t* row_new = &net.w0[(offset + next) * nnue::kNnueL1];
    sub_row(row_old, cache->acc.data());
    add_row(row_new, cache->acc.data());
    *cached = next;
  };
  update_bucket(&cache->pip_delta_bucket, nnue::kNnuePipDeltaOffset,
                pip_bucket);
  update_bucket(&cache->mobility_bucket, nnue::kNnueMobilityOffset,
                mobility_bucket);
  update_bucket(&cache->opp_mobility_bucket, nnue::kNnueOppMobilityOffset,
                opp_mobility_bucket);
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
  AddFeature(nnue::kNnuePipDeltaOffset + cache.pip_delta_bucket, active);
  AddFeature(nnue::kNnueMobilityOffset + cache.mobility_bucket, active);
  AddFeature(nnue::kNnueOppMobilityOffset + cache.opp_mobility_bucket, active);
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
  SetExtraBuckets(cache);
}

void UpdateCacheNoFlipSide(int side,
                           const PointDeltaList& delta,
                           int off_delta, const nnue::NnueNetwork& net,
                           nnue::NnueCache* cache, nnue::AddRowFn add_row,
                           nnue::SubRowFn sub_row) {
  const int16_t* w0 = net.w0.data();
  uint32_t old_bits = cache->blocked_bits[side];
  uint32_t new_bits = old_bits;
  int base_offset = side * nnue::kNnueBaseFeaturesPerSide;
  int run_offset =
      nnue::kNnueBaseFeatures + side * nnue::kNnueRunFeaturesPerSide;
  for (int i = 0; i < delta.count; ++i) {
    int point = delta.points[i];
    int delta_count = delta.deltas[i];
    uint8_t old_count = cache->counts[side][point];
    int new_count = old_count + delta_count;
    cache->counts[side][point] = static_cast<uint8_t>(new_count);
    int old_feat =
        base_offset + point * nnue::kNnueBucketCount + old_count;
    int new_feat =
        base_offset + point * nnue::kNnueBucketCount + new_count;
    if (old_feat != new_feat) {
      const int16_t* row_old = &w0[old_feat * nnue::kNnueL1];
      const int16_t* row_new = &w0[new_feat * nnue::kNnueL1];
      sub_row(row_old, cache->acc.data());
      add_row(row_new, cache->acc.data());
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
    uint8_t old_off = cache->off[side];
    int new_off = old_off + off_delta;
    cache->off[side] = static_cast<uint8_t>(new_off);
    int old_feat =
        base_offset + kNumPoints * nnue::kNnueBucketCount + old_off;
    int new_feat =
        base_offset + kNumPoints * nnue::kNnueBucketCount + new_off;
    if (old_feat != new_feat) {
      const int16_t* row_old = &w0[old_feat * nnue::kNnueL1];
      const int16_t* row_new = &w0[new_feat * nnue::kNnueL1];
      sub_row(row_old, cache->acc.data());
      add_row(row_new, cache->acc.data());
    }
  }

  cache->blocked_bits[side] = new_bits;
  if (old_bits == new_bits) {
    return;
  }
  RunStarts old_runs = ComputeRunStarts(old_bits);
  RunStarts new_runs = ComputeRunStarts(new_bits);
  UpdateRunStarts(run_offset, kRunOffset2, old_runs.run2, new_runs.run2, w0,
                  &cache->acc, &cache->run_bits[side], add_row, sub_row);
  UpdateRunStarts(run_offset, kRunOffset3, old_runs.run3, new_runs.run3, w0,
                  &cache->acc, &cache->run_bits[side], add_row, sub_row);
  UpdateRunStarts(run_offset, kRunOffset4, old_runs.run4, new_runs.run4, w0,
                  &cache->acc, &cache->run_bits[side], add_row, sub_row);
  UpdateRunStarts(run_offset, kRunOffset5, old_runs.run5, new_runs.run5, w0,
                  &cache->acc, &cache->run_bits[side], add_row, sub_row);
  UpdateRunStarts(run_offset, kRunOffset6, old_runs.run6, new_runs.run6, w0,
                  &cache->acc, &cache->run_bits[side], add_row, sub_row);
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
  SetExtraBuckets(dst);
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
  nnue::AddRowFn add_row = nnue::GetAddRowKernel();
  nnue::SubRowFn sub_row = nnue::GetSubRowKernel();

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

  UpdateCacheNoFlipSide(0, delta, off_delta, *network_, &child.cur, add_row,
                        sub_row);
  UpdateCacheNoFlipSide(1, delta_opp, off_delta, *network_, &child.opp, add_row,
                        sub_row);
  UpdateExtraFeatures(*network_, &child.cur, add_row, sub_row);
  UpdateExtraFeatures(*network_, &child.opp, add_row, sub_row);

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
