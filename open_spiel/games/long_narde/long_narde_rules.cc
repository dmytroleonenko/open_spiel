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

#include "open_spiel/games/long_narde/long_narde_internal.h"

#include <algorithm>
#include <utility>

namespace open_spiel {
namespace long_narde {
namespace internal {

namespace {

constexpr uint32_t kMask24 = (uint32_t{1} << 24) - 1u;
constexpr uint32_t kLow6 = (uint32_t{1} << 6) - 1u;
constexpr uint32_t kLow12 = (uint32_t{1} << 12) - 1u;
constexpr uint32_t kMask19 = (uint32_t{1} << 19) - 1u;
constexpr uint32_t kNonHomeMaskRot = kLow6 | (kLow12 << 12);

std::array<int, 64> BuildHomeMask6ToHomeMin() {
  std::array<int, 64> table{};
  for (int mask = 0; mask < 64; ++mask) {
    if (mask == 0) {
      table[mask] = 99;
      continue;
    }
    int min_idx = 0;
    for (; min_idx < 6; ++min_idx) {
      if ((mask >> min_idx) & 1) {
        break;
      }
    }
    table[mask] = 18 + min_idx;
  }
  return table;
}

const std::array<int, 64>& HomeMask6ToHomeMin() {
  static const std::array<int, 64> kTable = BuildHomeMask6ToHomeMin();
  return kTable;
}

}  // namespace

uint32_t OccBits(const std::array<int, kNumPoints>& row) {
  uint32_t bits = 0u;
  for (int i = 0; i < kNumPoints; ++i) {
    if (row[i] > 0) {
      bits |= (uint32_t{1} << i);
    }
  }
  return bits;
}

uint32_t Rot12Bits(uint32_t bits) {
  bits &= kMask24;
  return ((bits & kLow12) << 12) | (bits >> 12);
}

const std::array<uint32_t, kNumPoints>& CanonToRotBits() {
  static const std::array<uint32_t, kNumPoints> kBits = []() {
    std::array<uint32_t, kNumPoints> out{};
    for (int i = 0; i < kNumPoints; ++i) {
      out[i] = uint32_t{1} << ((i + 12) % 24);
    }
    return out;
  }();
  return kBits;
}

void FlipBoard(Board* board) {
  Board next{};
  const auto& row0 = (*board)[0];
  const auto& row1 = (*board)[1];
  std::copy_n(row1.begin() + 12, 12, next[0].begin());
  std::copy_n(row1.begin(), 12, next[0].begin() + 12);
  std::copy_n(row0.begin() + 12, 12, next[1].begin());
  std::copy_n(row0.begin(), 12, next[1].begin() + 12);
  *board = next;
}

std::pair<bool, uint32_t> OppNoAheadBits(
    const std::array<int, kNumPoints>& row) {
  uint32_t opp_bits = Rot12Bits(OccBits(row));
  bool opp_has = opp_bits != 0u;
  if (!opp_has) {
    return {false, 0u};
  }
  int msb = 31 - __builtin_clz(opp_bits);
  int thr = std::max(0, msb - 5);
  uint32_t no_opp_ahead = (kMask19 >> thr) << thr;
  return {true, no_opp_ahead & kMask19};
}

bool BridgeLegal(uint32_t occ_bits_rot, bool opp_has,
                 uint32_t no_opp_ahead_bits) {
  if (!opp_has) {
    return true;
  }
  uint32_t my_bits = occ_bits_rot & kMask24;
  uint32_t c3 = my_bits & (my_bits >> 1) & (my_bits >> 2);
  uint32_t c6 = c3 & (c3 >> 3);
  uint32_t block_starts = c6 & kMask19;
  return (block_starts & no_opp_ahead_bits) == 0u;
}

int HomeMinFromRotOccBits(uint32_t occ_bits_rot) {
  uint32_t home_mask6 = (occ_bits_rot >> 6) & 0x3Fu;
  return HomeMask6ToHomeMin()[home_mask6];
}

}  // namespace internal
}  // namespace long_narde
}  // namespace open_spiel
