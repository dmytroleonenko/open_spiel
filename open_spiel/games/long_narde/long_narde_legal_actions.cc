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

namespace open_spiel {
namespace long_narde {
namespace internal {

namespace {

constexpr uint32_t kLow6 = (uint32_t{1} << 6) - 1u;
constexpr uint32_t kLow12 = (uint32_t{1} << 12) - 1u;
constexpr uint32_t kNonHomeMaskRot = kLow6 | (kLow12 << 12);

template <size_t N>
bool AnyBits(const std::array<uint32_t, N>& bits) {
  for (uint32_t w : bits) {
    if (w != 0u) {
      return true;
    }
  }
  return false;
}

template <size_t N>
std::array<uint32_t, N> AndBits(const std::array<uint32_t, N>& a,
                                const std::array<uint32_t, N>& b) {
  std::array<uint32_t, N> out{};
  for (size_t i = 0; i < N; ++i) {
    out[i] = a[i] & b[i];
  }
  return out;
}

template <size_t N>
void AndBitsInplace(std::array<uint32_t, N>* a,
                    const std::array<uint32_t, N>& b) {
  for (size_t i = 0; i < N; ++i) {
    (*a)[i] &= b[i];
  }
}

bool HeadLimitOk(int head_used_total, int head_count, bool is_first,
                 bool is_doubles, const Dice& dice) {
  bool special_first = is_first && is_doubles &&
                       (dice[0] == 3 || dice[0] == 4 || dice[0] == 6);
  int threshold = special_first ? 2 : (1 - head_count);
  return head_used_total <= threshold;
}

OrderBitset MaskForOrder(const Board& board, int d1, int d2, int head_count,
                         bool is_doubles, bool is_first_turn,
                         const Dice& dice) {
  OrderBitset out{};
  out.fill(0u);

  const auto& my0 = board[0];
  const auto& opp0 = board[1];
  const auto& canon_to_rot = CanonToRotBits();

  int total0 = 0;
  for (int v : my0) {
    total0 += v;
  }

  uint32_t my_occ_rot = Rot12Bits(OccBits(my0));
  auto opp_ctx = OppNoAheadBits(opp0);
  bool opp_has = opp_ctx.first;
  uint32_t no_opp_ahead_bits = opp_ctx.second;

  bool all_home0 = (my_occ_rot & kNonHomeMaskRot) == 0u;
  int home_min0 = HomeMinFromRotOccBits(my_occ_rot);

  std::array<bool, 25> legal_1{};
  std::array<bool, 25> valid_1{};
  std::array<uint32_t, 25> occ_m1_bits{};
  std::array<int, 25> target_1{};
  std::array<bool, 25> is_pass_1{};
  std::array<bool, 25> is_bearing_1{};
  std::array<int, 25> total_1{};
  std::array<bool, 25> all_home_1{};
  std::array<int, 25> home_min_1{};

  for (int src1 = 0; src1 < 25; ++src1) {
    bool is_pass = (src1 == kActionPassSrc);
    int target = src1 + d1;
    int src_read = std::min(src1, 23);
    bool has_checker = is_pass ? true : (my0[src_read] > 0);

    bool is_bearing = target >= 24;
    int dst_read = std::min(target, 23);
    bool dest_blocked =
        (target < 24) ? (opp0[dst_read] > 0) : false;

    bool overkill_ok = (src1 == home_min0);
    bool exact_bear = (target == 24);
    bool can_bear = (!is_pass) && is_bearing && has_checker && all_home0 &&
                    (exact_bear || overkill_ok);

    bool home_shuffle_block = all_home0 && (total0 == 1);
    bool can_move_std = (!is_pass) && (!is_bearing) && has_checker &&
                        (!dest_blocked) && (!home_shuffle_block);
    bool legal = is_pass || can_move_std || can_bear;

    uint32_t bits = my_occ_rot;
    if (legal && !is_pass) {
      uint32_t src_bit = canon_to_rot[src_read];
      uint32_t dst_bit = canon_to_rot[dst_read];
      bool src_empty = my0[src_read] == 1;
      if (src_empty) {
        bits &= ~src_bit;
      }
      if (target < 24) {
        bits |= dst_bit;
      }
    }

    bool block_ok = BridgeLegal(bits, opp_has, no_opp_ahead_bits);
    bool valid = legal && block_ok;

    legal_1[src1] = legal;
    valid_1[src1] = valid;
    occ_m1_bits[src1] = bits;
    target_1[src1] = target;
    is_pass_1[src1] = is_pass;
    is_bearing_1[src1] = is_bearing;
    total_1[src1] =
        total0 - ((valid && !is_pass && is_bearing) ? 1 : 0);
    all_home_1[src1] = (bits & kNonHomeMaskRot) == 0u;
    home_min_1[src1] = HomeMinFromRotOccBits(bits);
  }

  for (int src2 = 0; src2 < 25; ++src2) {
    bool is_pass2 = (src2 == kActionPassSrc);
    int target2 = src2 + d2;
    int src2_read = std::min(src2, 23);
    int dst2_read = std::min(target2, 23);
    uint32_t src2_bit = canon_to_rot[src2_read];
    uint32_t dst2_bit = canon_to_rot[dst2_read];
    int my0_src2 = my0[src2_read];

    bool is_bearing2 = target2 >= 24;
    bool dest_blocked2 =
        (target2 < 24) ? (opp0[dst2_read] > 0) : false;
    bool exact_bear2 = target2 == 24;

    for (int src1 = 0; src1 < 25; ++src1) {
      bool moved_from_src2 =
          valid_1[src1] && !is_pass_1[src1] && (src1 == src2);
      bool moved_to_src2 =
          valid_1[src1] && !is_pass_1[src1] && (target_1[src1] == src2) &&
          (target_1[src1] < 24);
      int count_at_src2 =
          my0_src2 + (moved_to_src2 ? 1 : 0) - (moved_from_src2 ? 1 : 0);

      bool has_checker2 = is_pass2 || (count_at_src2 > 0);
      bool overkill_ok2 = (src2 == home_min_1[src1]);
      bool can_bear2 = (!is_pass2) && is_bearing2 && has_checker2 &&
                       all_home_1[src1] && (exact_bear2 || overkill_ok2);
      bool home_shuffle_block2 = all_home_1[src1] && (total_1[src1] == 1);
      bool can_move_std2 = (!is_pass2) && (!is_bearing2) && has_checker2 &&
                           (!dest_blocked2) && (!home_shuffle_block2);
      bool legal_2 = is_pass2 || can_move_std2 || can_bear2;

      bool valid_2 = false;
      if (legal_2) {
        uint32_t bits2 = occ_m1_bits[src1];
        if (!is_pass2) {
          if (count_at_src2 == 1) {
            bits2 &= ~src2_bit;
          }
          if (target2 < 24) {
            bits2 |= dst2_bit;
          }
        }
        valid_2 = BridgeLegal(bits2, opp_has, no_opp_ahead_bits);
      }

      if (valid_1[src1] && valid_2) {
        int head_used_total = (src1 == 0) + (src2 == 0);
        if (HeadLimitOk(head_used_total, head_count, is_first_turn,
                        is_doubles, dice)) {
          int idx = src1 + 25 * src2;
          int word = idx / 32;
          int bit = idx % 32;
          out[word] |= (uint32_t{1} << bit);
        }
      }
    }
  }

  return out;
}

}  // namespace

ActionBitset ApplyForceRulesBits(ActionBitset bits, const Dice& dice,
                                 bool keep_pass_order1, bool* has_move) {
  const ActionTables& tables = GetActionTables();
  int d_min = std::min(dice[0], dice[1]);
  int d_max = std::max(dice[0], dice[1]);

  if (keep_pass_order1) {
    for (size_t i = 0; i < bits.size(); ++i) {
      bits[i] &= ~tables.pass_action_order0_bits[i];
    }
  } else {
    for (size_t i = 0; i < bits.size(); ++i) {
      bits[i] &= ~tables.pass_action_order1_bits[i];
    }
  }

  for (size_t i = 0; i < bits.size(); ++i) {
    bits[i] &= ~tables.pass_first_one_die_bits[i];
  }

  ActionBitset bits_move = AndBits(bits, tables.action_uses_move_bits);
  ActionBitset bits_two = AndBits(bits, tables.action_uses_two_bits);
  ActionBitset bits_one = AndBits(bits, tables.action_uses_one_bits);
  ActionBitset bits_high = AndBits(bits_one, tables.action_uses_high_bits);

  bool has_move_flag = AnyBits(bits_move);
  bool has_two = AnyBits(bits_two);
  bool has_one = AnyBits(bits_one);
  bool high_exists = AnyBits(bits_high);

  if (has_move_flag) {
    AndBitsInplace(&bits, tables.action_uses_move_bits);
  }
  if (has_two) {
    AndBitsInplace(&bits, tables.action_uses_two_bits);
  }
  bool need_high = (!has_two) && (d_max != d_min) && has_one;
  if (need_high && high_exists) {
    AndBitsInplace(&bits, tables.action_uses_high_bits);
  }

  *has_move = has_move_flag;
  return bits;
}

ActionBitset GenerateLegalActionBits(const Board& board, const Dice& dice,
                                     int head_count, bool is_doubles,
                                     bool is_first_turn) {
  int d_min = std::min(dice[0], dice[1]);
  int d_max = std::max(dice[0], dice[1]);

  OrderBitset order0 = MaskForOrder(board, d_max, d_min, head_count, is_doubles,
                                    is_first_turn, dice);
  OrderBitset order1 = MaskForOrder(board, d_min, d_max, head_count, is_doubles,
                                    is_first_turn, dice);

  ActionBitset bits{};
  bits.fill(0u);
  for (int i = 0; i < kOrderBitsetWords; ++i) {
    bits[i] = order0[i];
  }
  // Order1 actions start at bit 625, not a word boundary.
  constexpr int kOrderShiftWords = kOrderActionCount / 32;
  constexpr int kOrderShiftBits = kOrderActionCount % 32;
  for (int i = 0; i < kOrderBitsetWords; ++i) {
    uint32_t word = order1[i];
    if (word == 0u) {
      continue;
    }
    int out = i + kOrderShiftWords;
    bits[out] |= word << kOrderShiftBits;
    if (kOrderShiftBits != 0 && out + 1 < kActionBitsetWords) {
      bits[out + 1] |= word >> (32 - kOrderShiftBits);
    }
  }

  if (is_doubles) {
    AndBitsInplace(&bits, GetActionTables().order0_only_bits);
  }

  return bits;
}

}  // namespace internal
}  // namespace long_narde
}  // namespace open_spiel
