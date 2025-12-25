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

namespace open_spiel {
namespace long_narde {
namespace internal {

namespace {

constexpr uint32_t kAllOnes = 0xFFFFFFFFu;

ActionTables BuildActionTables() {
  ActionTables t{};
  t.pass_first_one_die_bits.fill(0);
  t.action_uses_move_bits.fill(0);
  t.action_uses_two_bits.fill(0);
  t.action_uses_one_bits.fill(0);
  t.action_uses_high_bits.fill(0);
  t.pass_action_order0_bits.fill(0);
  t.pass_action_order1_bits.fill(0);
  t.order0_only_bits.fill(0);

  for (int a = 0; a < kNumDistinctActions; ++a) {
    int rem = a % kOrderActionCount;
    int src1 = rem % 25;
    int src2 = rem / 25;
    bool order_is_low_first = a >= kOrderActionCount;
    int dice_used = (src1 != kActionPassSrc) + (src2 != kActionPassSrc);
    bool uses_move = dice_used > 0;
    bool uses_two = dice_used == 2;
    bool uses_one = dice_used == 1;
    bool uses_high =
        (!order_is_low_first && src1 != kActionPassSrc) ||
        (order_is_low_first && src2 != kActionPassSrc);

    t.src1[a] = static_cast<uint8_t>(src1);
    t.src2[a] = static_cast<uint8_t>(src2);
    t.dice_used[a] = static_cast<uint8_t>(dice_used);
    t.uses_high[a] = static_cast<uint8_t>(uses_high);

    int word = a / 32;
    int bit = a % 32;
    uint32_t mask = uint32_t{1} << bit;
    if (dice_used == 1 && src1 == kActionPassSrc) {
      t.pass_first_one_die_bits[word] |= mask;
    }
    if (uses_move) {
      t.action_uses_move_bits[word] |= mask;
    }
    if (uses_two) {
      t.action_uses_two_bits[word] |= mask;
    }
    if (uses_one) {
      t.action_uses_one_bits[word] |= mask;
    }
    if (uses_high) {
      t.action_uses_high_bits[word] |= mask;
    }
    if (a == kOrderActionCount - 1) {
      t.pass_action_order0_bits[word] |= mask;
    }
    if (a == kNumDistinctActions - 1) {
      t.pass_action_order1_bits[word] |= mask;
    }
  }

  for (int w = 0; w < kActionBitsetWords; ++w) {
    if (w < 19) {
      t.order0_only_bits[w] = kAllOnes;
    } else if (w == 19) {
      t.order0_only_bits[w] = (uint32_t{1} << 17) - 1u;
    } else {
      t.order0_only_bits[w] = 0u;
    }
  }

  return t;
}

}  // namespace

const ActionTables& GetActionTables() {
  static const ActionTables kTables = BuildActionTables();
  return kTables;
}

const std::array<std::array<int, 2>, kNumChanceOutcomes>&
ChanceOutcomeValues() {
  static const std::array<std::array<int, 2>, kNumChanceOutcomes> kValues = {{
      {1, 2}, {2, 1}, {1, 3}, {3, 1}, {1, 4}, {4, 1},
      {1, 5}, {5, 1}, {1, 6}, {6, 1}, {2, 3}, {3, 2},
      {2, 4}, {4, 2}, {2, 5}, {5, 2}, {2, 6}, {6, 2},
      {3, 4}, {4, 3}, {3, 5}, {5, 3}, {3, 6}, {6, 3},
      {4, 5}, {5, 4}, {4, 6}, {6, 4}, {5, 6}, {6, 5},
      {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6},
  }};
  return kValues;
}

}  // namespace internal
}  // namespace long_narde
}  // namespace open_spiel
