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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_INTERNAL_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_INTERNAL_H_

#include <array>
#include <cstdint>
#include <utility>

#include "open_spiel/games/long_narde/long_narde.h"

namespace open_spiel {
namespace long_narde {
namespace internal {

constexpr int kOrderActionCount = 625;
constexpr int kOrderBitsetWords = (kOrderActionCount + 31) / 32;
constexpr int kActionBitsetWords = (kNumDistinctActions + 31) / 32;

using Board = std::array<std::array<int, kNumPoints>, 2>;
using Dice = std::array<int, 2>;
using ActionBitset = std::array<uint32_t, kActionBitsetWords>;
using OrderBitset = std::array<uint32_t, kOrderBitsetWords>;

struct ActionTables {
  std::array<uint8_t, kNumDistinctActions> src1;
  std::array<uint8_t, kNumDistinctActions> src2;
  std::array<uint8_t, kNumDistinctActions> dice_used;
  std::array<uint8_t, kNumDistinctActions> uses_high;
  ActionBitset pass_first_one_die_bits;
  ActionBitset action_uses_move_bits;
  ActionBitset action_uses_two_bits;
  ActionBitset action_uses_one_bits;
  ActionBitset action_uses_high_bits;
  ActionBitset pass_action_order0_bits;
  ActionBitset pass_action_order1_bits;
  ActionBitset order0_only_bits;
};

const ActionTables& GetActionTables();
const std::array<std::array<int, 2>, kNumChanceOutcomes>&
ChanceOutcomeValues();

struct DecodedAction {
  int src1;
  int src2;
  int order;
};

DecodedAction DecodeAction(Action action);

uint32_t OccBits(const std::array<int, kNumPoints>& row);
uint32_t Rot12Bits(uint32_t bits);
const std::array<uint32_t, kNumPoints>& CanonToRotBits();
void FlipBoard(Board* board);

std::pair<bool, uint32_t> OppNoAheadBits(
    const std::array<int, kNumPoints>& row);
bool BridgeLegal(uint32_t occ_bits_rot, bool opp_has,
                 uint32_t no_opp_ahead_bits);
int HomeMinFromRotOccBits(uint32_t occ_bits_rot);

ActionBitset ApplyForceRulesBits(ActionBitset bits, const Dice& dice,
                                 bool keep_pass_order1, bool* has_move);
ActionBitset GenerateLegalActionBits(const Board& board, const Dice& dice,
                                     int head_count, bool is_doubles,
                                     bool is_first_turn);

}  // namespace internal
}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_INTERNAL_H_
