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

#include "open_spiel/games/long_narde/long_narde_nnue_features.h"

#include <algorithm>

namespace open_spiel {
namespace long_narde {
namespace nnue {
namespace {

int PipCountRow(const std::array<int, kNumPoints>& row, bool rotated) {
  int total = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    int idx = rotated ? ((i + 12) % kNumPoints) : i;
    int dist = kNumPoints - idx;
    total += row[i] * dist;
  }
  return total;
}

int CountBits(const internal::ActionBitset& bits) {
  int total = 0;
  for (uint32_t word : bits) {
    total += __builtin_popcount(word);
  }
  return total;
}

int LegalActionCount(const internal::Board& board,
                     const internal::Dice& dice) {
  const internal::ActionTables& tables = internal::GetActionTables();
  internal::ActionBitset raw_bits = internal::GenerateLegalActionBits(
      board, dice, 0, dice[0] == dice[1], false);
  bool keep_pass_order1 = dice[0] < dice[1];
  bool has_move = false;
  internal::ActionBitset forced_bits = internal::ApplyForceRulesBits(
      raw_bits, dice, keep_pass_order1, &has_move);
  internal::ActionBitset pass_bits =
      keep_pass_order1 ? tables.pass_action_order1_bits
                       : tables.pass_action_order0_bits;
  const internal::ActionBitset& out_bits = has_move ? forced_bits : pass_bits;
  return CountBits(out_bits);
}

int ExpectedLegalActions(const internal::Board& board) {
  int total = 0;
  for (int d1 = 1; d1 <= 6; ++d1) {
    for (int d2 = 1; d2 <= d1; ++d2) {
      internal::Dice dice = {d1, d2};
      int weight = (d1 == d2) ? 1 : 2;
      total += weight * LegalActionCount(board, dice);
    }
  }
  return (total + kNumChanceOutcomes / 2) / kNumChanceOutcomes;
}

}  // namespace

int PipDeltaBucketFromBoard(const internal::Board& board) {
  int pip_self = PipCountRow(board[0], false);
  int pip_opp = PipCountRow(board[1], true);
  int delta = pip_self - pip_opp;
  delta = std::max(-kNnuePipDeltaMax, std::min(kNnuePipDeltaMax, delta));
  int shifted = delta + kNnuePipDeltaMax;
  return shifted / kNnuePipDeltaBucketWidth;
}

int MobilityBucketFromBoard(const internal::Board& board) {
  int expected = ExpectedLegalActions(board);
  expected = std::min(kNnueMobilityMax, std::max(0, expected));
  return expected / kNnueMobilityBucketWidth;
}

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel
