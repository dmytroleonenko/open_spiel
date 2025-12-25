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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_TEST_COMMON_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_TEST_COMMON_H_

#include <algorithm>
#include <array>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace testing {

inline int CanonicalSrc(int real_src, Player player) {
  if (real_src == kPassPos) {
    return kActionPassSrc;
  }
  if (player == kXPlayerId) {
    return 23 - real_src;
  }
  return (11 - real_src + 24) % 24;
}

inline std::array<int, kNumPoints> ToCanonicalRow(
    const std::vector<int>& real, Player player) {
  SPIEL_CHECK_EQ(real.size(), kNumPoints);
  std::array<int, kNumPoints> out{};
  for (int r = 0; r < kNumPoints; ++r) {
    int c = CanonicalSrc(r, player);
    if (c != kActionPassSrc) {
      out[c] = real[r];
    }
  }
  return out;
}

inline std::array<int, kNumPoints> ToOppRelativeRow(
    const std::vector<int>& real, Player opponent_player) {
  std::array<int, kNumPoints> opp_can = ToCanonicalRow(real, opponent_player);
  std::array<int, kNumPoints> opp_rel{};
  for (int i = 0; i < kNumPoints; ++i) {
    opp_rel[i] = opp_can[(i + 12) % 24];
  }
  return opp_rel;
}

inline std::array<std::array<int, kNumPoints>, 2> CanonicalBoardFromReal(
    const std::vector<std::vector<int>>& board_real, Player current_player) {
  SPIEL_CHECK_EQ(board_real.size(), kNumPlayers);
  std::array<std::array<int, kNumPoints>, 2> board{};
  board[0] = ToCanonicalRow(board_real[current_player], current_player);
  board[1] = ToOppRelativeRow(board_real[1 - current_player],
                             1 - current_player);
  return board;
}

inline Action EncodeAction(int src1, int src2, int order) {
  return src1 + 25 * src2 + (order ? 625 : 0);
}

inline Action EncodeActionFromMoves(const std::array<int, 2>& dice, int src1,
                                    int src2, int die1, int die2) {
  int d_min = std::min(dice[0], dice[1]);
  int d_max = std::max(dice[0], dice[1]);
  int order = (die1 == d_min && d_max != d_min) ? 1 : 0;
  return EncodeAction(src1, src2, order);
}

inline void SetupStateFromRealBoard(LongNardeState* state,
                                    const std::vector<std::vector<int>>& board,
                                    Player current_player,
                                    const std::array<int, 2>& dice,
                                    bool is_first_turn, int head_moved_count,
                                    int phase) {
  auto canonical = CanonicalBoardFromReal(board, current_player);
  state->SetStateForTesting(canonical, current_player, dice,
                            /*awaiting_roll=*/false,
                            /*initial_roll=*/false, is_first_turn,
                            head_moved_count, phase);
}

inline bool ActionsContain(const std::vector<Action>& actions, Action action) {
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

}  // namespace testing

void TestMovementRules();
void TestActionEncoding();
void TestBridgeFormation();
void TestEndgame();
void TestPassMoveBehavior();
void TestSearchInvariants();

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_TEST_COMMON_H_
