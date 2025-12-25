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

#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

bool HasBearOffMove(LongNardeState* state, Player player) {
  for (Action action : state->LegalActions()) {
    auto moves = state->LongNardeSpielMoveToCheckerMoves(player, action);
    for (const auto& move : moves) {
      if (state->IsOff(player, move.to_pos)) {
        return true;
      }
    }
  }
  return false;
}

bool HasMove(LongNardeState* state, Player player, int from_pos, int to_pos,
             int die) {
  for (Action action : state->LegalActions()) {
    auto moves = state->LongNardeSpielMoveToCheckerMoves(player, action);
    for (const auto& move : moves) {
      if (move.pos == from_pos && move.to_pos == to_pos && move.die == die) {
        return true;
      }
    }
  }
  return false;
}

void BearingOffAllowedWhenAllHome() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][0] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {1, 1},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  SPIEL_CHECK_TRUE(HasBearOffMove(lnstate, kXPlayerId));
}

void BearingOffBlockedWhenNotAllHome() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][0] = 1;
  board_real[kXPlayerId][1] = 13;
  board_real[kXPlayerId][7] = 1;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {1, 1},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  SPIEL_CHECK_FALSE(HasBearOffMove(lnstate, kXPlayerId));
}

void NormalMoveAllowedWhenExactBearOffPossible() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][0] = 1;
  board_real[kXPlayerId][1] = kNumCheckersPerPlayer - 1;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {1, 1},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  SPIEL_CHECK_TRUE(HasBearOffMove(lnstate, kXPlayerId));
  SPIEL_CHECK_TRUE(HasMove(lnstate, kXPlayerId, /*from_pos=*/1,
                           /*to_pos=*/0, /*die=*/1));
}

void OverkillRuleRespectsFarthestChecker() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][5] = 0;
  board_real[kXPlayerId][4] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {6, 1},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  bool can_bear_from_4_now = false;
  for (Action action : lnstate->LegalActions()) {
    auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 4 && lnstate->IsOff(kXPlayerId, move.to_pos) &&
          move.die == 6) {
        can_bear_from_4_now = true;
      }
    }
  }
  SPIEL_CHECK_TRUE(can_bear_from_4_now);
}

void ScoringTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::array<std::array<int, kNumPoints>, 2> board{};
  board[0].fill(0);
  board[1].fill(0);
  board[1][0] = kNumCheckersPerPlayer;
  lnstate->SetStateForTesting(board, kXPlayerId, {1, 1},
                              /*awaiting_roll=*/false,
                              /*initial_roll=*/false,
                              /*is_first_turn=*/false,
                              /*head_moved_count=*/0,
                              /*phase=*/0);
  auto returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2.0);

  board[1][0] = 10;
  lnstate->SetStateForTesting(board, kXPlayerId, {1, 1},
                              /*awaiting_roll=*/false,
                              /*initial_roll=*/false,
                              /*is_first_turn=*/false,
                              /*head_moved_count=*/0,
                              /*phase=*/0);
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1.0);

  std::shared_ptr<const Game> game_tie =
      LoadGame("long_narde(scoring_type=winlosstie_scoring)");
  std::unique_ptr<State> tie_state = game_tie->NewInitialState();
  auto* tie_lnstate = static_cast<LongNardeState*>(tie_state.get());
  board[0].fill(0);
  board[1].fill(0);
  tie_lnstate->SetStateForTesting(board, kXPlayerId, {1, 1},
                                  /*awaiting_roll=*/false,
                                  /*initial_roll=*/false,
                                  /*is_first_turn=*/false,
                                  /*head_moved_count=*/0,
                                  /*phase=*/0);
  returns = tie_lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 0.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 0.0);
}

}  // namespace

void TestEndgame() {
  BearingOffAllowedWhenAllHome();
  BearingOffBlockedWhenNotAllHome();
  NormalMoveAllowedWhenExactBearOffPossible();
  OverkillRuleRespectsFarthestChecker();
  ScoringTest();
  std::cout << "✓ Endgame tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
