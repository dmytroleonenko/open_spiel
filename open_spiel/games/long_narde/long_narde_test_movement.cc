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

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

void InitialBoardSetupTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  SPIEL_CHECK_EQ(lnstate->GetCount(kXPlayerId, 23), kNumCheckersPerPlayer);
  SPIEL_CHECK_EQ(lnstate->GetCount(kOPlayerId, 11), kNumCheckersPerPlayer);
}

void WhiteMovesFirstTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  if (lnstate->IsChanceNode()) {
    auto outcomes = lnstate->ChanceOutcomes();
    lnstate->ApplyAction(outcomes[0].first);
  }
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
}

void MovementDirectionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  SPIEL_CHECK_EQ(lnstate->GetToPos(kXPlayerId, 23, 4), 19);
  SPIEL_CHECK_EQ(lnstate->GetToPos(kOPlayerId, 11, 4), 7);
}

void HeadRuleSpecialFirstTurn() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][23] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {4, 4},
                                   /*is_first_turn=*/true,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<Action> actions = lnstate->LegalActions();
  bool found_two_from_head = false;
  for (Action action : actions) {
    auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
    int from_head = 0;
    for (const auto& move : moves) {
      if (move.pos == 23) {
        from_head++;
      }
    }
    if (from_head == 2) {
      found_two_from_head = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(found_two_from_head);
}

void HeadRuleNonFirstTurn() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][23] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {4, 4},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<Action> actions = lnstate->LegalActions();
  for (Action action : actions) {
    auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
    int from_head = 0;
    for (const auto& move : moves) {
      if (move.pos == 23) {
        from_head++;
      }
    }
    SPIEL_CHECK_LT(from_head, 2);
  }
}

void NoLandingOnOpponentWhite() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][23] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][22] = 1;
  board_real[kOPlayerId][21] = 1;
  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {1, 2},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<Action> actions = lnstate->LegalActions();
  for (Action action : actions) {
    auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == kPassPos) {
        continue;
      }
      SPIEL_CHECK_NE(move.to_pos, 22);
      SPIEL_CHECK_NE(move.to_pos, 21);
    }
  }
}

void NoLandingOnOpponentBlack() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;
  board_real[kXPlayerId][10] = 1;
  board_real[kXPlayerId][9] = 1;
  testing::SetupStateFromRealBoard(lnstate, board_real, kOPlayerId, {1, 2},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<Action> actions = lnstate->LegalActions();
  for (Action action : actions) {
    auto moves = lnstate->LongNardeSpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == kPassPos) {
        continue;
      }
      SPIEL_CHECK_NE(move.to_pos, 10);
      SPIEL_CHECK_NE(move.to_pos, 9);
    }
  }
}

}  // namespace

void TestMovementRules() {
  InitialBoardSetupTest();
  WhiteMovesFirstTest();
  MovementDirectionTest();
  HeadRuleSpecialFirstTurn();
  HeadRuleNonFirstTurn();
  NoLandingOnOpponentWhite();
  NoLandingOnOpponentBlack();
  std::cout << "✓ Movement rules tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
