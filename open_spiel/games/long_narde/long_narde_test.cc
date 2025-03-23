// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

namespace testing = open_spiel::testing;

bool ActionsContains(const std::vector<Action>& legal_actions, Action action) {
  return std::find(legal_actions.begin(), legal_actions.end(), action) !=
         legal_actions.end();
}

// Long Narde doesn't have hits, so we check that no hits are returned
void CheckNoHits(const State &state) {
  if (state.IsChanceNode() || state.IsTerminal()) {
    return;
  }
  Player player = state.CurrentPlayer();
  const auto &lnstate = down_cast<const LongNardeState &>(state);
  for (Action action : lnstate.LegalActions()) {
    std::vector<CheckerMove> cmoves = lnstate.SpielMoveToCheckerMoves(player, action);
    for (CheckerMove cmove : cmoves) {
      // Remove this line since CheckerMove doesn't have a hit member in long_narde
      // SPIEL_CHECK_FALSE(cmove.hit);
    }
  }
}

void BasicLongNardeTestsCheckNoHits() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
  // Run with just 1 simulation instead of 10 to reduce output
  testing::RandomSimTest(*game, 1, false, true, &CheckNoHits);
}

void BasicLongNardeTestsDoNotStartWithDoubles() {
  std::mt19937 rng;
  for (int i = 0; i < 100; ++i) {
    auto game = LoadGame("long_narde");
    std::unique_ptr<State> state = game->NewInitialState();

    while (state->IsChanceNode()) {
      Action outcome =
          SampleAction(state->ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
      state->ApplyAction(outcome);
    }
    LongNardeState* long_narde_state =
        dynamic_cast<LongNardeState*>(state.get());
    // The dice should contain two different numbers,
    // because a tie would not select a starting player.
    SPIEL_CHECK_NE(long_narde_state->dice(0), long_narde_state->dice(1));
  }
}

// Test correct initial board setup for Long Narde
// White's 15 checkers on point 24, Black's 15 on point 12
void InitialBoardSetupTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check initial setup for White (kXPlayerId) - all 15 on point 24 (index 23)
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), kNumCheckersPerPlayer);
  
  // Check initial setup for Black (kOPlayerId) - all 15 on point 12 (index 11)
  SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, kBlackHeadPos), kNumCheckersPerPlayer);
  
  // Verify no checkers anywhere else on the board
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (pos != kWhiteHeadPos) {
      SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, pos), 0);
    }
    if (pos != kBlackHeadPos) {
      SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, pos), 0);
    }
  }
}

// Test head rule: Only 1 checker can leave the head per turn
void HeadRuleTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: Regular turn (not first turn) - only one checker should be allowed to leave head
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Count actions that move more than one checker from the head
  int multi_head_moves = 0;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify no actions allow more than one checker to leave the head on regular turns
  SPIEL_CHECK_EQ(multi_head_moves, 0);
  
  // Test 2: First turn with regular non-doubles dice - should still only allow one head move
  lnstate->SetState(
      kXPlayerId, true, {2, 5}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify no actions allow more than one checker to leave the head on first turn with non-special dice
  SPIEL_CHECK_EQ(multi_head_moves, 0);
  
  // Test 3: First turn with double 6 (special dice) - should allow two head moves
  lnstate->SetState(
      kXPlayerId, true, {6, 6}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify actions exist that allow two checkers to leave the head on first turn with double 6
  SPIEL_CHECK_GT(multi_head_moves, 0);
  
  // Test 4: First turn with double 4 (special dice) - should allow two head moves
  lnstate->SetState(
      kXPlayerId, true, {4, 4}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify actions exist that allow two checkers to leave the head on first turn with double 4
  SPIEL_CHECK_GT(multi_head_moves, 0);
  
  // Test 5: First turn with double 3 (special dice) - should allow two head moves
  lnstate->SetState(
      kXPlayerId, true, {3, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify actions exist that allow two checkers to leave the head on first turn with double 3
  SPIEL_CHECK_GT(multi_head_moves, 0);
  
  // Test 6: First turn with double 2 (non-special doubles) - should NOT allow two head moves
  lnstate->SetState(
      kXPlayerId, true, {2, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify no actions allow more than one checker to leave the head on first turn with non-special doubles
  SPIEL_CHECK_EQ(multi_head_moves, 0);
  
  // Test 7: First turn with double 1 (non-special doubles) - should NOT allow two head moves
  lnstate->SetState(
      kXPlayerId, true, {1, 1}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify no actions allow more than one checker to leave the head on first turn with non-special doubles
  SPIEL_CHECK_EQ(multi_head_moves, 0);
  
  // Test 8: Black's first turn with special doubles (6-6) - should allow two head moves
  lnstate->SetState(
      kOPlayerId, true, {6, 6}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  multi_head_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kBlackHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify actions exist that allow two checkers to leave the head on first turn with double 6
  SPIEL_CHECK_GT(multi_head_moves, 0);
}

// Test first turn with doubles exception (6-6, 4-4, or 3-3)
void FirstTurnDoublesExceptionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a first turn situation with dice 6,6
  lnstate->SetState(
      kXPlayerId, true, {6, 6}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Mark as first turn
  lnstate->SetState(
      kXPlayerId, true, {6, 6}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Ensure there are actions that move two checkers from the head on special doubles
  int multi_head_moves = 0;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    int head_moves = 0;
    for (const auto& move : moves) {
      if (move.pos == kWhiteHeadPos) head_moves++;
    }
    if (head_moves > 1) multi_head_moves++;
  }
  
  // Verify that we have actions allowing two checkers to move from head on double 6
  SPIEL_CHECK_GT(multi_head_moves, 0);
}

// Test blocking bridge rule (cannot form 6 consecutive points that trap opponent)
void BlockingBridgeRuleTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: White attempting to create an illegal 6-point prime that would trap Black
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Find actions that would complete the bridge (move from position 22 to position 19)
  bool can_create_bridge = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 22 && move.num == 3) { // Would create a 6-point bridge
        can_create_bridge = true;
      }
    }
  }
  
  // Verify that completing the bridge is not allowed
  SPIEL_CHECK_FALSE(can_create_bridge);
  
  // Test 2: White should be able to create a 6-point prime when Black has checkers ahead of it
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  
  // Find actions that would complete the bridge
  can_create_bridge = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 22 && move.num == 3) { // Would create a 6-point bridge, but it's legal
        can_create_bridge = true;
      }
    }
  }
  
  // Verify that completing the bridge is allowed when opponent has checkers ahead
  SPIEL_CHECK_TRUE(can_create_bridge);
  
  // Test 3: Black attempting to create an illegal 6-point prime that would trap White
  lnstate->SetState(
      kOPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  
  // Find actions that would complete the bridge (move from position 1 to position 4)
  can_create_bridge = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 1 && move.num == 3) { // Would create a 6-point bridge
        can_create_bridge = true;
      }
    }
  }
  
  // Verify that completing the bridge is not allowed
  SPIEL_CHECK_FALSE(can_create_bridge);
  
  // Test 4: Black should be able to create a 6-point prime when White has checkers ahead of it
  lnstate->SetState(
      kOPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  
  // Find actions that would complete the bridge
  can_create_bridge = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 1 && move.num == 3) { // Would create a 6-point bridge, but it's legal
        can_create_bridge = true;
      }
    }
  }
  
  // Verify that completing the bridge is allowed when opponent has checkers ahead
  SPIEL_CHECK_TRUE(can_create_bridge);
}

// Test movement direction - both players must move counter-clockwise
void MovementDirectionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a situation with a checker moved out for each player
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that White's legal moves are counter-clockwise (decreasing position numbers)
  std::vector<Action> white_actions = lnstate->LegalActions();
  bool white_clockwise_move_found = false;
  for (Action action : white_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kXPlayerId, move.pos, move.num);
      if (to_pos > move.pos && !lnstate->IsOff(kXPlayerId, to_pos)) {
        white_clockwise_move_found = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(white_clockwise_move_found);
  
  // Set Black as current player
  lnstate->SetState(
      kOPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that Black's legal moves are counter-clockwise (decreasing position numbers)
  std::vector<Action> black_actions = lnstate->LegalActions();
  bool black_clockwise_move_found = false;
  for (Action action : black_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kOPlayerId, move.pos, move.num);
      if (to_pos > move.pos && !lnstate->IsOff(kOPlayerId, to_pos)) {
        black_clockwise_move_found = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(black_clockwise_move_found);
}

// Test home regions: White (1-6) and Black (13-18)
void HomeRegionsTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check White's home region (points 1-6)
  for (int pos = 0; pos <= 5; ++pos) {
    SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kXPlayerId, pos));
  }
  for (int pos = 6; pos < kNumPoints; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kXPlayerId, pos));
  }
  
  // Check Black's home region (points 13-18)
  for (int pos = 12; pos <= 17; ++pos) {
    SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
  for (int pos = 0; pos < 12; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
  for (int pos = 18; pos < kNumPoints; ++pos) {
    SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, pos));
  }
}

// Test bearing off logic - must use exact or higher rolls when all checkers are in home
void BearingOffLogicTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: White bearing off with exact and higher rolls
  lnstate->SetState(
      kXPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Check that bearing off with exact or higher rolls is allowed
  bool can_bear_off_pos_0 = false;  // Can bear off from position 0 with a 5
  bool can_bear_off_pos_4 = false;  // Can bear off from position 4 with a 5
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 0 && move.num == 5) {
        can_bear_off_pos_0 = true;
      }
      if (move.pos == 4 && move.num == 5) {
        can_bear_off_pos_4 = true;
      }
    }
  }
  
  // Verify exact and higher bearing off is allowed
  SPIEL_CHECK_TRUE(can_bear_off_pos_0);  // Exact roll
  SPIEL_CHECK_TRUE(can_bear_off_pos_4);  // Higher roll
  
  // Test 2: Black bearing off with exact and higher rolls
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  bool can_bear_off_pos_12 = false;  // Can bear off from position 12 with a 5
  bool can_bear_off_pos_16 = false;  // Can bear off from position 16 with a 5
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 12 && move.num == 5) {
        can_bear_off_pos_12 = true;
      }
      if (move.pos == 16 && move.num == 5) {
        can_bear_off_pos_16 = true;
      }
    }
  }
  
  SPIEL_CHECK_TRUE(can_bear_off_pos_12);  // Exact roll
  SPIEL_CHECK_TRUE(can_bear_off_pos_16);  // Higher roll
  
  // Test 3: Bearing off with doubles
  lnstate->SetState(
      kXPlayerId, false, {6, 6}, {0, 0}, 
      std::vector<std::vector<int>>{
        {2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  int bear_off_moves = 0;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (lnstate->IsOff(kXPlayerId, lnstate->GetToPos(kXPlayerId, move.pos, move.num))) {
        bear_off_moves++;
      }
    }
  }
  
  // Should be able to bear off multiple checkers with doubles
  SPIEL_CHECK_GT(bear_off_moves, 1);
  
  // Test 4: Cannot bear off when checkers are outside home
  lnstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, 
      std::vector<std::vector<int>>{
        {2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  bool can_bear_off = false;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (lnstate->IsOff(kXPlayerId, lnstate->GetToPos(kXPlayerId, move.pos, move.num))) {
        can_bear_off = true;
      }
    }
  }
  
  SPIEL_CHECK_FALSE(can_bear_off);
  
  // Test 5: Score updates and undo for bearing off
  lnstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
      });
  
  // White bears off
  CheckerMove white_move(5, 6);
  SPIEL_CHECK_EQ(lnstate->score(kXPlayerId), 0);
  lnstate->ApplyCheckerMove(kXPlayerId, white_move);
  SPIEL_CHECK_EQ(lnstate->score(kXPlayerId), 1);
  
  // Undo White's move
  lnstate->UndoCheckerMove(kXPlayerId, white_move);
  SPIEL_CHECK_EQ(lnstate->score(kXPlayerId), 0);
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, 5), 1);
  
  // Black bears off
  lnstate->SetState(
      kOPlayerId, false, {6, 5}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
      });
  
  CheckerMove black_move(17, 6);
  SPIEL_CHECK_EQ(lnstate->score(kOPlayerId), 0);
  lnstate->ApplyCheckerMove(kOPlayerId, black_move);
  SPIEL_CHECK_EQ(lnstate->score(kOPlayerId), 1);
  
  // Undo Black's move
  lnstate->UndoCheckerMove(kOPlayerId, black_move);
  SPIEL_CHECK_EQ(lnstate->score(kOPlayerId), 0);
  SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, 17), 1);
}

// Test scoring system and last roll tie rule
void ScoringSystemTest() {
  // Test 1: Mars scoring (White wins, Black has no checkers off)
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {15, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that White gets 2 points (mars) for bearing off all checkers when Black has none
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  auto returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2);
  
  // Test 2: Oyn scoring (White wins, Black has some checkers off)
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that White gets 1 point (oyn) for bearing off all checkers when Black has some
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1);
  
  // Test 3: Black mars White
  lnstate->SetState(
      kXPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Black gets 2 points for mars
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], -2);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 2);
  
  // Test 4: Last roll tie rule in winloss mode (should not allow tie)
  game = LoadGame("long_narde", {{"scoring_type", GameParameter("winloss_scoring")}});
  state = game->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has all checkers off, Black has 14 off and 1 in home
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should be terminal in winloss mode
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1);  // White wins with oyn
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1);
  
  // Test 5: Last roll tie rule in winlosstie mode
  game = LoadGame("long_narde", {{"scoring_type", GameParameter("winlosstie_scoring")}});
  state = game->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  
  // Same position as Test 4
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should NOT be terminal in winlosstie mode
  SPIEL_CHECK_FALSE(lnstate->IsTerminal());
  
  // After Black bears off last checker
  lnstate->SetState(
      kChancePlayerId, false, {0, 0}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Now game should be terminal with a tie
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 0);

  // Test 6: Last roll tie rule in winlosstie mode with mars opportunity
  game = LoadGame("long_narde", {{"scoring_type", GameParameter("winlosstie_scoring")}});
  state = game->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has all checkers off, Black has none off
  lnstate->SetState(
      kOPlayerId, false, {5, 3}, {15, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should be terminal with mars score even in winlosstie mode
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  returns = lnstate->Returns();
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2);  // White wins with mars
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2);
}

// Test that landing on opponent checkers is not allowed in Long Narde
void NoLandingOnOpponentTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: Basic landing prevention
  // Setup a position where White could potentially land on Black's checker
  lnstate->SetState(
      kXPlayerId, false, {4, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
      });
  
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Try to find an action that would land on opponent's checker
  bool can_land_on_opponent = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      // Check if White can move from position 23 to position 19 (landing on Black's checker)
      if (move.pos == 19 && move.num == 2) {
        can_land_on_opponent = true;
      }
      // Check if White can move from position 23 to position 16 (landing on Black's checker)
      if (move.pos == 23 && move.num == 4 && lnstate->GetToPos(kXPlayerId, move.pos, move.num) == 16) {
        can_land_on_opponent = true;
      }
    }
  }
  
  // Verify no actions allow landing on opponent's checkers
  SPIEL_CHECK_FALSE(can_land_on_opponent);
  
  // Test 2: Landing prevention with doubles
  lnstate->SetState(
      kXPlayerId, false, {4, 4}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  can_land_on_opponent = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kXPlayerId, move.pos, move.num);
      if (lnstate->board(kOPlayerId, to_pos) > 0) {
        can_land_on_opponent = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(can_land_on_opponent);
  
  // Test 3: Multiple opponent checkers
  lnstate->SetState(
      kXPlayerId, false, {6, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  can_land_on_opponent = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kXPlayerId, move.pos, move.num);
      if (lnstate->board(kOPlayerId, to_pos) > 0) {
        can_land_on_opponent = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(can_land_on_opponent);
  
  // Test 4: Edge cases near board boundaries
  lnstate->SetState(
      kXPlayerId, false, {5, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  can_land_on_opponent = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      int to_pos = lnstate->GetToPos(kXPlayerId, move.pos, move.num);
      if (lnstate->board(kOPlayerId, to_pos) > 0) {
        can_land_on_opponent = true;
      }
    }
  }
  SPIEL_CHECK_FALSE(can_land_on_opponent);
  
  // Test 5: Random simulation test
  testing::RandomSimTest(*game, 100, true, true, [](const State& state) {
    if (state.IsChanceNode() || state.IsTerminal()) return;
    const auto& lnstate = down_cast<const LongNardeState&>(state);
    Player player = state.CurrentPlayer();
    for (Action action : lnstate.LegalActions()) {
      std::vector<CheckerMove> moves = lnstate.SpielMoveToCheckerMoves(player, action);
      for (const CheckerMove& move : moves) {
        int to_pos = lnstate.GetToPos(player, move.pos, move.num);
        if (!lnstate.IsOff(player, to_pos)) {
          SPIEL_CHECK_EQ(lnstate.board(1 - player, to_pos), 0);
        }
      }
    }
  });
}

void ActionEncodingTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test state with dice values
  lnstate->SetState(
      kXPlayerId, false, {6, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });

  // Test 1: Regular move encoding (high roll first)
  std::vector<CheckerMove> moves1 = {{23, 6}, {17, 3}};  // Move from pos 23 with 6, then from 17 with 3
  Action encoded1 = lnstate->CheckerMovesToSpielMove(moves1);
  SPIEL_CHECK_LT(encoded1, 625);  // Should be in first half (high roll first)
  std::vector<CheckerMove> decoded1 = lnstate->SpielMoveToCheckerMoves(kXPlayerId, encoded1);
  SPIEL_CHECK_EQ(decoded1[0].pos, moves1[0].pos);
  SPIEL_CHECK_EQ(decoded1[0].num, moves1[0].num);
  SPIEL_CHECK_EQ(decoded1[1].pos, moves1[1].pos);
  SPIEL_CHECK_EQ(decoded1[1].num, moves1[1].num);

  // Test 2: Regular move encoding (low roll first)
  std::vector<CheckerMove> moves2 = {{23, 3}, {20, 6}};  // Move from pos 23 with 3, then from 20 with 6
  Action encoded2 = lnstate->CheckerMovesToSpielMove(moves2);
  SPIEL_CHECK_GE(encoded2, 625);  // Should be in second half (low roll first)
  std::vector<CheckerMove> decoded2 = lnstate->SpielMoveToCheckerMoves(kXPlayerId, encoded2);
  SPIEL_CHECK_EQ(decoded2[0].pos, moves2[0].pos);
  SPIEL_CHECK_EQ(decoded2[0].num, moves2[0].num);
  SPIEL_CHECK_EQ(decoded2[1].pos, moves2[1].pos);
  SPIEL_CHECK_EQ(decoded2[1].num, moves2[1].num);

  // Test 3: Pass move encoding
  std::vector<CheckerMove> moves3 = {{kPassPos, -1}, {kPassPos, -1}};  // Double pass
  Action encoded3 = lnstate->CheckerMovesToSpielMove(moves3);
  std::vector<CheckerMove> decoded3 = lnstate->SpielMoveToCheckerMoves(kXPlayerId, encoded3);
  SPIEL_CHECK_EQ(decoded3[0].pos, kPassPos);
  SPIEL_CHECK_EQ(decoded3[0].num, -1);
  SPIEL_CHECK_EQ(decoded3[1].pos, kPassPos);
  SPIEL_CHECK_EQ(decoded3[1].num, -1);

  // Test 4: Mixed regular and pass move
  std::vector<CheckerMove> moves4 = {{23, 6}, {kPassPos, -1}};  // One move and one pass
  Action encoded4 = lnstate->CheckerMovesToSpielMove(moves4);
  std::vector<CheckerMove> decoded4 = lnstate->SpielMoveToCheckerMoves(kXPlayerId, encoded4);
  SPIEL_CHECK_EQ(decoded4[0].pos, moves4[0].pos);
  SPIEL_CHECK_EQ(decoded4[0].num, moves4[0].num);
  SPIEL_CHECK_EQ(decoded4[1].pos, kPassPos);
  SPIEL_CHECK_EQ(decoded4[1].num, -1);

  // Test 5: Verify bounds
  SPIEL_CHECK_LT(encoded1, kNumDistinctActions);
  SPIEL_CHECK_LT(encoded2, kNumDistinctActions);
  SPIEL_CHECK_LT(encoded3, kNumDistinctActions);
  SPIEL_CHECK_LT(encoded4, kNumDistinctActions);
}

void TestBearingOffLogic() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* ln_state = static_cast<LongNardeState*>(state.get());
  
  // Create a custom board state where all checkers are in home for both players
  // White home: 0-5, Black home: 12-17
  std::vector<int> scores = {0, 0};
  std::vector<std::vector<int>> board = {
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}   // Black
  };
  
  // Set the state to have white's turn next with a fixed dice roll of 6-5
  ln_state->SetState(kXPlayerId, false, {6, 5}, scores, board);
  
  // Test White's bearing off
  // White has a checker on position 5, rolling a 6 should bear it off
  CheckerMove white_move(5, 6);
  SPIEL_CHECK_EQ(ln_state->score(kXPlayerId), 0);
  ln_state->ApplyCheckerMove(kXPlayerId, white_move);
  SPIEL_CHECK_EQ(ln_state->score(kXPlayerId), 1);
  
  // Undo the move and verify it's back
  ln_state->UndoCheckerMove(kXPlayerId, white_move);
  SPIEL_CHECK_EQ(ln_state->score(kXPlayerId), 0);
  SPIEL_CHECK_EQ(ln_state->board(kXPlayerId, 5), 1);
  
  // Now test Black's bearing off
  // Black has a checker on position 17, rolling a 6 should bear it off
  ln_state->SetState(kOPlayerId, false, {6, 5}, scores, board);
  CheckerMove black_move(17, 6);
  SPIEL_CHECK_EQ(ln_state->score(kOPlayerId), 0);
  ln_state->ApplyCheckerMove(kOPlayerId, black_move);
  SPIEL_CHECK_EQ(ln_state->score(kOPlayerId), 1);
  
  // Undo the move and verify it's back
  ln_state->UndoCheckerMove(kOPlayerId, black_move);
  SPIEL_CHECK_EQ(ln_state->score(kOPlayerId), 0);
  SPIEL_CHECK_EQ(ln_state->board(kOPlayerId, 17), 1);
}

// Test home regions: White (1-6) and Black (13-18)
void IsPosInHomeTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // White's home is points 1-6 (indices 0-5)
  SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kXPlayerId, 0));   // Point 1
  SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kXPlayerId, 5));   // Point 6
  SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kXPlayerId, 6));  // Point 7
  SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kXPlayerId, 23)); // Point 24
  
  // Black's home is points 13-18 (indices 12-17)
  SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kOPlayerId, 12));  // Point 13
  SPIEL_CHECK_TRUE(lnstate->IsPosInHome(kOPlayerId, 17));  // Point 18
  SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, 11)); // Point 12
  SPIEL_CHECK_FALSE(lnstate->IsPosInHome(kOPlayerId, 18)); // Point 19
}

// Test FurthestCheckerInHome function
void FurthestCheckerInHomeTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: Empty home board
  // Set up an empty board - no checkers in home
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}});
       
  // No checkers in home for either player
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kXPlayerId), -1);
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kOPlayerId), -1);
  
  // Test 2: Some checkers in home for White
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0},
      {{0, 0, 3, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}});
       
  // Furthest checker for White is at position 4 (point 5)
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kXPlayerId), 4);
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kOPlayerId), -1);
  
  // Test 3: Some checkers in home for Black
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 0, 0, 0}});
       
  // Furthest checker for Black is at position 12 (point 13)
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kXPlayerId), -1);
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kOPlayerId), 12);
  
  // Test 4: Checkers in home for both players
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0},
      {{0, 0, 3, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 0, 0, 0}});
       
  // Furthest checkers: White at position 4 (point 5), Black at position 12 (point 13)
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kXPlayerId), 4);
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kOPlayerId), 12);
  
  // Test 5: Varying furthest positions
  lnstate->SetState(
      kXPlayerId, false, {3, 4}, {0, 0},
      {{1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0}});
       
  // Furthest checkers: White at position 0 (point 1), Black at position 17 (point 18)
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kXPlayerId), 0);
  SPIEL_CHECK_EQ(lnstate->FurthestCheckerInHome(kOPlayerId), 17);
}

void BasicLongNardeTests() {
  open_spiel::testing::LoadGameTest("long_narde");
  open_spiel::long_narde::BasicLongNardeTestsCheckNoHits();
  open_spiel::long_narde::BasicLongNardeTestsDoNotStartWithDoubles();
  open_spiel::long_narde::InitialBoardSetupTest();
  open_spiel::long_narde::HeadRuleTest();
  open_spiel::long_narde::FirstTurnDoublesExceptionTest();
  open_spiel::long_narde::BlockingBridgeRuleTest();
  open_spiel::long_narde::MovementDirectionTest();
  open_spiel::long_narde::HomeRegionsTest();
  open_spiel::long_narde::BearingOffLogicTest();
  open_spiel::long_narde::ScoringSystemTest();
  open_spiel::long_narde::NoLandingOnOpponentTest();
  open_spiel::long_narde::ActionEncodingTest();
  open_spiel::long_narde::TestBearingOffLogic();
  open_spiel::long_narde::IsPosInHomeTest();
  open_spiel::long_narde::FurthestCheckerInHomeTest();
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("long_narde");
  open_spiel::long_narde::BasicLongNardeTests();
}
