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

#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

void BearingOffBasicTest() {
  std::cout << "[BearingOffBasicTest] Starting test..." << std::endl;
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test board where White has all checkers in home
  std::vector<std::vector<int>> test_board = {
    {2, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black
  };
  std::vector<int> dice = {1, 5};
  
  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal moves
  auto legal_actions = lnstate->LegalActions();
  
  // Check that bearing off is possible
  bool can_bear_off = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.to_pos == kBearOffPos) {
        can_bear_off = true;
        break;
      }
    }
    if (can_bear_off) break;
  }
  
  SPIEL_CHECK_TRUE(can_bear_off);
  std::cout << "[BearingOffBasicTest] Can bear off: " << (can_bear_off ? "YES" : "NO") << std::endl;
}

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
      if (move.pos == 0 && move.die == 5) {
        can_bear_off_pos_0 = true;
      }
      if (move.pos == 4 && move.die == 5) {
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
      if (move.pos == 12 && move.die == 5) {
        can_bear_off_pos_12 = true;
      }
      if (move.pos == 16 && move.die == 5) {
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
      if (lnstate->IsOff(kXPlayerId, lnstate->GetToPos(kXPlayerId, move.pos, move.die))) {
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
      if (lnstate->IsOff(kXPlayerId, lnstate->GetToPos(kXPlayerId, move.pos, move.die))) {
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

void BearingOffFromPosition1Test() {
  std::cout << "[BearingOffFromPosition1Test] Starting test..." << std::endl;
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Create our test board from scratch
  std::vector<std::vector<int>> test_board = {
    {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White: 7 at pos 1
    {0, 0, 0, 0, 3, 1, 5, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0}  // Black distribution
  };
  
  // Set up the test state with white to move, dice 1 and 3
  std::vector<int> dice = {1, 3};
  std::vector<int> scores = {8, 0}; // White has 8 checkers borne off
  lnstate->SetState(kXPlayerId, false, dice, scores, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Verify that bearing off actions are available
  bool can_bear_off_with_1 = false;
  bool can_bear_off_with_3 = false;
  bool has_pass = false;
  
  std::cout << "Board state for bearing off test:\n" << lnstate->ToString() << std::endl;
  std::cout << "Legal actions count: " << legal_actions.size() << std::endl;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    
    for (const CheckerMove& move : moves) {
      // For White, to_pos < 0 means bearing off
      if (move.pos == 1 && move.to_pos < 0) {
        if (move.die == 1) {
          can_bear_off_with_1 = true;
        } else if (move.die == 3) {
          can_bear_off_with_3 = true;
        }
      }
    }
    
    // Check for pass action
    std::vector<CheckerMove> pass_moves = {kPassMove, kPassMove};
    Action pass_action = lnstate->CheckerMovesToSpielMove(pass_moves);
    if (action == pass_action) {
      has_pass = true;
    }
  }
  
  // The bug is that we should be able to bear off with both dice (1 and 3)
  // but the current implementation only allows bearing off with the 1
  std::cout << "Can bear off with 1: " << (can_bear_off_with_1 ? "YES" : "NO") << std::endl;
  std::cout << "Can bear off with 3: " << (can_bear_off_with_3 ? "YES" : "NO") << std::endl;
  std::cout << "Has pass action: " << (has_pass ? "YES" : "NO") << std::endl;
  
  // We expect this test to fail initially - the bug is that we can only bear off with an exact roll
  // The correct behavior is that we should be able to bear off with any roll
  // So our test should check that we can bear off with both dice (1 and 3)
  SPIEL_CHECK_TRUE(can_bear_off_with_1);  // Should be able to bear off with 1
  SPIEL_CHECK_TRUE(can_bear_off_with_3);  // Should be able to bear off with 3
  SPIEL_CHECK_FALSE(has_pass);           // Should not need to pass
  std::cout << "✓ BearingOffFromPosition1Test " << 
    (can_bear_off_with_3 ? "passed" : "failed (as expected)") << std::endl;
}

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

void EndgameScoreTest() {
  std::cout << "[EndgameScoreTest] Starting test..." << std::endl;
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test board where White has all checkers in position 1
  std::vector<std::vector<int>> test_board = {
    {0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black
  };
  std::vector<int> dice = {1, 1};
  
  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Bear off all White checkers
  while (!lnstate->IsTerminal()) {
    // Handle chance nodes
    if (lnstate->IsChanceNode()) {
      lnstate->ApplyAction(15);  // Always roll double 1s for simplicity
      continue;
    }
    
    // Get legal moves
    auto legal_actions = lnstate->LegalActions();
    
    // Apply an action that bears off if possible
    bool applied = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(lnstate->CurrentPlayer(), action);
      bool bears_off = false;
      for (const CheckerMove& move : moves) {
        if (move.to_pos == kBearOffPos) {
          bears_off = true;
          break;
        }
      }
      
      if (bears_off) {
        lnstate->ApplyAction(action);
        applied = true;
        break;
      }
    }
    
    // If no bearing off move found, just apply the first legal action
    if (!applied && !legal_actions.empty()) {
      lnstate->ApplyAction(legal_actions[0]);
    }
  }
  
  // Verify the game is terminal and White has won (score should be positive)
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  
  std::vector<double> returns = lnstate->Returns();
  std::cout << "[EndgameScoreTest] Returns for White: " << returns[kXPlayerId] << std::endl;
  std::cout << "[EndgameScoreTest] Returns for Black: " << returns[kOPlayerId] << std::endl;
  
  SPIEL_CHECK_GT(returns[kXPlayerId], 0);
  SPIEL_CHECK_LT(returns[kOPlayerId], 0);
}

}  // namespace

// Exposed test function
void TestEndgame() {
  std::cout << "\n=== Testing Endgame Rules ===" << std::endl;
  
  std::cout << "\n=== Running BearingOffBasicTest ===\n";
  BearingOffBasicTest();
  std::cout << "✓ BearingOffBasicTest passed\n";
  
  std::cout << "\n=== Running BearingOffLogicTest ===\n";
  BearingOffLogicTest();
  std::cout << "✓ BearingOffLogicTest passed\n";
  
  std::cout << "\n=== Running BearingOffFromPosition1Test ===\n";
  BearingOffFromPosition1Test();
  std::cout << "✓ BearingOffFromPosition1Test passed\n";
  
  std::cout << "\n=== Running EndgameScoreTest ===\n";
  EndgameScoreTest();
  std::cout << "✓ EndgameScoreTest passed\n";
  
  std::cout << "\n=== Running ScoringSystemTest ===\n";
  ScoringSystemTest();
  std::cout << "✓ ScoringSystemTest passed\n";
  
  std::cout << "✓ All endgame tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel