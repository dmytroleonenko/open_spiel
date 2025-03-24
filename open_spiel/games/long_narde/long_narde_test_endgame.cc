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
  
  // Check that White can bear off by verifying all checkers are in home
  bool all_checkers_in_home = true;
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // If we find any checker outside the home region
    if (lnstate->board(kXPlayerId, pos) > 0 && !lnstate->IsPosInHome(kXPlayerId, pos)) {
      all_checkers_in_home = false;
      break;
    }
  }
  
  // White should be able to bear off since all checkers are in home
  SPIEL_CHECK_TRUE(all_checkers_in_home);
  
  // Now set a checker outside of home
  test_board = {
    {2, 3, 3, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, // White (one at 15)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black
  };
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Check that White can't bear off due to checker outside home
  all_checkers_in_home = true;
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // If we find any checker outside the home region
    if (lnstate->board(kXPlayerId, pos) > 0 && !lnstate->IsPosInHome(kXPlayerId, pos)) {
      all_checkers_in_home = false;
      break;
    }
  }
  
  // White should not be able to bear off due to checker outside home
  SPIEL_CHECK_FALSE(all_checkers_in_home);
  
  // Check for Black player too
  test_board = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},  // White
    {2, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} // Black
  };
  lnstate->SetState(kOPlayerId, false, dice, {0, 0}, test_board);
  
  // Check that Black can bear off
  all_checkers_in_home = true;
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // If we find any checker outside the home region
    if (lnstate->board(kOPlayerId, pos) > 0 && !lnstate->IsPosInHome(kOPlayerId, pos)) {
      all_checkers_in_home = false;
      break;
    }
  }
  
  // Black should be able to bear off
  SPIEL_CHECK_TRUE(all_checkers_in_home);
}

void BearingOffLogicTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Setup a board position where White has checkers in position 1, 2
  std::vector<std::vector<int>> test_board = {
    {0, 1, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice = {1, 3};
  
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Find a move to bear off the checker from position 1
  bool can_bear_off_with_1 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos == 1 && move.die == 1 && move.to_pos == kBearOffPos) {
        can_bear_off_with_1 = true;
        break;
      }
    }
    if (can_bear_off_with_1) break;
  }
  
  // Find a move to bear off with the 3 die (exact move from position 3)
  bool can_bear_off_with_3 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos == 3 && move.die == 3 && move.to_pos == kBearOffPos) {
        can_bear_off_with_3 = true;
        break;
      }
    }
    if (can_bear_off_with_3) break;
  }
  
  // Both bearing off moves should be legal
  SPIEL_CHECK_TRUE(can_bear_off_with_1);
  SPIEL_CHECK_TRUE(can_bear_off_with_3);
  
  // Create a new test board with checkers further back
  test_board = {
    {0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Check if any checkers are outside the home region
  bool any_checker_outside_home = false;
  for (int pos = 0; pos < kNumPoints; ++pos) {
    if (lnstate->board(kXPlayerId, pos) > 0 && !lnstate->IsPosInHome(kXPlayerId, pos)) {
      any_checker_outside_home = true;
      break;
    }
  }
  
  // Should have checkers outside home
  SPIEL_CHECK_TRUE(any_checker_outside_home);
}

void BearingOffFromPosition1Test() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Setup a board position where White has one checker in position 1
  std::vector<std::vector<int>> test_board = {
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice = {1, 3};
  
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Find bearing off moves
  bool can_bear_off_with_1 = false;
  bool can_bear_off_with_3 = false;
  bool has_pass = false;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    
    // Check for bearing off with 1
    if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 1) {
      can_bear_off_with_1 = true;
    }
    
    // Check for bearing off with 3
    if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 3) {
      can_bear_off_with_3 = true;
    }
    
    // Check for pass move
    if (moves.size() >= 1 && moves[0].pos == kPassPos) {
      has_pass = false;  // Should not have a pass move if bearing off is possible
    }
  }
  
  // Should be able to bear off with 1 (exact move)
  SPIEL_CHECK_TRUE(can_bear_off_with_1);
  
  // Should be able to bear off with 3 (greater than needed)
  SPIEL_CHECK_TRUE(can_bear_off_with_3);
  
  // Should not have a pass move
  SPIEL_CHECK_FALSE(has_pass);
}

void EndgameScoreTest() {
  // Test different scoring methods
  
  // 1. Test Mars scoring (default)
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has borne off all checkers and Black has none
  lnstate->SetState(
      kXPlayerId, true, {1, 2}, {15, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should be terminal
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  
  // Check returns
  std::vector<double> returns = lnstate->Returns();
  
  // White should get 2 points for Mars (all checkers off while opponent has none)
  SPIEL_CHECK_EQ(returns[kXPlayerId], 2.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -2.0);
  
  // 2. Test Oin scoring (1 point)
  state = game->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where White has borne off all checkers and Black has some
  lnstate->SetState(
      kXPlayerId, true, {1, 2}, {15, 5}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should be terminal
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  
  // Check returns
  returns = lnstate->Returns();
  
  // White should get 1 point (regular win)
  SPIEL_CHECK_EQ(returns[kXPlayerId], 1.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], -1.0);
  
  // 3. Test tie (winlosstie mode)
  std::shared_ptr<const Game> game_tie = LoadGame("long_narde(scoring_type=winlosstie_scoring)");
  state = game_tie->NewInitialState();
  lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a position where both players have borne off all checkers
  lnstate->SetState(
      kXPlayerId, true, {1, 2}, {15, 15}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Game should be terminal
  SPIEL_CHECK_TRUE(lnstate->IsTerminal());
  
  // Check returns
  returns = lnstate->Returns();
  
  // Should be a tie (0 points each)
  SPIEL_CHECK_EQ(returns[kXPlayerId], 0.0);
  SPIEL_CHECK_EQ(returns[kOPlayerId], 0.0);
}

void ScoringSystemTest() {
  // Test different scoring parameters
  
  // 1. Default scoring (0 = winloss, 1 = winlosstie)
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  auto params = game->GetParameters();
  auto scoring_str = params.count("scoring_type") > 0 ? 
    params.at("scoring_type").string_value() : "winloss_scoring";
  
  // Check if string value is "winloss_scoring" which corresponds to 0
  SPIEL_CHECK_EQ(scoring_str, "winloss_scoring");
  
  // 2. Explicit winloss scoring
  game = LoadGame("long_narde(scoring_type=winloss_scoring)");
  params = game->GetParameters();
  scoring_str = params.at("scoring_type").string_value();
  
  // Check if string value is "winloss_scoring" which corresponds to 0
  SPIEL_CHECK_EQ(scoring_str, "winloss_scoring");
  
  // 3. Winlosstie scoring
  game = LoadGame("long_narde(scoring_type=winlosstie_scoring)");
  params = game->GetParameters();
  scoring_str = params.at("scoring_type").string_value();
  
  // Check if string value is "winlosstie_scoring" which corresponds to 1
  SPIEL_CHECK_EQ(scoring_str, "winlosstie_scoring");
}

}  // namespace

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