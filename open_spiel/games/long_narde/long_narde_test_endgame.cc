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
  
  // Set up a test board where White has all checkers in home (points 0-5)
  std::vector<std::vector<int>> test_board = {
    {3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White - all 15 checkers in home
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black
  };
  std::vector<int> dice = {1, 5};
  
  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Check that White can bear off by verifying all checkers are in home
  bool all_checkers_in_home = lnstate->AllInHome(kXPlayerId);
  
  // White should be able to bear off since all checkers are in home
  SPIEL_CHECK_TRUE(all_checkers_in_home);
  
  // Now set a checker outside of home
  test_board = {
    {3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, // White (one at 15)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black
  };
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Check that White can't bear off due to checker outside home
  all_checkers_in_home = lnstate->AllInHome(kXPlayerId);
  
  // White should not be able to bear off due to checker outside home
  SPIEL_CHECK_FALSE(all_checkers_in_home);
  
  // Check for Black player too - set up board where Black has all checkers in home (indices 12-17)
  test_board = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15},  // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0} // Black - all 15 in home (points 13-18)
  };
  lnstate->SetState(kOPlayerId, false, dice, {0, 0}, test_board);
  
  // Check that Black can bear off
  all_checkers_in_home = lnstate->AllInHome(kOPlayerId);
  
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
  
  // Find a move that uses die 1 from position 1 (should move to pos 0, not bear off)
  bool can_bear_off_with_1 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      // Use IsOff to check if the move results in bearing off
      if (move.pos == 1 && move.die == 1 && lnstate->IsOff(kXPlayerId, move.to_pos)) {
        can_bear_off_with_1 = true;
        break;
      }
    }
    if (can_bear_off_with_1) break;
  }
  
  // Find a move to bear off with the 3 die from position 2
  bool can_bear_off_with_3 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      // Use IsOff to check if the move results in bearing off
      if (move.pos == 2 && move.die == 3 && lnstate->IsOff(kXPlayerId, move.to_pos)) {
        can_bear_off_with_3 = true;
        break;
      }
    }
    if (can_bear_off_with_3) break;
  }
  
  // Move with die 1 from pos 1 should NOT bear off (goes to pos 0)
  SPIEL_CHECK_FALSE(can_bear_off_with_1);
  // Move with die 3 from pos 2 SHOULD bear off
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
  
  // Setup a board position where White has all checkers in home.
  // Crucially, only checkers are on points 0 and 1 to test bearing off from 1 with a higher roll.
  std::vector<std::vector<int>> test_board = {
    {14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White (14 on point 0, 1 on point 1)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}  // Black (all at head)
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
    if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 1 && lnstate->IsOff(kXPlayerId, moves[0].to_pos)) {
      can_bear_off_with_1 = true;
    }
    
    // Check for bearing off with 3
    if (moves.size() >= 1 && moves[0].pos == 1 && moves[0].die == 3 && lnstate->IsOff(kXPlayerId, moves[0].to_pos)) {
      can_bear_off_with_3 = true;
    }
    
    // Check for pass move
    if (moves.size() >= 1 && moves[0].pos == kPassPos) {
      has_pass = true;
    }
  }
  
  // Should be able to bear off with 1 (exact move)
  SPIEL_CHECK_FALSE(can_bear_off_with_1);  // Die 1 from pos 1 goes to pos 0, not off
  
  // Should be able to bear off with 3 (greater than needed)
  // This is now TRUE because there are no checkers on points 2, 3, 4, or 5.
  SPIEL_CHECK_TRUE(can_bear_off_with_3);
  
  // Should not have a pass move
  SPIEL_CHECK_FALSE(has_pass);
}

void BearingOffBlackTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // Setup board: Black has checkers in home (12-17), including near the end (22, 23)
  // Points 12-17 are Black's home. Points 18-23 are the final quadrant.
  std::vector<std::vector<int>> test_board = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}, // White (all at head)
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1}  // Black (15 checkers in home 12-23)
  };
  std::vector<int> dice = {2, 3}; // Dice roll 2, 3

  // Set Black (kOPlayerId) to move
  lnstate->SetState(kOPlayerId, false, dice, {0, 0}, test_board);

  // Verify all Black checkers are in their home board (12-23 for bearing off)
  // Note: The definition of "home" for bearing off might differ slightly from IsPosInHome.
  // Let's assume AllInHome correctly checks the bearing-off region 12-23 for Black.
  // If AllInHome uses 12-17, this test setup needs adjustment or AllInHome needs review.
  // For now, we proceed assuming AllInHome checks the correct region for bearing off.
  SPIEL_CHECK_TRUE(lnstate->AllInHome(kOPlayerId));

  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have moves

  bool can_bear_off_22_with_2 = false; // Exact roll from pos 22
  bool can_bear_off_23_with_1 = false; // Exact roll from pos 23 (if die 1 was rolled) - Check with 2 instead
  bool can_bear_off_23_with_2 = false; // Higher roll from pos 23 (needs die 1 exactly)
  bool can_bear_off_23_with_3 = false; // Higher roll from pos 23 (needs die 1 exactly)

  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const CheckerMove& move : moves) {
      // Check bearing off from pos 22 with die 2 (Exact: 24 - 22 = 2)
      if (move.pos == 22 && move.die == 2 && lnstate->IsOff(kOPlayerId, move.to_pos)) {
        can_bear_off_22_with_2 = true;
      }
      // Check bearing off from pos 23 with die 2 (Higher: needs 1, has 2)
      if (move.pos == 23 && move.die == 2 && lnstate->IsOff(kOPlayerId, move.to_pos)) {
        can_bear_off_23_with_2 = true;
      }
      // Check bearing off from pos 23 with die 3 (Higher: needs 1, has 3)
      if (move.pos == 23 && move.die == 3 && lnstate->IsOff(kOPlayerId, move.to_pos)) {
        can_bear_off_23_with_3 = true;
      }
    }
  }

  // Assertions based on rules:
  // Pos 22 needs exactly 2 to bear off.
  SPIEL_CHECK_TRUE(can_bear_off_22_with_2);
  // Pos 23 needs exactly 1 to bear off. Since it's NOT the furthest checker
  // (checkers exist at 12, 13, 14, 22), higher rolls (2 and 3) should NOT work.
  SPIEL_CHECK_FALSE(can_bear_off_23_with_2); // Changed from TRUE
  SPIEL_CHECK_FALSE(can_bear_off_23_with_3); // Changed from TRUE
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
  
  std::cout << "\n=== Running BearingOffBlackTest ===\n";
  BearingOffBlackTest();
  std::cout << "✓ BearingOffBlackTest passed\n";
  
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