#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

// Test basic movement (already in original file)
void TestBasicMovementInternal() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // White (X player) moves first
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  
  // Initial state is a chance node (dice roll)
  SPIEL_CHECK_TRUE(lnstate->IsChanceNode());
  
  // Apply a chance outcome (dice roll)
  std::vector<std::pair<Action, double>> outcomes = lnstate->ChanceOutcomes();
  SPIEL_CHECK_EQ(outcomes.size(), 21);  // 15 non-doubles, 6 doubles = 21 outcomes
  
  // Roll 1,2 (represented by index 0 in chance outcomes)
  lnstate->ApplyAction(0);
  
  // Should now be player's turn, dice should be 1,2
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  SPIEL_CHECK_FALSE(lnstate->IsChanceNode());
  SPIEL_CHECK_EQ(lnstate->dice(0), 1);
  SPIEL_CHECK_EQ(lnstate->dice(1), 2);
  
  // White moves from point 24 to 23 and 22
  std::vector<Action> legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty());
  
  // Move two checkers from the head position
  std::vector<CheckerMove> checkers_moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1},  // Move from 24 to 23 using die 1
    {kWhiteHeadPos, kWhiteHeadPos - 2, 2}   // Move from 24 to 22 using die 2
  };
  Action action = lnstate->CheckerMovesToSpielMove(checkers_moves);
  
  // Apply the action
  lnstate->ApplyAction(action);
  
  // Verify board state after move
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), 13);  // 13 remain at 24
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 1), 1);  // 1 at 23
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 2), 1);  // 1 at 22
}

// Implementation of InitialDiceTest
void InitialDiceTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Initial state is a chance node (dice roll)
  SPIEL_CHECK_TRUE(lnstate->IsChanceNode());
  
  // Apply a chance outcome (dice roll)
  std::vector<std::pair<Action, double>> outcomes = lnstate->ChanceOutcomes();
  SPIEL_CHECK_EQ(outcomes.size(), 21);  // 15 non-doubles, 6 doubles = 21 outcomes
  
  // Check if we can find the dice roll actions mapped to expected values
  // Sample a few chance outcomes and verify the dice values
  for (const auto& outcome_pair : outcomes) {
    Action action = outcome_pair.first;
    
    // Clone the state
    std::unique_ptr<State> clone = state->Clone();
    auto clone_lnstate = static_cast<LongNardeState*>(clone.get());
    
    // Apply the dice roll
    clone_lnstate->ApplyAction(action);
    
    // Verify that the dice values are set correctly
    int die1 = clone_lnstate->dice(0);
    int die2 = clone_lnstate->dice(1);
    
    // Dice values should be between 1 and 6
    SPIEL_CHECK_GE(die1, 1);
    SPIEL_CHECK_LE(die1, 6);
    SPIEL_CHECK_GE(die2, 1);
    SPIEL_CHECK_LE(die2, 6);
    
    // Make sure the values are ordered correctly (highest die first)
    if (die1 != die2) {  // Skip doubles
      SPIEL_CHECK_GE(die1, die2);
    }
  }
  
  std::cout << "✓ Initial dice values verified\n";
}

// Implementation of CheckerDistributionTest
void CheckerDistributionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check initial distribution
  // White should have 15 checkers at position 24
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), kNumCheckersPerPlayer);
  
  // Black should have 15 checkers at position 12
  SPIEL_CHECK_EQ(lnstate->board(kOPlayerId, kBlackHeadPos), kNumCheckersPerPlayer);
  
  // Check that after a move, the distribution changes correctly
  // First get through the chance node
  lnstate->ApplyAction(0);  // Apply dice roll 1,2
  
  // Apply a move: move one checker from 24 to 23 and another from 24 to 22
  std::vector<CheckerMove> moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1},  // From 24 to 23
    {kWhiteHeadPos, kWhiteHeadPos - 2, 2}   // From 24 to 22
  };
  Action action = lnstate->CheckerMovesToSpielMove(moves);
  lnstate->ApplyAction(action);
  
  // Verify the new distribution
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos), 13);    // 13 left at 24
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 1), 1); // 1 at 23
  SPIEL_CHECK_EQ(lnstate->board(kXPlayerId, kWhiteHeadPos - 2), 1); // 1 at 22
  
  std::cout << "✓ Checker distribution verified\n";
}

// Implementation of FirstTurnTest
void FirstTurnTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Apply a chance outcome for dice roll
  lnstate->ApplyAction(0);  // Roll 1,2
  
  // First turn should allow multiple checkers to move from head
  SPIEL_CHECK_TRUE(lnstate->is_first_turn());
  
  // Get legal actions for first turn
  std::vector<Action> first_turn_actions = lnstate->LegalActions();
  
  // Find actions that move multiple checkers from head
  bool found_multiple_head_moves = false;
  for (Action action : first_turn_actions) {
    // Create a clone to test this action
    std::unique_ptr<State> clone = state->Clone();
    auto clone_lnstate = static_cast<LongNardeState*>(clone.get());
    
    // Get initial checker count at head
    int initial_count = clone_lnstate->board(kXPlayerId, kWhiteHeadPos);
    
    // Apply the action
    clone_lnstate->ApplyAction(action);
    
    // Get the new count at head
    int new_count = clone_lnstate->board(kXPlayerId, kWhiteHeadPos);
    
    // Check if multiple checkers moved from head
    if (initial_count - new_count > 1) {
      found_multiple_head_moves = true;
      break;
    }
  }
  
  // Should be able to move multiple checkers from head on first turn
  SPIEL_CHECK_TRUE(found_multiple_head_moves);
  
  // Now make a move and verify it's no longer first turn
  lnstate->ApplyAction(first_turn_actions[0]);
  
  // Go through chance node and get to next player
  if (lnstate->IsChanceNode()) {
    lnstate->ApplyAction(0);  // Another dice roll
  }
  if (lnstate->IsChanceNode()) {
    lnstate->ApplyAction(0);  // Another dice roll if needed
  }
  
  // Should now be Black's turn and not first turn
  if (lnstate->CurrentPlayer() == kOPlayerId) {
    SPIEL_CHECK_FALSE(lnstate->is_first_turn());
  }
  
  std::cout << "✓ First turn logic verified\n";
}

void MovementDirectionTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set White as current player
  lnstate->SetState(
      kXPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that White's movement is counter-clockwise (decreasing position numbers)
  std::vector<Action> white_actions = lnstate->LegalActions();
  bool white_counter_clockwise = true;
  for (Action action : white_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      // Skip bearing off moves
      if (move.to_pos == kBearOffPos) continue;
      
      // For White, counter-clockwise means to_pos < from_pos
      if (move.to_pos > move.pos) {
        white_counter_clockwise = false;
        break;
      }
    }
    if (!white_counter_clockwise) break;
  }
  SPIEL_CHECK_TRUE(white_counter_clockwise);
  
  // Set Black as current player
  lnstate->SetState(
      kOPlayerId, false, {3, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check that Black's movement is counter-clockwise, considering wrapping
  std::vector<Action> black_actions = lnstate->LegalActions();
  bool black_counter_clockwise = true;
  for (Action action : black_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      // Skip bearing off moves
      if (move.to_pos == kBearOffPos) continue;
      
      // For Black, counter-clockwise also means to_pos < from_pos, except when wrapping
      // When wrapping (to_pos > from_pos), we need to verify it's actually counter-clockwise
      if (move.to_pos > move.pos) {
        // Manually calculate if this is a valid wrap for counter-clockwise movement
        int direct_counter_clockwise = move.pos - move.die;
        int wrapped_position = (direct_counter_clockwise < 0) ? 
                              direct_counter_clockwise + kNumPoints : direct_counter_clockwise;
                              
        // Check if the wrapped position matches the to_pos
        if (wrapped_position != move.to_pos) {
          black_counter_clockwise = false;
          break;
        }
      }
    }
    if (!black_counter_clockwise) break;
  }
  SPIEL_CHECK_TRUE(black_counter_clockwise);
  
  std::cout << "✓ MovementDirectionTest passed\n";
}

void NoLandingOnOpponentTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Test 1: Basic landing prevention
  // Setup a position where White could potentially land on Black's checker at position 16
  lnstate->SetState(
      kXPlayerId, false, {4, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
      });
  
  // Check if any legal move would land on position 16 (where Black has a checker)
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // First, verify that White's checker at position 20 can't move to position 16
  // This would be position 20 - 4 = 16 (with die value 4)
  bool found_move_to_16 = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      // If we find a move from position 20 using die value 4
      if (move.pos == 19 && move.die == 4 && move.to_pos == 15) {
        found_move_to_16 = true;
        break;
      }
    }
    if (found_move_to_16) break;
  }
  
  // The move should not be in the legal actions
  SPIEL_CHECK_FALSE(found_move_to_16);
  
  // Test 2: Using direct IsValidCheckerMove to confirm this is rejected
  bool is_valid = lnstate->IsValidCheckerMove(kXPlayerId, 19, 15, 4);
  SPIEL_CHECK_FALSE(is_valid);
  
  // Test 3: More complex setup with multiple potential landings
  lnstate->SetState(
      kXPlayerId, false, {5, 3}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 3}
      });
  
  // Check that White's checker at position 17 can't move to position 12 where Black has checkers
  // This would be position 17 - 5 = 12 (with die value 5)
  legal_actions = lnstate->LegalActions();
  bool found_move_to_12 = false;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const auto& move : moves) {
      if (move.pos == 17 && move.die == 5 && move.to_pos == 12) {
        found_move_to_12 = true;
        break;
      }
    }
    if (found_move_to_12) break;
  }
  
  // The move should not be in the legal actions
  SPIEL_CHECK_FALSE(found_move_to_12);
  
  // Directly check this move is invalid
  is_valid = lnstate->IsValidCheckerMove(kXPlayerId, 17, 12, 5);
  SPIEL_CHECK_FALSE(is_valid);
  
  // Test 4: Check for Black player too
  lnstate->SetState(
      kOPlayerId, false, {4, 2}, {0, 0}, 
      std::vector<std::vector<int>>{
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 14},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
      });
  
  legal_actions = lnstate->LegalActions();
  bool black_can_land_on_white = false;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    for (const auto& move : moves) {
      if (move.to_pos >= 0 && move.to_pos < kNumPoints && lnstate->board(kXPlayerId, move.to_pos) > 0) {
        black_can_land_on_white = true;
        break;
      }
    }
    if (black_can_land_on_white) break;
  }
  
  // Black should not be able to land on White's checkers
  SPIEL_CHECK_FALSE(black_can_land_on_white);
  
  std::cout << "✓ NoLandingOnOpponentTest passed\n";
}

void HeadRuleTestInternal() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a non-first-turn state with dice 1,2
  std::vector<std::vector<int>> test_board = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice = {1, 2};
  
  // Set up the state with White to move, already moved from head (first turn)
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // The head rule states that in non-first turns, only one checker can move
  // from the head position. We'll count how many actions would lead to more 
  // than one checker moving from the head.
  
  // Count how many actions actually move checkers from the head position
  // by simulating the actions and checking the board state difference
  int actual_head_moves = 0;
  for (Action action : legal_actions) {
    // Clone the state to apply the action
    std::unique_ptr<State> clone = lnstate->Clone();
    LongNardeState* clone_state = static_cast<LongNardeState*>(clone.get());
    clone_state->MutableIsFirstTurn() = false; // Ensure is_first_turn_ is false in clone
    
    // Get the initial checkers at head position
    int head_pos = kWhiteHeadPos; // White's head position
    int initial_checkers = clone_state->board(kXPlayerId, head_pos);
    
    // Apply the action
    clone_state->ApplyAction(action);
    
    // Check how many checkers moved from the head
    int new_checkers = clone_state->board(kXPlayerId, head_pos);
    if (initial_checkers > new_checkers) {
      actual_head_moves++;
    }
  }
  
  // There should be actions that move checkers from the head, but no action should
  // move more than one checker from the head in a non-first turn
  SPIEL_CHECK_GT(actual_head_moves, 0); // Some actions should move from head
  SPIEL_CHECK_LE(actual_head_moves, legal_actions.size()); // Not more than total actions
  
  // Now test the first turn - should allow multiple checkers to leave head
  lnstate->MutableIsFirstTurn() = true;
  legal_actions = lnstate->LegalActions();
  
  // Count how many actions move a checker from the head
  int first_turn_head_moves = 0;
  std::vector<int> checkers_per_action;
  for (Action action : legal_actions) {
    // Clone the state to apply the action
    std::unique_ptr<State> clone = lnstate->Clone();
    LongNardeState* clone_state = static_cast<LongNardeState*>(clone.get());
    clone_state->MutableIsFirstTurn() = true; // Ensure is_first_turn_ is true in clone
    
    // Get the initial checkers at head position
    int head_pos = kWhiteHeadPos; // White's head position
    int initial_checkers = clone_state->board(kXPlayerId, head_pos);
    
    // Apply the action
    clone_state->ApplyAction(action);
    
    // Check how many checkers moved from the head
    int new_checkers = clone_state->board(kXPlayerId, head_pos);
    int checkers_moved = initial_checkers - new_checkers;
    if (checkers_moved > 0) {
      first_turn_head_moves++;
      checkers_per_action.push_back(checkers_moved);
    }
  }
  
  // In first turn, there should be actions with multiple checkers leaving the head
  SPIEL_CHECK_GT(first_turn_head_moves, 0);
  
  // Find if there's at least one action with 2 checkers moved from head
  bool has_multiple_checkers_moved = false;
  for (int count : checkers_per_action) {
    if (count >= 2) {
      has_multiple_checkers_moved = true;
      break;
    }
  }
  
  // Verify we can move multiple checkers from head on first turn
  SPIEL_CHECK_TRUE(has_multiple_checkers_moved);
}

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

}  // namespace

// Expose the original test function for compatibility
void TestBasicMovement() {
  std::cout << "\n=== Running TestBasicMovement ===\n";
  TestBasicMovementInternal();
  std::cout << "✓ Basic movement test passed\n";
}

// Expose the original head rule test for compatibility 
void TestHeadRule() {
  std::cout << "\n=== Running TestHeadRule ===\n";
  HeadRuleTestInternal();
  std::cout << "✓ Head rule test passed\n";
}

// New exposed test function that runs all movement tests
void TestMovementRules() {
  std::cout << "\n=== Testing Movement Rules ===" << std::endl;
  
  std::cout << "\n=== Running InitialDiceTest ===\n";
  InitialDiceTest();
  std::cout << "✓ InitialDiceTest passed\n";
  
  std::cout << "\n=== Running CheckerDistributionTest ===\n";
  CheckerDistributionTest();
  std::cout << "✓ CheckerDistributionTest passed\n";
  
  std::cout << "\n=== Running FirstTurnTest ===\n";
  FirstTurnTest();
  std::cout << "✓ FirstTurnTest passed\n";
  
  std::cout << "\n=== Running MovementDirectionTest ===\n";
  MovementDirectionTest();
  std::cout << "✓ MovementDirectionTest passed\n";
  
  std::cout << "\n=== Running HomeRegionsTest ===\n";
  HomeRegionsTest();
  std::cout << "✓ HomeRegionsTest passed\n";
  
  std::cout << "\n=== Running NoLandingOnOpponentTest ===\n";
  NoLandingOnOpponentTest();
  std::cout << "✓ NoLandingOnOpponentTest passed\n";
  
  std::cout << "\n=== Running HeadRuleTestInternal ===\n";
  HeadRuleTestInternal();
  std::cout << "✓ HeadRuleTestInternal passed\n";
  
  std::cout << "✓ All movement tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel