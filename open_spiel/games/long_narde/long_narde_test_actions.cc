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

void ActionEncodingTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Get the kNumDistinctActions constant - using NumDistinctActions() method
  int kNumDistinctActions = lnstate->NumDistinctActions();
  
  // Apply a specific dice roll instead of using global variables
  // Roll dice 3, 5 for testing
  lnstate->ApplyAction(1);  // Just pick an action from chance outcomes to set dice
  
  // Make sure we're in player mode, not chance mode
  while (lnstate->IsChanceNode()) {
    lnstate->ApplyAction(1);  // Apply another chance outcome if needed
  }
  
  // White's turn
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  
  // Test 1: Regular move encoding (high roll first)
  // Create a move: Position 15 with die 5, position 19 with die 3
  std::vector<CheckerMove> test_moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 5, 5},
    {kWhiteHeadPos, kWhiteHeadPos - 3, 3}
  };
  
  // Encode the moves to a SpielMove (Action)
  Action action = lnstate->CheckerMovesToSpielMove(test_moves);
  
  // Verify the action is within range
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  
  // Decode back to checker moves
  std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
  
  // Verify decoded moves match original
  SPIEL_CHECK_EQ(decoded_moves.size(), 2);
  
  bool first_move_found = false;
  bool second_move_found = false;
  
  for (const CheckerMove& move : decoded_moves) {
    if (move.pos == test_moves[0].pos && move.die == test_moves[0].die) {
      first_move_found = true;
    }
    if (move.pos == test_moves[1].pos && move.die == test_moves[1].die) {
      second_move_found = true;
    }
  }
  
  SPIEL_CHECK_TRUE(first_move_found);
  SPIEL_CHECK_TRUE(second_move_found);
  
  // Test 2: Pass move encoding
  std::vector<CheckerMove> pass_moves = {
    {kPassPos, kPassPos, 5},
    {kPassPos, kPassPos, 3}
  };
  
  action = lnstate->CheckerMovesToSpielMove(pass_moves);
  
  // Verify the action is within range
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  
  // Decode back to checker moves
  decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
  
  // Verify decoded moves are passes
  SPIEL_CHECK_EQ(decoded_moves.size(), 2);
  
  bool first_pass_found = false;
  bool second_pass_found = false;
  
  for (const CheckerMove& move : decoded_moves) {
    if (move.pos == kPassPos && move.die == 5) {
      first_pass_found = true;
    }
    if (move.pos == kPassPos && move.die == 3) {
      second_pass_found = true;
    }
  }
  
  SPIEL_CHECK_TRUE(first_pass_found);
  SPIEL_CHECK_TRUE(second_pass_found);
}

void SingleLegalMoveTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test board where White has only one legal move
  std::vector<std::vector<int>> test_board = {
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14}, // White
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice = {1, 2};
  
  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal moves
  auto legal_actions = lnstate->LegalActions();
  
  // White should have only one legal move: the checker at point 1
  // Note: There might be multiple action encodings depending on the order of dice use
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  
  // Verify all actions move the checker from point 1
  bool all_from_point_1 = true;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.pos != 1 && move.pos != kPassPos) {
        all_from_point_1 = false;
        break;
      }
    }
    if (!all_from_point_1) break;
  }
  
  SPIEL_CHECK_TRUE(all_from_point_1);
}

void ConsecutiveMovesTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Apply a chance outcome for dice roll
  lnstate->ApplyAction(15);  // Double 1s
  
  // Initial white turn - move from head
  std::vector<CheckerMove> moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1},
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1}
  };
  
  Action action = lnstate->CheckerMovesToSpielMove(moves);
  lnstate->ApplyAction(action);
  
  // Should still be White's turn with a new dice roll (consecutive moves for doubles)
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
  
  // Apply another dice roll
  lnstate->ApplyAction(15);  // Double 1s again
  
  // Now White moves again
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  
  // Apply another move
  lnstate->ApplyAction(action);
  
  // Should still be White's turn with a new dice roll
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
  
  // Apply a non-double dice roll
  lnstate->ApplyAction(0);  // Dice roll 1,2
  
  // Now White moves again
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  
  // Apply another move
  std::vector<CheckerMove> moves_non_double = {
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1},
    {kWhiteHeadPos, kWhiteHeadPos - 2, 2}
  };
  Action action_non_double = lnstate->CheckerMovesToSpielMove(moves_non_double);
  lnstate->ApplyAction(action_non_double);
  
  // Now it should be Black's turn with a dice roll
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);
  
  // Verify next player is Black
  lnstate->ApplyAction(0);  // Roll dice
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kOPlayerId);
}

void UndoRedoTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Apply a chance outcome for dice roll
  lnstate->ApplyAction(0);  // Roll 1,2
  
  // Get the board state before any moves
  std::vector<std::vector<int>> board_before;
  for (int p = 0; p < 2; ++p) {
    std::vector<int> player_board;
    for (int i = 0; i < kNumPoints; ++i) {
      player_board.push_back(lnstate->board(p, i));
    }
    board_before.push_back(player_board);
  }
  
  // Make a move
  std::vector<CheckerMove> moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 1, 1},
    {kWhiteHeadPos, kWhiteHeadPos - 2, 2}
  };
  Action action = lnstate->CheckerMovesToSpielMove(moves);
  lnstate->ApplyAction(action);
  
  // Check that the board state changed
  bool board_changed = false;
  for (int p = 0; p < 2; ++p) {
    for (int i = 0; i < kNumPoints; ++i) {
      if (lnstate->board(p, i) != board_before[p][i]) {
        board_changed = true;
        break;
      }
    }
    if (board_changed) break;
  }
  SPIEL_CHECK_TRUE(board_changed);
  
  // Now undo the move
  lnstate->UndoAction(kXPlayerId, action);
  
  // Check that the board state is back to the original
  bool board_restored = true;
  for (int p = 0; p < 2; ++p) {
    for (int i = 0; i < kNumPoints; ++i) {
      if (lnstate->board(p, i) != board_before[p][i]) {
        board_restored = false;
        break;
      }
    }
    if (!board_restored) break;
  }
  SPIEL_CHECK_TRUE(board_restored);
}

void UseHigherDieRuleTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test board where White has checkers at positions 1 and 2
  std::vector<std::vector<int>> test_board = {
    {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  
  // Dice roll 2,5 (higher die is 5)
  std::vector<int> dice = {2, 5};
  
  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Create a modified board with a Black checker at position 23
  std::vector<std::vector<int>> modified_board = test_board;
  modified_board[kOPlayerId][23] = 1;  // Place a Black checker at position 23
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, modified_board);
  
  // Create a modified board where White can only move the checker at position 2
  // with the higher die (5), and position 1 is blocked with the lower die (2)
  modified_board = test_board;
  modified_board[kOPlayerId][1 - 2 + kNumPoints] = 1;  // Block position 1 with die 2 (adjusted for board bounds)
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, modified_board);
  
  // Get legal actions
  auto legal_actions = lnstate->LegalActions();
  
  // White should have at least one legal move (using the higher die 5)
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  
  // Check if any legal action uses the higher die (5)
  bool found_higher_die_move = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.die == 5) {
        found_higher_die_move = true;
        break;
      }
    }
    if (found_higher_die_move) break;
  }
  
  // Verify that the higher die (5) can be used
  SPIEL_CHECK_TRUE(found_higher_die_move);
  
  // Now set up a position where White can only move with the lower die (2)
  test_board = {
    {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Block both checkers' moves with the higher die (5)
  modified_board = test_board;
  modified_board[kOPlayerId][1 - 5 + kNumPoints] = 1;  // Block position 1 with die 5 (adjusted for board bounds)
  modified_board[kOPlayerId][2 - 5 + kNumPoints] = 1;  // Block position 2 with die 5 (adjusted for board bounds)
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, modified_board);
  
  // Get legal actions
  legal_actions = lnstate->LegalActions();
  
  // White should have at least one legal move (using the lower die 2)
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  
  // Check if any legal action uses the lower die (2)
  bool found_lower_die_move = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      if (move.die == 2) {
        found_lower_die_move = true;
        break;
      }
    }
    if (found_lower_die_move) break;
  }
  
  // Verify that the lower die (2) can be used when higher die (5) is blocked
  SPIEL_CHECK_TRUE(found_lower_die_move);
}

// Test function to validate pass move behavior
void PassMoveBehaviorTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Create pass move encoding for later comparison
  std::vector<CheckerMove> pass_move_encoding = {kPassMove, kPassMove};
  Action pass_action = lnstate->CheckerMovesToSpielMove(pass_move_encoding);
  
  // TEST CASE 1: No valid moves available
  // Set up a board state where there are no legal moves
  // White has checkers at positions that can't move with dice 1,3
  // (opponent's checkers block all landing spots)
  std::vector<std::vector<int>> no_moves_board = {
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice_1_3 = {1, 3};
  
  // Set up the state
  lnstate->SetState(kXPlayerId, false, dice_1_3, {0, 0}, no_moves_board);
  
  // Get legal actions
  auto legal_actions = lnstate->LegalActions();
  
  // Check that only the pass action is available
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  SPIEL_CHECK_EQ(legal_actions[0], pass_action);
  
  // TEST CASE 2: At least one valid move is available
  // White has checkers that can move with dice 1,3
  std::vector<std::vector<int>> valid_moves_board = {
    {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  
  // Set up the state
  lnstate->SetState(kXPlayerId, false, dice_1_3, {0, 0}, valid_moves_board);
  
  // Get legal actions
  legal_actions = lnstate->LegalActions();
  
  // Check that pass action is NOT among the legal actions
  bool pass_found = false;
  for (Action action : legal_actions) {
    if (action == pass_action) {
      pass_found = true;
      break;
    }
  }
  SPIEL_CHECK_FALSE(pass_found);
  SPIEL_CHECK_GT(legal_actions.size(), 0);  // Should have at least one legal move
  
  // TEST CASE 3: Doubles with partial moves
  // Test with doubles where player can use some but not all of the moves
  std::vector<std::vector<int>> partial_moves_board = {
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
    {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> doubles_dice = {2, 2};  // Double 2s
  
  // Set up the state - Black's turn with doubles
  lnstate->SetState(kOPlayerId, false, doubles_dice, {0, 0}, partial_moves_board);
  
  // Get legal actions
  legal_actions = lnstate->LegalActions();
  
  // Verify there are some legal moves but not the pass move
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  pass_found = false;
  for (Action action : legal_actions) {
    if (action == pass_action) {
      pass_found = true;
      break;
    }
  }
  SPIEL_CHECK_FALSE(pass_found);
  
  // Apply a move that uses up one of the dice
  // Find a legal move that uses one die
  Action partial_action = -1;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
    if (moves.size() == 2 && moves[0].pos != kPassPos && moves[1].pos == kPassPos) {
      partial_action = action;
      break;
    }
  }
  
  // If we found a partial action, apply it
  if (partial_action != -1) {
    // Now create a board state where the next die can't be used
    std::vector<std::vector<int>> after_partial_board = {
      {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
    };
    
    // Block possible moves for the next die
    std::vector<int> remaining_dice = {2};  // One die left
    
    // Set up state - still Black's turn but with one die and a blocked position
    lnstate->SetState(kOPlayerId, true, remaining_dice, {0, 0}, after_partial_board);
    
    // Get legal actions
    legal_actions = lnstate->LegalActions();
    
    // Now only pass should be available
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    SPIEL_CHECK_EQ(legal_actions[0], pass_action);
  }
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

// Register the test function in the common interface
namespace open_spiel {
namespace long_narde {

// Implement the global test function that was declared in the header
void TestPassMoveBehavior() {
  std::cout << "\n=== Testing Pass Move Behavior ===\n";
  PassMoveBehaviorTest();
  std::cout << "✓ Pass Move Behavior Test passed\n";
}

void TestActionEncoding() {
  std::cout << "\n=== Testing Action Encoding ===\n";
  ActionEncodingTest();
  SingleLegalMoveTest();
  ConsecutiveMovesTest();
  UndoRedoTest();
  
  std::cout << "✓ All action encoding tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
 