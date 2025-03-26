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
  
  // Create a fresh initial board state to ensure turns_ is -1
  std::vector<std::vector<int>> initial_board(2, std::vector<int>(kNumPoints, 0));
  initial_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  initial_board[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
  lnstate->SetState(kXPlayerId, false, {}, {0, 0}, initial_board);
  
  // Removed the forced chance roll. We'll directly set the dice when calling SetState.

  // White's turn
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);
  
  // Modify the board state so that not all white checkers are at the head
  // The initial board has all white checkers at kWhiteHeadPos, which enforces the head rule.
  // To allow encoding two moves (one from a non-head point), remove two checkers from the head
  // and place one each at positions 14 and 19.
  std::vector<std::vector<int>> modified_board(2, std::vector<int>(kNumPoints, 0));
  
  // Copy the current board state
  for (int pos = 0; pos < kNumPoints; ++pos) {
    modified_board[kXPlayerId][pos] = lnstate->board(kXPlayerId, pos);
    modified_board[kOPlayerId][pos] = lnstate->board(kOPlayerId, pos);
  }
  
  // Remove two checkers from the head (kWhiteHeadPos) and add one to positions 14 and 19
  modified_board[kXPlayerId][kWhiteHeadPos] -= 2;
  modified_board[kXPlayerId][14] += 1;
  modified_board[kXPlayerId][19] += 1;
  
  // Update the state with the modified board while keeping the same dice and scores
  std::vector<int> scores = {lnstate->score(kXPlayerId), lnstate->score(kOPlayerId)};
  lnstate->SetState(kXPlayerId, false, std::vector<int>{5, 3}, scores, modified_board);
  
  // Test 1: Regular move encoding (high roll first)
  // Create a move: Position 14 with die 5, position 19 with die 3
  std::vector<CheckerMove> test_moves = {
    {14, lnstate->GetToPos(kXPlayerId, 14, 5), 5},
    {19, lnstate->GetToPos(kXPlayerId, 19, 3), 3}
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
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White
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

  // Apply a chance outcome for dice roll (double 1s)
  lnstate->ApplyAction(15);  // This sets dice to {1, 1} for the first turn

  // First turn: gather legal actions for White with double 1s (first turn).
  auto first_turn_legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(first_turn_legal_actions.empty());

  // Pick one valid action. In practice you may want to decode them
  // and pick an action that actually moves two from the head, or one from the head, etc.
  Action first_action = first_turn_legal_actions[0];
  lnstate->ApplyAction(first_action);

  // After White's first move on doubles, it should be a chance node for the extra turn.
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);

  // Apply another dice roll (again double 1s).
  lnstate->ApplyAction(15);

  // White's extra turn with double 1s (second turn):
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);

  // Gather new legal actions for the second turn. (Important to do this again!)
  auto second_turn_legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(second_turn_legal_actions.empty());

  // Usually we expect something like 19938 or 20832 if the head rule permits only 1 from the head now.
  // But let's just pick the first valid one in the list:
  Action second_action = second_turn_legal_actions[0];
  lnstate->ApplyAction(second_action);

  // With doubles used up, we should be back to chance for Black's normal turn.
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);

  // Apply a non-double dice roll (e.g. roll 1,2).
  lnstate->ApplyAction(0);  // means dice {1,2} or {2,1}, depending on internal logic

  // Now it's Black's turn.
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kOPlayerId);

  // Gather black's legal actions and apply the first one, for simplicity.
  auto black_legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(black_legal_actions.empty());
  lnstate->ApplyAction(black_legal_actions[0]);

  // Done: chance node for White's next turn.
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId);

  // Possibly roll again or do more checks. For this test, we are done.
}

void UndoRedoTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // --- Define a mid-game state ---
  std::vector<std::vector<int>> mid_game_board(2, std::vector<int>(kNumPoints, 0));
  // White checkers (kXPlayerId)
  mid_game_board[kXPlayerId][3] = 2;
  mid_game_board[kXPlayerId][5] = 3;
  mid_game_board[kXPlayerId][8] = 1;
  mid_game_board[kXPlayerId][10] = 2;
  mid_game_board[kXPlayerId][14] = 2;
  mid_game_board[kXPlayerId][17] = 3;
  mid_game_board[kXPlayerId][20] = 2; // Total 15 - 2 borne off = 13 on board

  // Black checkers (kOPlayerId)
  mid_game_board[kOPlayerId][1] = 3;
  mid_game_board[kOPlayerId][6] = 2;
  mid_game_board[kOPlayerId][9] = 2;
  mid_game_board[kOPlayerId][12] = 1; // Black's head
  mid_game_board[kOPlayerId][15] = 2;
  mid_game_board[kOPlayerId][18] = 2;
  mid_game_board[kOPlayerId][22] = 2; // Total 15 - 1 borne off = 14 on board

  std::vector<int> dice = {4, 2}; // White rolled 4, 2
  std::vector<int> scores = {2, 1}; // White: 2 borne off, Black: 1 borne off
  int current_player = kXPlayerId;
  bool double_turn = false; // Not a double roll

  // Set the state
  lnstate->SetState(current_player, double_turn, dice, scores, mid_game_board);
  // --- End mid-game state definition ---

  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId); // Verify White's turn

  // Get the board state before any moves
  std::vector<std::vector<int>> board_before;
  for (int p = 0; p < 2; ++p) {
    std::vector<int> player_board;
    for (int i = 0; i < kNumPoints; ++i) {
      player_board.push_back(lnstate->board(p, i));
    }
    board_before.push_back(player_board);
  }

  // Get legal actions for the current state
  std::vector<Action> legal_actions = lnstate->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty()); // Ensure there are legal actions in this mid-game state

  // Choose the first legal action
  Action action_to_apply = legal_actions[0];

  // Apply the chosen legal action
  lnstate->ApplyAction(action_to_apply);

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
  lnstate->UndoAction(kXPlayerId, action_to_apply); // Use the same action we applied

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

  // Removed generic pass_action calculation from here.
  // We will calculate the expected pass action within each test case based on the dice.

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

  // Calculate the expected pass action for dice {1, 3}
  std::vector<CheckerMove> expected_pass_encoding_1_3 = {{kPassPos, kPassPos, 1}, {kPassPos, kPassPos, 3}};
  Action expected_pass_action_1_3 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_1_3);

  // Get legal actions
  auto legal_actions = lnstate->LegalActions();

  // Check that only the pass action (specific to dice 1,3) is available
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_1_3); // Compare against specific pass action

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

  // Check that pass action (specific to dice 1,3) is NOT among the legal actions
  bool pass_found = false;
  for (Action action : legal_actions) {
    if (action == expected_pass_action_1_3) { // Compare against specific pass action
      pass_found = true;
      break;
    }
  }
  SPIEL_CHECK_FALSE(pass_found);
  SPIEL_CHECK_GT(legal_actions.size(), 0);  // Should have at least one legal move

  // TEST CASE 3: Doubles with no moves possible
  // Test with doubles where player cannot use any of the moves
  std::vector<std::vector<int>> no_moves_doubles_board = {
    // White blocks positions 4 and 9 (targets for Black's checkers with die 2)
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13}, // White
    // Black has checkers at 2 and 7 that could potentially move with die 2
    {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13}  // Black
  };
  std::vector<int> doubles_dice = {2, 2};  // Double 2s

  // Set up the state - Black's turn with doubles, no moves possible
  lnstate->SetState(kOPlayerId, false, doubles_dice, {0, 0}, no_moves_doubles_board);

  // Calculate the expected pass action for dice {2, 2}
  std::vector<CheckerMove> expected_pass_encoding_2_2 = {{kPassPos, kPassPos, 2}, {kPassPos, kPassPos, 2}};
  Action expected_pass_action_2_2 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_2_2);

  // Get legal actions
  legal_actions = lnstate->LegalActions();

  // Verify only the pass move is available
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_2_2); // Compare against specific pass action for {2, 2}

  // The previous logic simulating a partial move and then a pass was flawed
  // because LegalActions correctly requires playing the single available die if possible.
  // This revised test case directly checks the scenario where a pass is forced with doubles.
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
 