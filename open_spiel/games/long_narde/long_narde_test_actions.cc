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

  // Set up a test board where White has only one checker at pos 1 (point 2)
  // Assume all other 14 white checkers are already borne off.
  std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints, 0));
  test_board[kXPlayerId][1] = 1;
  // Place black checkers somewhere irrelevant (e.g., pos 11, head)
  test_board[kOPlayerId][11] = 15;
  std::vector<int> dice = {1, 2}; // Dice 1 and 2
  std::vector<int> scores = {14, 0}; // White has 14 borne off

  // Set White to move
  lnstate->SetState(kXPlayerId, false, dice, scores, test_board);

  // --- Verification ---
  // Possible moves for checker at pos 1:
  // - Die 1: 1 -> 0 (Valid, pos 0 is empty)
  // - Die 2: 1 -> -1 (Bear off - Valid, all checkers home, die >= exact roll 2)
  // Can play both dice? YES.
  // - Sequence: Move 1->0 (die 1). State: checker at 0. Play die 2? 0 -> -2 (Bear off). Valid. Uses dice {1, 2}.
  // Rule: Must play maximum number of dice. The sequence move is mandatory.

  auto legal_actions = lnstate->LegalActions();

  // Expect exactly one legal action corresponding to the mandatory sequence.
  SPIEL_CHECK_EQ(legal_actions.size(), 1);

  // Decode the single legal action
  std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actions[0]);

  // DEBUG: Print decoded moves
  std::cout << "DEBUG: Decoded moves for action " << legal_actions[0] << ":" << std::endl;
  for(const auto& move : moves) {
      std::cout << "  pos=" << move.pos << ", to_pos=" << move.to_pos << ", die=" << move.die << std::endl;
  }

  // Verify the sequence is (1->0, die=1) and (0->-2, die=2)
  // Note: The order within the decoded 'moves' vector depends on the encoding logic
  // (CheckerMovesToSpielMove sorts by die value descending for non-pass),
  // so the bear off move (die=2) might appear first in the decoded vector.
  SPIEL_CHECK_EQ(moves.size(), 2);

  bool found_1_to_0_d1 = false;
  bool found_0_to_off_d2 = false;

  for(const auto& move : moves) {
      // Check for the first part of the sequence
      if (move.pos == 1 && move.die == 1 && move.to_pos == 0) {
          found_1_to_0_d1 = true;
      }
      // Check for the second part of the sequence (bear off from pos 0 with die 2)
      // IsOff checks the target position based on player.
      if (move.pos == 0 && move.die == 2 && lnstate->IsOff(kXPlayerId, move.to_pos)) {
          found_0_to_off_d2 = true;
      }
  }

  // Check if BOTH parts of the expected sequence were found within the decoded action
  SPIEL_CHECK_TRUE(found_1_to_0_d1);
  SPIEL_CHECK_TRUE(found_0_to_off_d2);
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

// Test function to validate pass move behavior
void PassMoveBehaviorTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

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
}

// Add the new test function in the anonymous namespace
void VerifySingleDiePlayBehavior() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<int> dice = {5, 3}; // Higher=5, Lower=3

  // --- Scenario 1: Only Higher Die (5) is playable (due to rule) ---
  {
    std::cout << "  Testing Scenario 1 (Both Singles Playable, MaxLen=1, Force Higher)..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][5] = 1; // White checker at pos 5, can move 5->0 (d5)
    board[kXPlayerId][3] = 1; // White checker at pos 3, can move 3->0 (d3)
    board[kOPlayerId][2] = 1; // Black blocks 5->2 (d3)

    // Verify sequences are blocked:
    // 1. Try 5->0 (d5): State W@0,3 B@2. Dice {11, 3}. Next moves:
    //    - 3->0(d3): Blocked by W@0.
    //    - 0->-3(d3): Blocked by W@3. -> No 2nd move.
    // 2. Try 3->0 (d3): State W@0,5 B@2. Dice {5, 9}. Next moves:
    //    - 5->0(d5): Blocked by W@0.
    //    - 0->-4(d5): Blocked by W@5. -> No 2nd move.
    // Result: Max sequence length is 1. max_non_pass = 1. Rule applies.

    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);

    // Expected: Only the higher die move 5->0 (d5) should be legal.
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have a move

    bool found_correct_action = false;
    bool found_incorrect_action = false; // Track if lower die move was found
    SPIEL_CHECK_GE(legal_actions.size(), 1); // Should have at least one action

    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      // Encoded action usually has 2 moves (1 actual, 1 pass padding)
      // For doubles, it can have up to 4. Check structure carefully.
      SPIEL_GE(moves.size(), 2); // Should have at least 2, maybe more for doubles encoding structure

      int non_pass_count = 0;
      CheckerMove non_pass_move = kPassMove;
      for (const auto& move : moves) {
        if (move.pos != kPassPos) {
          non_pass_count++;
          non_pass_move = move;
        }
      }

      // Because max_non_pass is 1, the *encoded action* should represent exactly one non-pass move.
      SPIEL_CHECK_EQ(non_pass_count, 1);

      // Check if this single non-pass move uses the higher die (5) from pos 5.
      if (non_pass_move.pos == 5 && non_pass_move.die == 5) {
        found_correct_action = true;
      }
      // Check if it incorrectly uses the lower die (3) from pos 3.
      if (non_pass_move.pos == 3 && non_pass_move.die == 3) {
        found_incorrect_action = true;
      }
    }
    // Verify the higher die move was found AND the lower die move was NOT found.
    SPIEL_CHECK_TRUE(found_correct_action);
    SPIEL_CHECK_FALSE(found_incorrect_action);
    std::cout << "  Scenario 1: Passed\n";
  }

  // --- Scenario 2: Only Lower Die (3) is playable ---
  {
    std::cout << "  Testing Scenario 2 (Only Lower Playable, MaxLen=1)..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][5] = 1; // White checker at pos 5
    board[kOPlayerId][0] = 1; // Black blocks target of die 5 (5-5=0)
    // Add another white checker to prevent bear-off after moving 5->2, ensuring max_len=1
    board[kXPlayerId][3] = 1; // Prevents 2 -> -3 bear off with die 5
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);

    // Expected: Only move possible is 5->2 (d3). Max non-pass = 1. Rule applies.
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have a move

    bool found_correct_action = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      SPIEL_CHECK_EQ(moves.size(), 2); // Encoded action always has 2 moves (1 actual, 1 pass padding)

      int non_pass_count = 0;
      CheckerMove non_pass_move = kPassMove;
      for (const auto& move : moves) {
        if (move.pos != kPassPos) {
          non_pass_count++;
          non_pass_move = move;
        }
      }

      // Expect exactly one non-pass move, using the lower die (3)
      if (non_pass_count == 1 && non_pass_move.pos == 5 && non_pass_move.die == 3) {
        found_correct_action = true;
      }
    }
    SPIEL_CHECK_TRUE(found_correct_action); // Verify the correct single move action was generated
    std::cout << "  Scenario 2: Passed\n";
  }

  // --- Scenario 3: Both dice individually playable, max_non_pass=1 -> Force Higher ---
  // This scenario aims to test the case where individual moves with both dice are
  // possible from the initial state, but no sequence of two moves is possible,
  // thus max_non_pass = 1. The rule dictates only the higher die move is legal.
  {
    std::cout << "  Testing Scenario 3 (Both Singles Playable, MaxLen=1, Force Higher)..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    // White can play 5->2 (d3) or 8->3 (d5) individually.
    board[kXPlayerId][5] = 1;
    board[kXPlayerId][8] = 1;
    // Block alternative moves AND sequences
    board[kOPlayerId][0] = 1; // Blocks 5->0 (d5)
    board[kOPlayerId][5] = 1; // Blocks 8->5 (d3)
    // Block positions needed for sequences
    board[kOPlayerId][3] = 1; // Blocks 8->3(d5) if 5->2(d3) was played first
    board[kOPlayerId][2] = 1; // Blocks 5->2(d3) if 8->3(d5) was played first
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);
    // Expected: Individual moves 5->2(d3) and 8->3(d5) are possible. No sequences possible.
    // max_non_pass should be 1. Rule applies, force higher die (d5).

    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());

    bool found_move_d5 = false; // Check if the mandatory higher die move exists
    bool found_move_d3 = false; // Check if the forbidden lower die move exists

    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      // Check the *actual* played move in the action encoding
      for(const auto& move : moves) {
        if (move.pos != kPassPos) { // Find the non-pass move
          if (move.die == 5) found_move_d5 = true;
          if (move.die == 3) found_move_d3 = true;
          break; // Only check the first non-pass move found
        }
      }
    }

    // If the higher-die rule was correctly applied (assuming the setup yielded max_non_pass = 1),
    // only actions using die 5 should be present.
    SPIEL_CHECK_TRUE(found_move_d5);  // The higher die move MUST be possible/present
    SPIEL_CHECK_FALSE(found_move_d3); // The lower die move MUST NOT be present
    std::cout << "  Scenario 3: Passed\n";
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
  VerifySingleDiePlayBehavior();
  
  std::cout << "✓ All action encoding tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
 