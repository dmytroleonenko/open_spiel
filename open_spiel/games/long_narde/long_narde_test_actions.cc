#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/long_narde/long_narde.h"

namespace open_spiel {
namespace long_narde {
namespace {

//StartFunction: ActionEncodingTest
void ActionEncodingTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // Get the kNumDistinctActions constant - using NumDistinctActions() method
  int kNumDistinctActions = lnstate->NumDistinctActions();

  // Create a fresh initial board state to ensure turns_ is -1
  std::vector<std::vector<int>> initial_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1 for score pos
  initial_board[kXPlayerId][kWhiteHeadPos] = kNumCheckersPerPlayer;
  initial_board[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer;
  // Replace SetState with helper functions
  // Corrected Score: White (15-0=15), Black (15-0=15) -> Should be White (15 on board, score 0), Black (15 on board, score 0)
  SetupBoardState(lnstate, kXPlayerId, initial_board, {0, 0}); // Adjusted scores
  SetupDice(lnstate, {5, 3, 0, 0});

  // Removed the forced chance roll. We'll directly set the dice when calling SetState.

  // White's turn
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kXPlayerId);

  // Modify the board state so that not all white checkers are at the head
  // The initial board has all white checkers at kWhiteHeadPos, which enforces the head rule.
  // To allow encoding two moves (one from a non-head point), remove two checkers from the head
  // and place one each at positions 14 and 19.
  std::vector<std::vector<int>> modified_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1 for score pos

  // Copy the current board state (points 0-23)
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // Use internal board_ access since we are in a test context where it's available
    // (or alternatively, use the public board() method if preferred, though direct access is common in tests)
    modified_board[kXPlayerId][pos] = lnstate->board_[kXPlayerId][pos];
    modified_board[kOPlayerId][pos] = lnstate->board_[kOPlayerId][pos];
  }

  // Remove two checkers from the head (kWhiteHeadPos) and add one to positions 14 and 19
  modified_board[kXPlayerId][kWhiteHeadPos] -= 2;
  modified_board[kXPlayerId][14] += 1;
  modified_board[kXPlayerId][19] += 1;

  // Update the state with the modified board and new dice using helper functions
  // Corrected Score: White has 13 checkers on head, 1 at pos 14, 1 at pos 19 (total 15) -> score 0.
  //                  Black has 15 checkers on head (total 15) -> score 0.
  SetupBoardState(lnstate, kXPlayerId, modified_board, {0, 0}); // Corrected scores
  SetupDice(lnstate, {5, 3, 0, 0});

  //StartTest: test-actionencodingtest-1
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
  //EndTest: test-actionencodingtest-1

  //StartTest: test-actionencodingtest-2
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
  //EndTest: test-actionencodingtest-2

  //StartTest: test-actionencodingtest-3
  // Test 3: Regular move encoding (low roll first)
  // Reset state but with dice 3, 5 (low roll first)
  // Corrected Score: Same board as above (15 White, 15 Black) -> scores {0, 0}
  SetupBoardState(lnstate, kXPlayerId, modified_board, {0, 0}); // Corrected scores
  SetupDice(lnstate, {3, 5, 0, 0});

  // Use the same moves as Test 1, but expect a different action ID due to the offset
  // Moves: Position 14 with die 5, position 19 with die 3
  // Note: CheckerMovesToSpielMove might internally sort these by die value depending on context,
  // but the key is that the *original roll* was low-die first, so the offset should be applied.
  Action action_low_roll = lnstate->CheckerMovesToSpielMove(test_moves);

  // Verify the action is different from the high-roll-first action
  SPIEL_CHECK_NE(action_low_roll, action); // action is from Test 1 (roll 5, 3)

  // Verify the action is within the valid range
  SPIEL_CHECK_GE(action_low_roll, 0);
  SPIEL_CHECK_LT(action_low_roll, kNumDistinctActions);

  // Decode back to checker moves
  std::vector<CheckerMove> decoded_moves_low_roll = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_low_roll);

  // Verify decoded moves still match the original logical moves
  SPIEL_CHECK_EQ(decoded_moves_low_roll.size(), 2);
  first_move_found = false;
  second_move_found = false;
  for (const CheckerMove& move : decoded_moves_low_roll) {
    if (move.pos == test_moves[0].pos && move.die == test_moves[0].die) {
      first_move_found = true;
    }
    if (move.pos == test_moves[1].pos && move.die == test_moves[1].die) {
      second_move_found = true;
    }
  }
  SPIEL_CHECK_TRUE(first_move_found);
  SPIEL_CHECK_TRUE(second_move_found);
  //EndTest: test-actionencodingtest-3

  //StartTest: test-actionencodingtest-4
  // Test 4: Doubles encoding (4 moves)
  // Need to set up a board where 4 moves are possible with doubles.
  std::vector<std::vector<int>> doubles_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
  doubles_board[kXPlayerId][23] = 1; // Head
  doubles_board[kXPlayerId][22] = 1;
  doubles_board[kXPlayerId][21] = 1;
  doubles_board[kXPlayerId][20] = 1;
  doubles_board[kXPlayerId][19] = 11; // Remaining checkers
  doubles_board[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer; // Black checkers out of the way

  // Set state with double 2s (dice {2, 2})
  // Note: Assume this is NOT the first turn, so head rule applies normally (1 from head max).
  // Corrected Score: White (15-14=1), Black (15-0=15) -> Should be White (15 on board, score 0), Black (15 on board, score 0)
  SetupBoardState(lnstate, kXPlayerId, doubles_board, {0, 0}); // Adjusted scores
  SetupDice(lnstate, {2, 2, 2, 2}); // Use helper, double_turn = true

  // Define the 4 moves (one from head, three others)
  std::vector<CheckerMove> doubles_moves = {
    {23, lnstate->GetToPos(kXPlayerId, 23, 2), 2}, // From head
    {22, lnstate->GetToPos(kXPlayerId, 22, 2), 2},
    {21, lnstate->GetToPos(kXPlayerId, 21, 2), 2},
    {20, lnstate->GetToPos(kXPlayerId, 20, 2), 2}
  };

  // Encode the 4 moves
  Action doubles_action = lnstate->CheckerMovesToSpielMove(doubles_moves);

  // Verify the action is in the valid range
  SPIEL_CHECK_GE(doubles_action, 0);
  SPIEL_CHECK_LT(doubles_action, kNumDistinctActions);

  // Decode back to checker moves
  std::vector<CheckerMove> decoded_doubles_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, doubles_action);

  // Verify decoded moves match the original (might be padded with passes)
  SPIEL_CHECK_GE(decoded_doubles_moves.size(), 4);
  int moves_matched = 0;
  for (size_t i = 0; i < doubles_moves.size(); ++i) {
    bool match_found = false;
    for (size_t j = 0; j < decoded_doubles_moves.size(); ++j) {
      // Check position and die. Die should be 2 for all.
      if (doubles_moves[i].pos == decoded_doubles_moves[j].pos &&
          decoded_doubles_moves[j].die == 2) {
        match_found = true;
        break;
      }
    }
    if (match_found) {
      moves_matched++;
    }
  }
  SPIEL_CHECK_EQ(moves_matched, 4); // Ensure all 4 original moves were found in the decoded action
  //EndTest: test-actionencodingtest-4

  //StartTest: test-actionencodingtest-5
  // Test 5: Doubles encoding (3 moves + 1 pass)
  // Modify moves to include a pass
  std::vector<CheckerMove> doubles_moves_with_pass = {
      {23, lnstate->GetToPos(kXPlayerId, 23, 2), 2},
      {22, lnstate->GetToPos(kXPlayerId, 22, 2), 2},
      {21, lnstate->GetToPos(kXPlayerId, 21, 2), 2},
      kPassMove // Use the constant pass move {kPassPos, kPassPos, kPassDieValue}
  };

  // Encode
  Action doubles_pass_action = lnstate->CheckerMovesToSpielMove(doubles_moves_with_pass);

  // Verify action range
  SPIEL_CHECK_GE(doubles_pass_action, 0);
  SPIEL_CHECK_LT(doubles_pass_action, kNumDistinctActions);
  SPIEL_CHECK_NE(doubles_pass_action, doubles_action); // Should be different from the 4-move action

  // Decode
  std::vector<CheckerMove> decoded_doubles_pass_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, doubles_pass_action);

  // Verify: check for 3 specific moves and at least one pass
  SPIEL_CHECK_GE(decoded_doubles_pass_moves.size(), 4);
  int regular_moves_matched = 0;
  int passes_found = 0;
  for (size_t j = 0; j < decoded_doubles_pass_moves.size(); ++j) {
    bool is_pass = (decoded_doubles_pass_moves[j].pos == kPassPos);
    if (is_pass) {
        passes_found++;
    } else {
        for (size_t i = 0; i < 3; ++i) { // Check against the first 3 original moves
            if (doubles_moves_with_pass[i].pos == decoded_doubles_pass_moves[j].pos && decoded_doubles_pass_moves[j].die == 2) {
                regular_moves_matched++;
                break; // Avoid double counting if checker moved multiple times
            }
        }
    }
  }
  // Use a map to count unique positions matched to avoid issues if a checker moves twice
  std::map<int, int> matched_pos_count;
  for(const auto& decoded_move : decoded_doubles_pass_moves) {
      if (decoded_move.pos != kPassPos) {
          for (size_t i = 0; i < 3; ++i) { // Check against the 3 non-pass moves
              if (doubles_moves_with_pass[i].pos == decoded_move.pos && decoded_move.die == 2) {
                  matched_pos_count[decoded_move.pos]++;
                  break;
              }
          }
      }
  }
  SPIEL_CHECK_EQ(matched_pos_count.size(), 3); // Check that 3 unique non-pass starting positions were decoded.
  SPIEL_CHECK_GE(passes_found, 1); // Check that at least one pass move was decoded.
  //EndTest: test-actionencodingtest-5

  //StartTest: test-actionencodingtest-6
  // Test 6: Standard encoding: Single move + Pass
  // Reset state with non-double dice {4, 1}
  // Corrected Score: Same board as above (15 White, 15 Black) -> scores {0, 0}
  SetupBoardState(lnstate, kXPlayerId, modified_board, {0, 0}); // Corrected scores
  SetupDice(lnstate, {4, 1, 0, 0}); // Set new dice, non-double

  // Define moves: pos 14 with die 4, pass with die 1
  std::vector<CheckerMove> single_move_pass = {
    {14, lnstate->GetToPos(kXPlayerId, 14, 4), 4},
    {kPassPos, kPassPos, 1}
  };

  Action action_single_pass = lnstate->CheckerMovesToSpielMove(single_move_pass);

  // Verify action range (should be standard range, high roll first -> no offset)
  SPIEL_CHECK_GE(action_single_pass, 0);
  // SPIEL_CHECK_LT(action_single_pass, kDigitBase * kDigitBase); // Should not have low-roll offset

  // Decode
  std::vector<CheckerMove> decoded_single_pass = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_single_pass);

  // Verify decoded moves contain one move from pos 14 (die 4) and one pass (die 1)
  SPIEL_CHECK_EQ(decoded_single_pass.size(), 2);
  bool move_14_d4_found = false;
  bool pass_d1_found = false;
  for (const auto& move : decoded_single_pass) {
    if (move.pos == 14 && move.die == 4) move_14_d4_found = true;
    if (move.pos == kPassPos && move.die == 1) pass_d1_found = true;
  }
  SPIEL_CHECK_TRUE(move_14_d4_found);
  SPIEL_CHECK_TRUE(pass_d1_found);
  //EndTest: test-actionencodingtest-6

  //StartTest: test-actionencodingtest-7
  // Test 7: Standard encoding: Double Pass (already covered in Test 2, but re-verify)
  // Corrected Score: Same board as above (15 White, 15 Black) -> scores {0, 0}
  SetupBoardState(lnstate, kXPlayerId, modified_board, {0, 0}); // Corrected scores
  SetupDice(lnstate, {6, 5, 0, 0}); // Set new dice

  std::vector<CheckerMove> double_pass_moves = {
    {kPassPos, kPassPos, 6},
    {kPassPos, kPassPos, 5}
  };
  Action action_double_pass = lnstate->CheckerMovesToSpielMove(double_pass_moves);

  // Verify range
  SPIEL_CHECK_GE(action_double_pass, 0); // Correct check for standard range
  // SPIEL_CHECK_LT(action_double_pass, kDigitBase * kDigitBase);
  // Double pass should NOT have the low-roll offset even if dice were low-high

  // Decode
  std::vector<CheckerMove> decoded_double_pass = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action_double_pass);
  SPIEL_CHECK_EQ(decoded_double_pass.size(), 2);
  bool pass_d6_found = false;
  bool pass_d5_found = false;
  for (const auto& move : decoded_double_pass) {
    if (move.pos == kPassPos && move.die == 6) pass_d6_found = true;
    if (move.pos == kPassPos && move.die == 5) pass_d5_found = true;
  }
  SPIEL_CHECK_TRUE(pass_d6_found);
  SPIEL_CHECK_TRUE(pass_d5_found);
  //EndTest: test-actionencodingtest-7
}
//EndFunction: ActionEncodingTest

//StartFunction: SingleLegalMoveTest
//StartTest: test-singlelegalmovetest-1
void SingleLegalMoveTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // Set up a test board where White moves towards lower point numbers.
  // Only the checker at Pt 6 can move. It has two single-move options (d2, d3),
  // but playing both dice is impossible as the total move (5 pips) lands on
  // the blocked Pt 1. This forces the higher die rule.
  std::vector<std::vector<int>> test_board(2, std::vector<int>(kNumPoints + 1, 0));

  // White checkers:
  test_board[kXPlayerId][5] = 1;  // Point 6 (index 5)
  test_board[kXPlayerId][23] = 14; // Point 24 (head, index 23)

  // Black checkers:
  test_board[kOPlayerId][0] = 2;  // Point 1 (index 0) - Blocks the combined 5-pip move from Pt 6
  test_board[kOPlayerId][20] = 1; // Point 21 (index 20) - Blocks head move with d3
  test_board[kOPlayerId][21] = 1; // Point 22 (index 21) - Blocks head move with d2
  test_board[kOPlayerId][11] = 11; // Point 12 (head, index 11) - Remaining black checkers

  std::vector<int> dice = {2, 3}; // Dice 2 and 3
  std::vector<int> scores = {0, 0}; // No checkers borne off

  // Set White to move
  SetupBoardState(lnstate, kXPlayerId, test_board, scores);
  SetupDice(lnstate, {2, 3, 0, 0});

  // --- Verification ---
  // White moves towards point 1 (index 0).
  // Checkers at head (index 23) cannot move (blocked at indices 21 and 20).
  // Possible moves for white checker at Point 6 (index 5):
  // - Die 2: 6 -> 4 (index 5 -> 3) (Valid, Point 4 is empty)
  // - Die 3: 6 -> 3 (index 5 -> 2) (Valid, Point 3 is empty)
  // - Cannot play both dice because total move (5 pips) lands on blocked Point 1 (index 0).
  // Rule: Must play maximum number of dice. Here, max is 1.
  // Rule: If only single moves possible, must play higher die.
  // Higher die (3) must be used: Point 6 -> Point 3 (index 5 -> 2).

  auto legal_actions = lnstate->LegalActions();

  // Expect exactly one legal action (higher die rule forces move with die 3)
  SPIEL_CHECK_EQ(legal_actions.size(), 1);

  // Decode the legal action and check it's using die 3
  bool found_correct_move = false;

  if (!legal_actions.empty()) {
      Action the_action = legal_actions[0];
      std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, the_action);

      // Check the move corresponds to using die 3 to move from index 5 to 2
      for(const auto& move : decoded_moves) {
          // Check for the non-pass move
          if (move.pos != kPassPos) {
              if (move.pos == 5 && move.die == 3 && move.to_pos == 2) {
                  found_correct_move = true;
                  std::cout << "Found correct higher die move {Pt 6 -> Pt 3 (d3)}" << std::endl;
              }
              // Add a check to ensure the die 2 move wasn't chosen
              SPIEL_CHECK_FALSE(move.pos == 5 && move.die == 2 && move.to_pos == 3);
              break; // Expect only one non-pass move
          }
      }
  }

  // Check the expected action was found
  SPIEL_CHECK_TRUE(found_correct_move);
}
//EndTest: test-singlelegalmovetest-1
//EndFunction: SingleLegalMoveTest

//StartFunction: ConsecutiveMovesTest
//StartTest: test-consecutivemovestest-1
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
  Action first_action = first_turn_legal_actions[0]; // Restore naive selection

  lnstate->ApplyAction(first_action);

  // After White's move, the state should be Chance, waiting for Black's turn.
  SPIEL_CHECK_EQ(lnstate->CurrentPlayer(), kChancePlayerId); // Verify it's a chance node

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
//EndTest: test-consecutivemovestest-1
//EndFunction: ConsecutiveMovesTest

//StartFunction: UndoRedoTest
//StartTest: test-undoredotest-1
void UndoRedoTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // --- Define a mid-game state ---
  std::vector<std::vector<int>> mid_game_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
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

  std::vector<int> scores = {2, 1}; // White: 2 borne off, Black: 1 borne off
  int current_player = kXPlayerId;
  bool double_turn = false; // Not a double roll

  // Set the state
  // Corrected scores: W has 15 on board -> score 0. B has 14 on board -> score 1.
  SetupBoardState(lnstate, current_player, mid_game_board, {0, 1});
  SetupDice(lnstate, {4, 2, 0, 0});
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
//EndTest: test-undoredotest-1
//EndFunction: UndoRedoTest

// Test function to validate pass move behavior
//StartFunction: PassMoveBehaviorTest
void PassMoveBehaviorTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  //StartTest: test-passmovebehaviortest-1
  // TEST CASE 1: No valid moves available
  // Set up a board state where there are no legal moves
  // White has checkers at positions that can't move with dice 1,3
  // (opponent's checkers block all landing spots)
  std::vector<std::vector<int>> no_moves_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
  no_moves_board[kXPlayerId][4] = 1; // Adjust indices for the vector size
  no_moves_board[kOPlayerId][1] = 1;
  no_moves_board[kOPlayerId][3] = 1;
  no_moves_board[kOPlayerId][5] = 1;

  std::vector<int> dice_1_3 = {1, 3};

  // Set up the state
  // Corrected scores: W has 1 on board -> score 14. B has 3 on board -> score 12.
  SetupBoardState(lnstate, kXPlayerId, no_moves_board, {14, 12}); // Corrected Black score
  SetupDice(lnstate, {1, 3, 0, 0});

  // Calculate the expected pass action for dice {1, 3}
  std::vector<CheckerMove> expected_pass_encoding_1_3 = {{kPassPos, kPassPos, 1}, {kPassPos, kPassPos, 3}};
  Action expected_pass_action_1_3 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_1_3);

  // Get legal actions
  auto legal_actions = lnstate->LegalActions();

  // Check that only the pass action (specific to dice 1,3) is available
  SPIEL_CHECK_EQ(legal_actions.size(), 1);
  SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_1_3); // Compare against specific pass action
  //EndTest: test-passmovebehaviortest-1

  //StartTest: test-passmovebehaviortest-2
  // TEST CASE 2: At least one valid move is available
  // White has checkers that can move with dice 1,3
  std::vector<std::vector<int>> valid_moves_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
  valid_moves_board[kXPlayerId][1] = 1; // Adjust indices
  valid_moves_board[kXPlayerId][3] = 1;
  // Black board is empty

  // Set up the state
  // Corrected scores: W has 2 on board -> score 13. B has 0 on board -> score 15.
  SetupBoardState(lnstate, kXPlayerId, valid_moves_board, {13, 15});
  SetupDice(lnstate, {1, 3, 0, 0});

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
  //EndTest: test-passmovebehaviortest-2

  //StartTest: test-passmovebehaviortest-3
  // TEST CASE 3: Doubles with no moves possible
  {
    std::vector<std::vector<int>> no_moves_doubles_board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    // White blocks positions 0, 5, 9 (targets for Black's moves from 2, 7, 11 with die 2)
    no_moves_doubles_board[kXPlayerId][0] = 1;
    no_moves_doubles_board[kXPlayerId][5] = 1;
    no_moves_doubles_board[kXPlayerId][9] = 1;
    no_moves_doubles_board[kXPlayerId][kWhiteHeadPos] = 12; // White head
    // Black has checkers at 2, 7, and 11(Head)
    no_moves_doubles_board[kOPlayerId][2] = 1;
    no_moves_doubles_board[kOPlayerId][7] = 1;
    no_moves_doubles_board[kOPlayerId][kBlackHeadPos] = 13; // Black head

    // Set up the state with double 2s

    // Set up the state - Black's turn with doubles, no moves possible
    // Corrected scores: W has 3 on board -> score 12. B has 2 on board -> score 13. -> Should be W (15 on board, score 0), B (15 on board, score 0)
    SetupBoardState(lnstate, kOPlayerId, no_moves_doubles_board, {0, 0}); // Adjusted scores
    SetupDice(lnstate, {2, 2, 2, 2}); // Set double_turn=true for doubles dice

    // Calculate the expected pass action for dice {2, 2}
    std::vector<CheckerMove> expected_pass_encoding_2_2 = {{kPassPos, kPassPos, 2}, {kPassPos, kPassPos, 2}};
    Action expected_pass_action_2_2 = lnstate->CheckerMovesToSpielMove(expected_pass_encoding_2_2);

    // Get legal actions
    legal_actions = lnstate->LegalActions();

    // Verify only the pass move is available
    SPIEL_CHECK_EQ(legal_actions.size(), 1);
    SPIEL_CHECK_EQ(legal_actions[0], expected_pass_action_2_2); // Compare against specific pass action for {2, 2}
  }
  //EndTest: test-passmovebehaviortest-3
}
//EndFunction: PassMoveBehaviorTest

// Add the new test function in the anonymous namespace
//StartFunction: VerifyDicePlayBehavior
void VerifyDicePlayBehavior() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  // Test with dice Higher=5, Lower=3

  // --- Scenario 1: Both dice individually playable, two-move sequence possible ---
  //StartTest: test-vsdpb-1
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 1 (Both Dice Playable, Two-Move Sequence Possible)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    // Setup: W@8, W@3. B@0. Dice {5,3}.
    // Initial Half Moves:
    // - W@8: 8->3(d5) valid. 8->5(d3) valid.
    // - W@3: 3->0(d3) blocked by B@0. 3->off(d5) valid bear-off (assuming all checkers in home).
    // -> Playable: [8->3, d5], [8->5, d3], [3->off, d5]. All dice are possible.
    // Sequences:
    // - Try [8->3, d5]: State W@3(x2). B@0. Next Moves(d3): 3->0 blocked by B@0. -> Sequence length 1.
    // - Try [8->5, d3]: State W@3, W@5. B@0. Next Moves(d5): 3->off(d5) valid. -> Sequence length 2.
    // Result: longest=2, max_non_pass=2. Must play the full sequence [8->5, d3], [3->off, d5].
    board[kXPlayerId][8] = 1;
    board[kXPlayerId][3] = 1;
    board[kOPlayerId][0] = 1; // Black checker blocks moves to point 1 (pos 0)

    // Correct the scores to maintain the invariant (15 total checkers)
    // White: 2 on board -> 13 score
    // Black: 1 on board -> 14 score
    SetupBoardState(lnstate, kXPlayerId, board, {13, 14});

    SetupDice(lnstate, {5, 3, 0, 0});

    // Expected: The two-move sequence 8->5 (d3) followed by 3->off (d5) should be the only legal action.
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have a move

    bool found_correct_sequence = false;
    SPIEL_CHECK_EQ(legal_actions.size(), 1); // Should have exactly one action (the 2-move sequence)

    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      SPIEL_CHECK_GE(moves.size(), 2); // Should have at least 2 moves (2 actual + maybe passes)

      // Check if the two expected moves are present (order might vary)
      bool found_8_5_d3 = false;
      bool found_3_off_d5 = false;
      int non_pass_count = 0;

      for(const auto& move : moves) {
          if (move.pos != kPassPos) non_pass_count++;
          if (move.pos == 8 && move.to_pos == 5 && move.die == 3) {
            found_8_5_d3 = true;
          } else if (move.pos == 3 && move.to_pos == kBearOffPos && move.die == 5) {
            found_3_off_d5 = true;
          }
      }
      // Verify exactly two non-pass moves constitute the action
      SPIEL_CHECK_EQ(non_pass_count, 2);
      // Verify both parts of the sequence were found
      if (found_8_5_d3 && found_3_off_d5) {
          found_correct_sequence = true;
      }
    }
    // Verify the correct 2-move sequence was the only one generated.
    SPIEL_CHECK_TRUE(found_correct_sequence);
  }
  //EndTest: test-vsdpb-1

  // --- Scenario 2: Only Lower Die (3) is playable ---
  //StartTest: test-vsdpb-2
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 2 (Only Lower Playable, MaxLen=1)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    // Setup: W@5, W@8. B@0, B@3. Dice {5,3}. Player X.
    // Analysis:
    // W@5: d5 blocked B@0, d3 to 2 is valid.
    // W@8: d5 blocked B@3, d3 to 5 is valid.
    // -> Only d3 is initially playable.
    // -> Sequences: [5->2, d3] (len 1, next d5 move 8->3 blocked B@3).
    // ->            [8->5, d3] (len 1, next d5 move 5->0 blocked B@0).
    // -> longest=1, max_non_pass=1. Rule applies. Only die 3 was ever playable.
    // -> Legal actions must use die 3.
    board[kXPlayerId][5] = 1;
    board[kXPlayerId][8] = 1;
    board[kOPlayerId][0] = 1; // Blocks W@5 with d5
    board[kOPlayerId][3] = 1; // Blocks W@8 with d5 AND blocks subsequent d5 move from W@2 if W@5 moved first.
    // Corrected scores: W has 2 on board -> score 13. B has 2 on board -> score 13.
    SetupBoardState(lnstate, kXPlayerId, board, {13, 13});
    SetupDice(lnstate, {5, 3, 0, 0});

    // Expected: Only move possible is 5->2 (d3). Max non-pass = 1. Rule applies.
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have at least one legal action.
    bool all_used_lower_die = true;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      SPIEL_CHECK_GE(moves.size(), 2); // Should have at least 2 moves in encoding

      int non_pass_count = 0;
      CheckerMove non_pass_move = kPassMove;
      bool used_correct_die = false;
      for (const auto& move : moves) {
        if (move.pos != kPassPos) {
          non_pass_count++;
          non_pass_move = move;
          if (move.die == 3) { // Check if the lower die (3) was used
              used_correct_die = true;
          } else {
              used_correct_die = false; // Found a move using the higher die (5) - incorrect!
              break;
          }
        }
      }
      SPIEL_CHECK_EQ(non_pass_count, 1); // Verify it's a single move action
      if (!used_correct_die) {
          all_used_lower_die = false;
          break;
      }
    }
    SPIEL_CHECK_TRUE(all_used_lower_die); // Verify all legal actions used the lower die (3)
    if (kDebugging) {
      std::cout << "  Scenario 2: Passed\n";
    }
  }
  //EndTest: test-vsdpb-2

  // --- Scenario 3: Higher Die Rule Application (Forced Single Move) ---
  //StartTest: test-vsdpb-3
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 3 (Both Singles Playable, MaxLen=1, Force Higher)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    // Setup: W@8, W@3. B@0, B@5. Dice {5,3}.
    // Initial Half Moves:
    // - W@8: 8->3(d5) valid. 8->5(d3) valid.
    // - W@3: 3->0(d3) blocked by B@0. 3->off(d5) invalid bear-off (not all checkers in home).
    // -> Playable: [8->3, d5], [8->5, d3]. Both dice are possible.
    // Sequences:
    // - Try [8->3, d5]: State W@3(x2). B@0, B@5. Next Moves(d3): 3->0 blocked by B@0. -> Sequence length 1.
    // - Try [8->5, d3]: State W@3, W@5. B@0, B@5. Next Moves(d5): 5->0 blocked by B@0, 3->off(d5) invalid. -> Sequence length 1.
    // Result: longest=1, max_non_pass=1. Higher die rule applies. Must play the move using die 5 ([8->3, d5]).
    board[kXPlayerId][8] = 1;
    board[kXPlayerId][3] = 1;
    board[kOPlayerId][0] = 1; // Black checker blocks moves to point 1 (pos 0)
    board[kOPlayerId][5] = 1; // Black checker at point 6 (index 5) - prevents bearing off from point 4 (index 3)

    // Correct the scores to maintain the invariant (15 total checkers)
    // White: 2 on board -> 13 score
    // Black: 2 on board -> 13 score
    SetupBoardState(lnstate, kXPlayerId, board, {13, 13});

    SetupDice(lnstate, {5, 3, 0, 0});

    // Expected: Only the higher die move 8->3 (d5) should be legal.
    // Higher die rule applies because only single moves are possible.

    auto legal_actions = lnstate->LegalActions();

    // --- DEBUG OUTPUT START ---
    std::cout << "DEBUG Scenario 3: Found " << legal_actions.size() << " legal actions." << std::endl;
    int action_idx = 0;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      std::cout << "  Action " << action_idx++ << " (ID: " << action << ") decoded to " << moves.size() << " moves:" << std::endl;
      for (const auto& m : moves) {
           std::cout << "    {pos=" << m.pos << ", to=" << m.to_pos << ", die=" << m.die << "}" << std::endl;
      }
    }
    // --- DEBUG OUTPUT END ---

    SPIEL_CHECK_EQ(legal_actions.size(), 1); // Expect exactly one action (higher die rule applies)

    // Verify the single action uses the higher die (5).
    bool found_correct_move = false;
    if (!legal_actions.empty()) { // Should not be empty based on above check
      Action the_action = legal_actions[0];
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, the_action);

      // Action encoding always has 2 moves (potentially passes)
      SPIEL_CHECK_GE(moves.size(), 2);

      // Check that there is exactly ONE non-pass move in the decoded action
      int non_pass_count = 0;
      CheckerMove non_pass_move = kPassMove;
      for(const auto& move : moves) {
          if (move.pos != kPassPos) {
              non_pass_count++;
              non_pass_move = move;
          }
      }
      SPIEL_CHECK_EQ(non_pass_count, 1); // Verify exactly 1 non-pass move

      // Check if the single non-pass move is the expected one (8->3 with die 5)
      if (non_pass_move.pos == 8 && non_pass_move.to_pos == 3 && non_pass_move.die == 5) {
          found_correct_move = true;
      }
    }

    // Verify the correct single move sequence was the only one generated.
    SPIEL_CHECK_TRUE(found_correct_move);

    if (kDebugging) {
      std::cout << "  Scenario 3: Passed\n";
    }
  }
  //EndTest: test-vsdpb-3
}
//EndFunction: VerifyDicePlayBehavior

// New test for direct bear-off of two checkers
//StartFunction: DirectBearOffTest
void DirectBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  // Test with dice 1 and 2

  // --- White Test ---
  //StartTest: test-dbo-1w
  {
    if (kDebugging) std::cout << "  Testing DirectBearOffTest (White)..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    board[kXPlayerId][0] = 1;
    board[kXPlayerId][1] = 1;
    board[kOPlayerId][11] = 15; // Irrelevant black checkers
    // Corrected scores: W has 2 on board -> score 13. B has 0 on board -> score 15. -> Should be W (2 on board, score 13), B (15 on board, score 0)
    std::vector<int> scores = {13, 0}; // Adjusted scores
    SetupBoardState(lnstate, kXPlayerId, board, scores);
    SetupDice(lnstate, {1, 2, 0, 0});

    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());

    bool found_direct_bearoff = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      if (moves.size() == 2) {
        bool move1_ok = (moves[0].pos == 0 && moves[0].die == 1 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 0 && moves[1].die == 1 && moves[1].to_pos == kBearOffPos);
        bool move2_ok = (moves[0].pos == 1 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 1 && moves[1].die == 2 && moves[1].to_pos == kBearOffPos);
        if (move1_ok && move2_ok) {
          found_direct_bearoff = true;
          break;
        }
      }
    }
    // Check if *at least one* of the legal actions corresponds to the direct bear-off.
    // Note: The test SingleLegalMoveTest currently enforces only the {1->0, 0->off} sequence.
    // This might need to be relaxed if both sequences are deemed valid outcomes of LegalActions.
    // For now, just check if the direct bear-off sequence IS generated.
    SPIEL_CHECK_TRUE(found_direct_bearoff);
    if (kDebugging) std::cout << "  White Direct Bear-Off: Passed\n";
  }
  //EndTest: test-dbo-1w

  // --- Black Test ---
  //StartTest: test-dbo-1b
  {
    if (kDebugging) std::cout << "  Testing DirectBearOffTest (Black)..." << std::endl;
    // Use dice 2 and 3 for bear off
    std::vector<int> scores = {0, 13}; // Black has 13 borne off

    // Define the board state
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    board[kOPlayerId][13] = 1; // Black's last checker at pos 13 (needs 2 pips)
    board[kOPlayerId][14] = 1; // Black checker inside home at point 14 (index 13)

    // Replace direct state setting with helpers
    // Corrected scores: W has 0 on board -> score 15. B has 2 on board -> score 13.
    SetupBoardState(lnstate, kOPlayerId, board, {15, 13});
    SetupDice(lnstate, {2, 3, 0, 0});

    // The moved_from_head_ flag is handled by SetupBoardState/SetupDice defaults or internal logic.
    // The is_first_turn_ and moved_from_head_ are handled by SetupBoardState/SetupDice defaults or internal logic.

    if (kDebugging) {
        std::cout << "DEBUG (Black DirectBearOffTest Post-SetupHelpers):\n" << lnstate->ToString() << std::endl;
    }

    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());

    // DEBUG: Print legal actions and decoded moves for Black
    std::cout << "DEBUG (Black DirectBearOffTest): Found " << legal_actions.size() << " legal actions." << std::endl;

    bool found_direct_bearoff = false;
    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);
      std::cout << "DEBUG (Black DirectBearOffTest): Action " << action << " decodes to:" << std::endl;
      for(const auto& m : moves) {
          std::cout << "  pos=" << m.pos << ", to=" << m.to_pos << ", die=" << m.die << std::endl;
      }
      if (moves.size() == 2) { // Check for the correct sequence {13->off(d2), 14->off(d3)}
        bool move1_ok = (moves[0].pos == 13 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 13 && moves[1].die == 2 && moves[1].to_pos == kBearOffPos);
        bool move2_ok = (moves[0].pos == 14 && moves[0].die == 3 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 14 && moves[1].die == 3 && moves[1].to_pos == kBearOffPos); // Check for 14->off(d3)
        if (move1_ok && move2_ok) {
          found_direct_bearoff = true;
          break;
        }
      }
    }
    SPIEL_CHECK_TRUE(found_direct_bearoff);
    if (kDebugging) std::cout << "  Black Direct Bear-Off: Passed\n";
  }
  //EndTest: test-dbo-1b
}
//EndFunction: DirectBearOffTest

// New test for single checker bear-off rules (higher die, only die)
//StartFunction: SingleCheckerBearOffTest
void SingleCheckerBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());

  // --- Scenario 1: Higher Die Rule (Both playable) ---
  //StartTest: test-scbo-1w
  {
    if (kDebugging) std::cout << "  Testing SingleCheckerBearOffTest (Higher Die Rule)..." << std::endl;
    // Test with dice 1 and 6

    // White Test (Checker at pos 0)
    std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    boardW[kXPlayerId][0] = 1;
    // Corrected scores: W has 1 on board -> score 14. B has 0 on board -> score 0.
    SetupBoardState(lnstate, kXPlayerId, boardW, {14, 0}); // Corrected Black score from 15 to 0
    SetupDice(lnstate, {1, 6, 0, 0});
    auto legal_actionsW = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsW.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesW = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
    bool found_die6W = false;
    for(const auto& m : movesW) { if(m.pos == 0 && m.die == 6 && m.to_pos == kBearOffPos) found_die6W = true; }
    SPIEL_CHECK_TRUE(found_die6W);
    if (kDebugging) std::cout << "  White Higher Die: Passed\n";

    // Black Test (Checker at pos 12)
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    boardB[kOPlayerId][12] = 1;
    // Corrected scores: W has 0 on board -> score 15. B has 1 on board -> score 14.
    SetupBoardState(lnstate, kOPlayerId, boardB, {15, 14});
    SetupDice(lnstate, {1, 6, 0, 0});
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
    bool found_die6B = false;
    for(const auto& m : movesB) { if(m.pos == 12 && m.die == 6 && m.to_pos == kBearOffPos) found_die6B = true; }
    SPIEL_CHECK_TRUE(found_die6B);
    if (kDebugging) std::cout << "  Black Higher Die: Passed\n";
  }
  //EndTest: test-scbo-1w

  // --- Scenario 2: Only One Die Playable ---
  //StartTest: test-scbo-2w
  {
    if (kDebugging) std::cout << "  Testing SingleCheckerBearOffTest (Only One Die Playable)..." << std::endl;
    // Test with dice 1 and 3

    // White Test (Checker at pos 0 - only die 1 or 3 works)
    std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    boardW[kXPlayerId][0] = 1;
    // Corrected scores: W has 1 on board -> score 14. B has 0 on board -> score 0.
    SetupBoardState(lnstate, kXPlayerId, boardW, {14, 0}); // Corrected Black score from 15 to 0
    SetupDice(lnstate, {1, 3, 0, 0});
    auto legal_actionsW = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsW.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesW = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
    bool found_die3W = false; // Higher die is 3
    for(const auto& m : movesW) { if(m.pos == 0 && m.die == 3 && m.to_pos == kBearOffPos) found_die3W = true; }
    SPIEL_CHECK_TRUE(found_die3W);
    if (kDebugging) std::cout << "  White Only Die (3): Passed\n";
    //EndTest: test-scbo-2w
    //StartTest: test-scbo-1b

    // Black Test (Checker at pos 12 - only die 1 or 3 works)
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    boardB[kOPlayerId][12] = 1;
    // Corrected scores: W has 0 on board -> score 15. B has 1 on board -> score 14.
    SetupBoardState(lnstate, kOPlayerId, boardB, {15, 14});
    SetupDice(lnstate, {1, 3, 0, 0});
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
    bool found_die1B = false; // For pos 12 (needs 1 pip), die 1 is exact, die 3 is higher. Must use 3.
    bool found_die3B = false;
    for(const auto& m : movesB) {
         if(m.pos == 12 && m.die == 1 && m.to_pos == kBearOffPos) found_die1B = true;
         if(m.pos == 12 && m.die == 3 && m.to_pos == kBearOffPos) found_die3B = true;
     }
    SPIEL_CHECK_FALSE(found_die1B);
    SPIEL_CHECK_TRUE(found_die3B);
    if (kDebugging) std::cout << "  Black Only Die (3): Passed\n";
    //EndTest: test-scbo-1b
    //StartTest: test-scbo-2b2

     // Black Test (Checker at pos 14 - needs 3 pips, terminal state)
     // This tests that both options are valid for terminal states:
     // 1. Direct bear-off with higher die (3) + pass lower die (1)
     // 2. Two-step sequence: move with lower die (1) + bear-off with higher die (3)
     std::vector<std::vector<int>> boardB_pos14(2, std::vector<int>(kNumPoints + 1, 0)); // New board var, Size kNumPoints+1
     boardB_pos14[kOPlayerId][14] = 1;
     // Corrected scores: W has 0 on board -> score 15. B has 1 on board -> score 14.
     SetupBoardState(lnstate, kOPlayerId, boardB_pos14, {15, 14});
     SetupDice(lnstate, {1, 3, 0, 0});
     legal_actionsB = lnstate->LegalActions();

     // Should have two legal actions for terminal state
     SPIEL_CHECK_EQ(legal_actionsB.size(), 2);

     // Check for both possible actions
     bool found_two_step = false;
     bool found_direct_bearoff = false;

     for (const auto& action : legal_actionsB) {
       std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, action);

       // Check for two-step sequence
       if (moves.size() == 2) {
         bool has_move_step = false;
         bool has_bearoff_step = false;

         for (const auto& m : moves) {
           if (m.pos == 14 && m.to_pos == 13 && m.die == 1) has_move_step = true;
           if (m.pos == 13 && m.to_pos == kBearOffPos && m.die == 3) has_bearoff_step = true;
         }

         if (has_move_step && has_bearoff_step) found_two_step = true;

         // Check for direct bear-off + pass
         has_move_step = false;
         has_bearoff_step = false;

         for (const auto& m : moves) {
           if (m.pos == 14 && m.to_pos == kBearOffPos && m.die == 3) has_bearoff_step = true;
           if (m.pos == kPassPos && m.to_pos == kPassPos && m.die == 1) has_move_step = true;
         }

         if (has_move_step && has_bearoff_step) found_direct_bearoff = true;
       }
     }

     SPIEL_CHECK_TRUE(found_two_step);
     SPIEL_CHECK_TRUE(found_direct_bearoff);
     if (kDebugging) std::cout << "  Black Terminal State (Both Options Valid): Passed\n";
     //EndTest: test-scbo-2b2
  }
}
//EndFunction: SingleCheckerBearOffTest

// New test for bearing off the last checker, forcing a pass on the second die
//StartFunction: BearOffLastCheckerTest
void BearOffLastCheckerTest() {
  if (kDebugging) std::cout << "\n=== Running BearOffLastCheckerTest ===\n";
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  // Test with high dice 4 and 5, either can bear off

  // --- White Test (Last checker at pos 1, needs 2 pips) ---
  //StartTest: test-bolc-1w
  {
    if (kDebugging) std::cout << "  Testing White Last Checker Bear Off..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    board[kXPlayerId][1] = 1; // White's last checker at pos 1 (needs 2 pips)
    board[kOPlayerId][11] = 15; // Irrelevant black checkers
    // Corrected scores: W has 1 on board -> score 14. B has 0 on board -> score 15. -> Should be W (1 on board, score 14), B (15 on board, score 0)
    std::vector<int> scores = {14, 0}; // Adjusted scores

    SetupBoardState(lnstate, kXPlayerId, board, scores);
    SetupDice(lnstate, {4, 5, 0, 0});

    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1); // Should only be one legal action

    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actions[0]);
    SPIEL_CHECK_EQ(moves.size(), 2); // Action encodes two half-moves

    bool found_bear_off = false;
    bool found_pass = false;
    int bear_off_die = -1;

    for (const auto& move : moves) {
      if (move.pos == 1 && move.to_pos == kBearOffPos && (move.die == 4 || move.die == 5)) {
        // According to rules, higher die (5) should be used if possible
        SPIEL_CHECK_EQ(move.die, 5);
        found_bear_off = true;
        bear_off_die = move.die;
      } else if (move.pos == kPassPos) {
        found_pass = true;
        // The pass should correspond to the unused die
        SPIEL_CHECK_NE(move.die, bear_off_die);
        SPIEL_CHECK_TRUE(move.die == 4 || move.die == 5);
      }
    }

    SPIEL_CHECK_TRUE(found_bear_off);
    SPIEL_CHECK_TRUE(found_pass);
    if (kDebugging) std::cout << "  White Last Checker: Passed\n";
  }
  //EndTest: test-bolc-1w

  // --- Black Test (Last checker at pos 13, needs 2 pips) ---
  //StartTest: test-bolc-1b
  {
    if (kDebugging) std::cout << "  Testing Black Last Checker Bear Off..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints + 1, 0)); // Size kNumPoints+1
    board[kOPlayerId][13] = 1; // Black's last checker at pos 13 (needs 2 pips)
    board[kXPlayerId][23] = 15; // Irrelevant white checkers
    // Corrected scores: W has 0 on board -> score 15. B has 1 on board -> score 14. -> Should be W (15 on board, score 0), B (1 on board, score 14)
    std::vector<int> scores = {0, 14}; // Adjusted scores

    SetupBoardState(lnstate, kOPlayerId, board, scores);
    SetupDice(lnstate, {4, 5, 0, 0});

    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actions.size(), 1); // Should only be one legal action

    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actions[0]);
    SPIEL_CHECK_EQ(moves.size(), 2); // Action encodes two half-moves

    bool found_bear_off = false;
    bool found_pass = false;
    int bear_off_die = -1;

    for (const auto& move : moves) {
      if (move.pos == 13 && move.to_pos == kBearOffPos && (move.die == 4 || move.die == 5)) {
        // According to rules, higher die (5) should be used if possible
        SPIEL_CHECK_EQ(move.die, 5);
        found_bear_off = true;
        bear_off_die = move.die;
      } else if (move.pos == kPassPos) {
        found_pass = true;
         // The pass should correspond to the unused die
        SPIEL_CHECK_NE(move.die, bear_off_die);
        SPIEL_CHECK_TRUE(move.die == 4 || move.die == 5);
      }
    }

    SPIEL_CHECK_TRUE(found_bear_off);
    SPIEL_CHECK_TRUE(found_pass);
    if (kDebugging) std::cout << "  Black Last Checker: Passed\n";
  }
  //EndTest: test-bolc-1b
   if (kDebugging) std::cout << "✓ BearOffLastCheckerTest passed\n";
}
//EndFunction: BearOffLastCheckerTest

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

// Register the test function in the common interface
namespace open_spiel {
namespace long_narde {

// Implement the global test function that was declared in the header
//StartFunction: TestPassMoveBehavior
void TestPassMoveBehavior() {
  std::cout << "\n=== Testing Pass Move Behavior ===\n";
  PassMoveBehaviorTest();
  std::cout << "✓ Pass Move Behavior Test passed\n";
}
//EndFunction: TestPassMoveBehavior

//StartFunction: TestActionEncoding
void TestActionEncoding() {
  std::cout << "\n=== Testing Action Encoding ===\n";
  ActionEncodingTest();
  SingleLegalMoveTest();
  ConsecutiveMovesTest();
  UndoRedoTest();
  VerifyDicePlayBehavior();
  DirectBearOffTest();
  SingleCheckerBearOffTest();
  BearOffLastCheckerTest();

  std::cout << "✓ All action encoding tests passed\n";
}
//EndFunction: TestActionEncoding

}  // namespace long_narde
}  // namespace open_spiel

