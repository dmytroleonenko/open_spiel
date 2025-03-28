#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/long_narde/long_narde.h"

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

  // Test 3: Regular move encoding (low roll first)
  // Reset state but with dice 3, 5 (low roll first)
  lnstate->SetState(kXPlayerId, false, std::vector<int>{3, 5}, scores, modified_board);

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

  // Verify the action is in the upper half of the non-doubles range, indicating the offset was applied
  SPIEL_CHECK_GE(action_low_roll, kDigitBase * kDigitBase);

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

  // Test 4: Doubles encoding (4 moves)
  // Need to set up a board where 4 moves are possible with doubles.
  // Put 4 checkers near the start for White.
  std::vector<std::vector<int>> doubles_board(2, std::vector<int>(kNumPoints, 0));
  doubles_board[kXPlayerId][23] = 1; // Head
  doubles_board[kXPlayerId][22] = 1;
  doubles_board[kXPlayerId][21] = 1;
  doubles_board[kXPlayerId][20] = 1;
  doubles_board[kXPlayerId][19] = 11; // Remaining checkers
  doubles_board[kOPlayerId][kBlackHeadPos] = kNumCheckersPerPlayer; // Black checkers out of the way

  // Set state with double 2s (dice {2, 2})
  // Note: Assume this is NOT the first turn, so head rule applies normally (1 from head max).
  lnstate->SetState(kXPlayerId, true, std::vector<int>{2, 2}, scores, doubles_board);
  lnstate->MutableIsFirstTurn() = false; // Ensure it's not the first turn

  // Define the 4 moves (one from head, three others)
  std::vector<CheckerMove> doubles_moves = {
    {23, lnstate->GetToPos(kXPlayerId, 23, 2), 2}, // From head
    {22, lnstate->GetToPos(kXPlayerId, 22, 2), 2},
    {21, lnstate->GetToPos(kXPlayerId, 21, 2), 2},
    {20, lnstate->GetToPos(kXPlayerId, 20, 2), 2}
  };

  // Constants needed from .cc file (Ideally these would be in the header too)
  const int kEncodingBaseDouble = 25;
  const int kDoublesOffset = 2 * kDigitBase * kDigitBase;

  // Encode the 4 moves
  Action doubles_action = lnstate->CheckerMovesToSpielMove(doubles_moves);

  // Verify the action is in the doubles range
  SPIEL_CHECK_GE(doubles_action, kDoublesOffset);
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
  SPIEL_CHECK_GE(doubles_pass_action, kDoublesOffset);
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
            if (doubles_moves_with_pass[i].pos == decoded_doubles_pass_moves[j].pos &&
                decoded_doubles_pass_moves[j].die == 2) {
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

  // Test 6: Standard encoding: Single move + Pass
  // Reset state with non-double dice {4, 1}
  lnstate->SetState(kXPlayerId, false, std::vector<int>{4, 1}, scores, modified_board);
  
  // Define moves: pos 14 with die 4, pass with die 1
  std::vector<CheckerMove> single_move_pass = {
    {14, lnstate->GetToPos(kXPlayerId, 14, 4), 4},
    {kPassPos, kPassPos, 1} 
  };

  Action action_single_pass = lnstate->CheckerMovesToSpielMove(single_move_pass);

  // Verify action range (should be standard range, high roll first -> no offset)
  SPIEL_CHECK_GE(action_single_pass, 0);
  SPIEL_CHECK_LT(action_single_pass, kDoublesOffset);
  SPIEL_CHECK_LT(action_single_pass, kDigitBase * kDigitBase); // Should not have low-roll offset

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

  // Test 7: Standard encoding: Double Pass (already covered in Test 2, but re-verify)
  lnstate->SetState(kXPlayerId, false, std::vector<int>{6, 5}, scores, modified_board);
  std::vector<CheckerMove> double_pass_moves = {
    {kPassPos, kPassPos, 6},
    {kPassPos, kPassPos, 5}
  };
  Action action_double_pass = lnstate->CheckerMovesToSpielMove(double_pass_moves);

  // Verify range
  SPIEL_CHECK_GE(action_double_pass, 0);
  SPIEL_CHECK_LT(action_double_pass, kDoublesOffset);
  // Double pass should NOT have the low-roll offset even if dice were low-high
  SPIEL_CHECK_LT(action_double_pass, kDigitBase * kDigitBase);

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
      // Directly check if the decoded to_pos is the bear-off position.
      if (move.pos == 0 && move.die == 2 && move.to_pos == kBearOffPos) {
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
  {
    std::vector<std::vector<int>> no_moves_doubles_board = {
      // White blocks positions 0, 5, 9 (targets for Black's moves from 2, 7, 11 with die 2)
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12}, // White (at 0, 5, 9, Head=12)
      // Black has checkers at 2, 7, and 11(Head)
      {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black (at 2, 7, Head=13)
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
}

// Add the new test function in the anonymous namespace
void VerifySingleDiePlayBehavior() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<int> dice = {5, 3}; // Higher=5, Lower=3

  // --- Scenario 1: Both dice individually playable, max_non_pass=1 -> Force Higher ---
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 1 (Both Singles Playable, MaxLen=1, Force Higher)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    // Setup: W@8, W@3. B@0. Dice {5,3}.
    // Initial Half Moves:
    // - W@8: 8->3(d5) valid. 8->5(d3) valid.
    // - W@3: 3->0(d3) blocked B@0. 3->-2(d5) invalid bear-off.
    // -> Playable: [8->3, d5], [8->5, d3]. Both dice are possible.
    // Sequences:
    // - Try [8->3, d5]: State W@3(x2). B@0. Dice {11,3}. Next Moves(d3): 3->0 blocked B@0. -> Sequence length 1.
    // - Try [8->5, d3]: State W@3, W@5. B@0. Dice {5,9}. Next Moves(d5): 5->0 blocked B@0. -> Sequence length 1.
    // Result: longest=1, max_non_pass=1. Higher die rule applies. Must play the move using die 5.
    board[kXPlayerId][8] = 1;
    board[kXPlayerId][3] = 1;
    board[kOPlayerId][0] = 1; // Black checker blocks moves to point 1 (pos 0)

    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);

    // Expected: Only the higher die move 8->3 (d5) should be legal.
    auto legal_actions = lnstate->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty()); // Should have a move

    bool found_correct_action = false;
    bool found_incorrect_action = false; // Track if lower die move was found
    SPIEL_CHECK_GE(legal_actions.size(), 1); // Should have at least one action

    for (Action action : legal_actions) {
      std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
      // Encoded action usually has 2 moves (1 actual, 1 pass padding)
      // For doubles, it can have up to 4. Check structure carefully.
      SPIEL_CHECK_GE(moves.size(), 2); // Should have at least 2, maybe more for doubles encoding structure

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

      // Check if this single non-pass move uses the higher die (5) from pos 8.
      if (non_pass_move.pos == 8 && non_pass_move.die == 5) {
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
    if (kDebugging) {
      std::cout << "  Scenario 1: Passed\n";
    }
  }

  // --- Scenario 2: Only Lower Die (3) is playable ---
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 2 (Only Lower Playable, MaxLen=1)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
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
    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);

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

  // --- Scenario 3: Both dice individually playable, max_non_pass=1 -> Force Higher ---
  {
    if (kDebugging) {
      std::cout << "  Testing Scenario 3 (Both Singles Playable, MaxLen=1, Force Higher)..." << std::endl;
    }
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    // Setup: W@8, W@3. B@0. Dice {5,3}.
    // As analyzed in Scenario 1:
    // Initial Half Moves: [8->3, d5], [8->5, d3]. Both dice possible.
    // Sequences: Only length 1 sequences are possible ([8->3, d5] and [8->5, d3]).
    // Result: longest=1, max_non_pass=1. Higher die rule applies. Must play the move using die 5 ([8->3, d5]).
    board[kXPlayerId][8] = 1;
    board[kXPlayerId][3] = 1;
    board[kOPlayerId][0] = 1; // Black checker blocks moves to point 1 (pos 0)

    lnstate->SetState(kXPlayerId, false, dice, {0, 0}, board);

    // Expected: Individual moves 8->3(d5) and 8->5(d3) are possible. No sequences possible.
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
    if (kDebugging) {
      std::cout << "  Scenario 3: Passed\n";
    }
  }
}

// New test for direct bear-off of two checkers
void DirectBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<int> dice = {1, 2}; // Dice 1 and 2

  // --- White Test --- 
  {
    if (kDebugging) std::cout << "  Testing DirectBearOffTest (White)..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][0] = 1;
    board[kXPlayerId][1] = 1;
    board[kOPlayerId][11] = 15; // Irrelevant black checkers
    std::vector<int> scores = {13, 0}; // White has 13 borne off
    lnstate->SetState(kXPlayerId, false, dice, scores, board);

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

  // --- Black Test --- 
  {
    if (kDebugging) std::cout << "  Testing DirectBearOffTest (Black)..." << std::endl;
    std::vector<int> dice = {1, 2}; // Use dice 1 and 2 for bear off
    std::vector<int> scores = {0, 13}; // Black has 13 borne off

    // Directly set the state
    lnstate->board_.assign(2, std::vector<int>(kNumPoints, 0)); // Clear board
    lnstate->board_[kOPlayerId][12] = 1; // Black checker inside home at point 13 (index 12)
    lnstate->board_[kOPlayerId][13] = 1; // Black checker inside home at point 14 (index 13)
    lnstate->dice_ = dice;             // Set dice
    lnstate->scores_ = scores;         // Set scores
    lnstate->cur_player_ = kOPlayerId; // Set current player
    lnstate->is_first_turn_ = false;   // Ensure it's not the first turn
    lnstate->moved_from_head_ = false; // Reset head move flag

    if (kDebugging) {
        std::cout << "DEBUG (Black DirectBearOffTest Post-Direct-Set):\n" << lnstate->ToString() << std::endl;
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
      if (moves.size() == 2) { // Check for the correct sequence {12->off(d1), 13->off(d2)}
        bool move1_ok = (moves[0].pos == 12 && moves[0].die == 1 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 12 && moves[1].die == 1 && moves[1].to_pos == kBearOffPos);
        bool move2_ok = (moves[0].pos == 13 && moves[0].die == 2 && moves[0].to_pos == kBearOffPos) || \
                        (moves[1].pos == 13 && moves[1].die == 2 && moves[1].to_pos == kBearOffPos); // Check for 13->off(d2)
        if (move1_ok && move2_ok) {
          found_direct_bearoff = true;
          break;
        }
      }
    }
    SPIEL_CHECK_TRUE(found_direct_bearoff); 
    if (kDebugging) std::cout << "  Black Direct Bear-Off: Passed\n";
  }
}

// New test for single checker bear-off rules (higher die, only die)
void SingleCheckerBearOffTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // --- Scenario 1: Higher Die Rule (Both playable) ---
  {
    if (kDebugging) std::cout << "  Testing SingleCheckerBearOffTest (Higher Die Rule)..." << std::endl;
    std::vector<int> dice = {1, 6}; // Dice 1 and 6
    
    // White Test (Checker at pos 0)
    std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints, 0));
    boardW[kXPlayerId][0] = 1;
    lnstate->SetState(kXPlayerId, false, dice, {14, 0}, boardW);
    auto legal_actionsW = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsW.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesW = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
    bool found_die6W = false;
    for(const auto& m : movesW) { if(m.pos == 0 && m.die == 6 && m.to_pos == kBearOffPos) found_die6W = true; }
    SPIEL_CHECK_TRUE(found_die6W);
    if (kDebugging) std::cout << "  White Higher Die: Passed\n";

    // Black Test (Checker at pos 12)
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
    boardB[kOPlayerId][12] = 1;
    lnstate->SetState(kOPlayerId, false, dice, {0, 14}, boardB);
    auto legal_actionsB = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsB.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
    bool found_die6B = false;
    for(const auto& m : movesB) { if(m.pos == 12 && m.die == 6 && m.to_pos == kBearOffPos) found_die6B = true; }
    SPIEL_CHECK_TRUE(found_die6B);
    if (kDebugging) std::cout << "  Black Higher Die: Passed\n";
  }

  // --- Scenario 2: Only One Die Playable ---
  {
    if (kDebugging) std::cout << "  Testing SingleCheckerBearOffTest (Only One Die Playable)..." << std::endl;
    std::vector<int> dice = {1, 3}; // Dice 1 and 3
    
    // White Test (Checker at pos 0 - only die 1 or 3 works)
    std::vector<std::vector<int>> boardW(2, std::vector<int>(kNumPoints, 0));
    boardW[kXPlayerId][0] = 1;
    lnstate->SetState(kXPlayerId, false, dice, {14, 0}, boardW);
    auto legal_actionsW = lnstate->LegalActions();
    SPIEL_CHECK_EQ(legal_actionsW.size(), 1); // Expect only one action
    std::vector<CheckerMove> movesW = lnstate->SpielMoveToCheckerMoves(kXPlayerId, legal_actionsW[0]);
    bool found_die3W = false; // Higher die is 3
    for(const auto& m : movesW) { if(m.pos == 0 && m.die == 3 && m.to_pos == kBearOffPos) found_die3W = true; }
    SPIEL_CHECK_TRUE(found_die3W);
    if (kDebugging) std::cout << "  White Only Die (3): Passed\n";

    // Black Test (Checker at pos 12 - only die 1 or 3 works)
    std::vector<std::vector<int>> boardB(2, std::vector<int>(kNumPoints, 0));
    boardB[kOPlayerId][12] = 1;
    lnstate->SetState(kOPlayerId, false, dice, {0, 14}, boardB);
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

     // Black Test (Checker at pos 14 - needs 3 pips. Only die 3 works)
     boardB.assign(2, std::vector<int>(kNumPoints, 0)); // Clear board
     boardB[kOPlayerId][14] = 1;
     lnstate->SetState(kOPlayerId, false, dice, {0, 14}, boardB);
     legal_actionsB = lnstate->LegalActions();
     SPIEL_CHECK_EQ(legal_actionsB.size(), 1); // Expect only one sequence (the two-step one)
     movesB = lnstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actionsB[0]);
     // Correctly check for the TWO-STEP sequence: {14,13,1} {13,-1,3}
     SPIEL_CHECK_EQ(movesB.size(), 2);
     bool step1_ok = (movesB[0].pos == 14 && movesB[0].to_pos == 13 && movesB[0].die == 1) || \
                     (movesB[1].pos == 14 && movesB[1].to_pos == 13 && movesB[1].die == 1);
     bool step2_ok = (movesB[0].pos == 13 && movesB[0].to_pos == kBearOffPos && movesB[0].die == 3) || \
                     (movesB[1].pos == 13 && movesB[1].to_pos == kBearOffPos && movesB[1].die == 3);
     SPIEL_CHECK_TRUE(step1_ok && step2_ok);
     if (kDebugging) std::cout << "  Black Only Die (3 from pos 14): Passed\n";

  }
}

// New test for bearing off the last checker, forcing a pass on the second die
void BearOffLastCheckerTest() {
  if (kDebugging) std::cout << "\n=== Running BearOffLastCheckerTest ===\n";
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  std::vector<int> dice = {4, 5}; // High dice, either can bear off

  // --- White Test (Last checker at pos 1, needs 2 pips) ---
  {
    if (kDebugging) std::cout << "  Testing White Last Checker Bear Off..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kXPlayerId][1] = 1; // White's last checker at pos 1 (needs 2 pips)
    board[kOPlayerId][11] = 15; // Irrelevant black checkers
    std::vector<int> scores = {14, 0}; // White has 14 borne off

    lnstate->SetState(kXPlayerId, false, dice, scores, board);

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

  // --- Black Test (Last checker at pos 13, needs 2 pips) ---
  {
    if (kDebugging) std::cout << "  Testing Black Last Checker Bear Off..." << std::endl;
    std::vector<std::vector<int>> board(2, std::vector<int>(kNumPoints, 0));
    board[kOPlayerId][13] = 1; // Black's last checker at pos 13 (needs 2 pips)
    board[kXPlayerId][23] = 15; // Irrelevant white checkers
    std::vector<int> scores = {0, 14}; // Black has 14 borne off

    lnstate->SetState(kOPlayerId, false, dice, scores, board);

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
   if (kDebugging) std::cout << "✓ BearOffLastCheckerTest passed\n";
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
  DirectBearOffTest();
  SingleCheckerBearOffTest();
  BearOffLastCheckerTest();
  
  std::cout << "✓ All action encoding tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
 