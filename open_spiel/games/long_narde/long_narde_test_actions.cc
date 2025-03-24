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

void ActionEncodingTestInternal() {
  std::cout << "[ActionEncodingTest] Starting test..." << std::endl;
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  std::cout << "[ActionEncodingTest] kNumDistinctActions: " << kNumDistinctActions << std::endl;
  
  // Force the state into player mode - apply a dice roll
  lnstate->ApplyAction(0);  // Roll 1,2
  
  // Set it to first turn mode for testing encoding/decoding
  lnstate->MutableIsFirstTurn() = true;
  std::cout << "[ActionEncodingTest] Forcing first turn mode for testing" << std::endl;
  
  // Test 1: Regular move encoding (high roll first)
  std::vector<CheckerMove> moves = {
    {kWhiteHeadPos, kWhiteHeadPos - 6, 6},  // Move from 24 to 18
    {kPassPos, kPassPos, kPassDie}         // Pass second die
  };
  
  Action action = lnstate->CheckerMovesToSpielMove(moves);
  
  std::cout << "[ActionEncodingTest] Test 1: Regular move encoding (high roll first)" << std::endl;
  std::cout << "[ActionEncodingTest] Encoded action: " << action << std::endl;
  
  std::vector<CheckerMove> decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
  
  std::cout << "[ActionEncodingTest] Decoded moves:" << std::endl;
  std::cout << "[ActionEncodingTest]   pos=" << decoded_moves[0].pos 
            << ", die=" << decoded_moves[0].die
            << " (to_pos=" << decoded_moves[0].to_pos << ")" << std::endl;
  std::cout << "[ActionEncodingTest]   pos=" << decoded_moves[1].pos 
            << ", die=" << decoded_moves[1].die
            << " (to_pos=" << decoded_moves[1].to_pos << ")" << std::endl;
  
  bool first_move_found = (decoded_moves[0].pos == kWhiteHeadPos && 
                          decoded_moves[0].to_pos == kWhiteHeadPos - 6);
  bool second_move_found = (decoded_moves[1].pos == kPassPos &&
                           decoded_moves[1].to_pos == kPassPos);
  
  std::cout << "[ActionEncodingTest] First move found: " << (first_move_found ? "YES" : "NO") << std::endl;
  std::cout << "[ActionEncodingTest] Second move found: " << (second_move_found ? "YES" : "NO") << std::endl;
  
  SPIEL_CHECK_TRUE(first_move_found);
  SPIEL_CHECK_TRUE(second_move_found);
  
  // Test 2: Test pass moves
  std::vector<CheckerMove> pass_moves = {
    {kPassPos, kPassPos, kPassDie},
    {kPassPos, kPassPos, kPassDie}
  };
  
  action = lnstate->CheckerMovesToSpielMove(pass_moves);
  
  std::cout << "[ActionEncodingTest] Test 2: Testing pass moves" << std::endl;
  std::cout << "[ActionEncodingTest] Encoded action: " << action << std::endl;
  
  decoded_moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
  
  std::cout << "[ActionEncodingTest] Decoded moves:" << std::endl;
  std::cout << "[ActionEncodingTest]   pos=" << decoded_moves[0].pos 
            << ", die=" << decoded_moves[0].die 
            << " (to_pos=" << decoded_moves[0].to_pos << ")" << std::endl;
  std::cout << "[ActionEncodingTest]   pos=" << decoded_moves[1].pos 
            << ", die=" << decoded_moves[1].die
            << " (to_pos=" << decoded_moves[1].to_pos << ")" << std::endl;
  
  bool first_pass_found = (decoded_moves[0].pos == kPassPos && 
                          decoded_moves[0].to_pos == kPassPos);
  bool second_pass_found = (decoded_moves[1].pos == kPassPos &&
                           decoded_moves[1].to_pos == kPassPos);
  
  std::cout << "[ActionEncodingTest] First pass found: " << (first_pass_found ? "YES" : "NO") << std::endl;
  std::cout << "[ActionEncodingTest] Second pass found: " << (second_pass_found ? "YES" : "NO") << std::endl;
  
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
  
  // White should have only one legal move: the checker at point 1 can move
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

}  // namespace

// Exposed test function
void TestActionEncoding() {
  std::cout << "\n=== Testing Action Encoding ===" << std::endl;
  
  std::cout << "\n=== Running ActionEncodingTest with diagnostics ===\n";
  ActionEncodingTestInternal();
  std::cout << "✓ ActionEncodingTest passed\n";
  
  std::cout << "\n=== Running SingleLegalMoveTest ===\n";
  SingleLegalMoveTest();
  std::cout << "✓ SingleLegalMoveTest passed\n";
  
  std::cout << "\n=== Running ConsecutiveMovesTest ===\n";
  ConsecutiveMovesTest();
  std::cout << "✓ ConsecutiveMovesTest passed\n";
  
  std::cout << "\n=== Running UndoRedoTest ===\n";
  UndoRedoTest();
  std::cout << "✓ UndoRedoTest passed\n";
  
  std::cout << "✓ All action encoding tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
 