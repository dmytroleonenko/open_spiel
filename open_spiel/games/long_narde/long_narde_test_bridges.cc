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

void TestBridgeFormation() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<LongNardeState*>(state.get());
  
  // Set up a test board where White has 6 points in a row (illegal bridge)
  // but Black has a checker ahead of the bridge (making it legal)
  std::vector<std::vector<int>> test_board = {
    {0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}, // White 1-6
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  std::vector<int> dice = {1, 1};
  
  // Set white to move
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  std::cout << "DEBUG: Testing bridge formation with dice {1,1}" << std::endl;
  std::cout << "DEBUG: Board state:" << std::endl;
  std::cout << "DEBUG: White: ";
  for (int i = 0; i < kNumPoints; ++i) {
    if (lnstate->board(kXPlayerId, i) > 0) {
      std::cout << i << ":" << lnstate->board(kXPlayerId, i) << " ";
    }
  }
  std::cout << std::endl;
  std::cout << "DEBUG: Black: ";
  for (int i = 0; i < kNumPoints; ++i) {
    if (lnstate->board(kOPlayerId, i) > 0) {
      std::cout << i << ":" << lnstate->board(kOPlayerId, i) << " ";
    }
  }
  std::cout << std::endl;
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  std::cout << "DEBUG: Number of legal actions: " << legal_actions.size() << std::endl;
  
  // Check if we can find a move that moves from point 1 to 0, forming a 7-point bridge
  bool found_legal_bridge_move = false;
  for (Action action : legal_actions) {
    std::cout << "DEBUG: Checking action " << action << " with 2 moves" << std::endl;
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      std::cout << "DEBUG: Move pos=" << move.pos << ", to_pos=" << move.to_pos << ", die=" << move.die << std::endl;
      // Look for moves from position 1 (index 0) to position 0 (off-board/bearing off)
      if (move.pos == 1 && move.to_pos == 0) {
        found_legal_bridge_move = true;
        std::cout << "DEBUG: Found legal bridge move!" << std::endl;
        // This is a legal move, so forming a 7-point bridge should be allowed when
        // the opponent has checkers ahead of the bridge
        std::cout << "DEBUG: WouldFormBlockingBridge(0, 6, 5) = " 
                  << lnstate->WouldFormBlockingBridge(0, 6, 5) << std::endl;
        break;
      }
    }
    if (found_legal_bridge_move) break;
  }
  
  // There should be a legal move to form the bridge since opponent has checkers ahead
  SPIEL_CHECK_TRUE(found_legal_bridge_move);
  
  // Now test an illegal bridge - opponent has no checkers ahead
  test_board = {
    {0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10}, // White 1-5
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  
  // Set white to move, dice 1 (to move to position 6)
  dice = {1, 6};
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Get legal actions
  legal_actions = lnstate->LegalActions();
  
  // Check for moves from point 7 to point 6 (forming a blocking bridge)
  bool found_illegal_bridge_move = false;
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    for (const CheckerMove& move : moves) {
      // Look for moves from position 6 to position 5 (forming a 6-point bridge)
      if (move.pos == 6 && move.to_pos == 5) {
        found_illegal_bridge_move = true;
        break;
      }
    }
    if (found_illegal_bridge_move) break;
  }
  
  // There should be no legal moves to form a blocking bridge since Black has no checkers ahead
  SPIEL_CHECK_FALSE(found_illegal_bridge_move);
  
  // Test bridge detection functions directly
  // Set up a state where White has checkers on points 1-5 and we want to move to point 6
  test_board = {
    {0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10}, // White 1-5
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black
  };
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Directly test the bridge detection function
  bool would_form_bridge = lnstate->WouldFormBlockingBridge(0, 5, 5);
  SPIEL_CHECK_TRUE(would_form_bridge);
  
  // Test with opponent checker ahead
  test_board = {
    {0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10}, // White 1-5
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black with checker at 7
  };
  lnstate->SetState(kXPlayerId, false, dice, {0, 0}, test_board);
  
  // Should not be a blocking bridge since opponent has a checker ahead
  would_form_bridge = lnstate->WouldFormBlockingBridge(0, 5, 5);
  SPIEL_CHECK_FALSE(would_form_bridge);
}

}  // namespace long_narde
}  // namespace open_spiel 