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

#include "open_spiel/games/long_narde/long_narde.h"

#include <iostream>
#include "open_spiel/spiel.h"

int main(int argc, char** argv) {
  using namespace open_spiel;
  std::cout << "\n=== Running Bearing Off From Position 1 Test ===\n";
  
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = static_cast<long_narde::LongNardeState*>(state.get());
  
  // Create our test board with one checker in position 1
  std::vector<std::vector<int>> test_board = {
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White: 1 at pos 1
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  // Black distribution
  };
  
  // Set up the test state with white to move, dice 1 and 3
  std::vector<int> dice = {1, 3};
  std::vector<int> scores = {14, 0}; // White has 14 checkers borne off
  lnstate->SetState(long_narde::kXPlayerId, false, dice, scores, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Find bearing off moves
  bool can_bear_off_with_1 = false;
  bool can_bear_off_with_3 = false;
  
  for (Action action : legal_actions) {
    std::vector<long_narde::CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(long_narde::kXPlayerId, action);
    
    for (const auto& move : moves) {
      if (move.pos == 1 && move.to_pos < 0) {
        if (move.die == 1) {
          can_bear_off_with_1 = true;
        } else if (move.die == 3) {
          can_bear_off_with_3 = true;
        }
      }
    }
  }
  
  // Should be able to bear off with 1 (exact move)
  bool test_passed = can_bear_off_with_1 && can_bear_off_with_3;
  
  if (test_passed) {
    std::cout << "✓ Bearing off test PASSED - can bear off with any die value when all checkers are in home\n";
    return 0;
  } else {
    std::cout << "❌ Bearing off test FAILED - should be able to bear off with any die value when all checkers are in home\n";
    return 1;
  }
}
