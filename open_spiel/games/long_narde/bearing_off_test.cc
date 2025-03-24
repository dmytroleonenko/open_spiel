// Test the bearing off bug in Long Narde
#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include <iostream>

using namespace open_spiel;
using namespace open_spiel::long_narde;

int main(int argc, char** argv) {
  std::cout << "Running Bearing Off From Position 1 Test\n";
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto lnstate = dynamic_cast<LongNardeState*>(state.get());
  
  // Create our test board from scratch
  std::vector<std::vector<int>> test_board = {
    {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // White: 7 at pos 1
    {0, 0, 0, 0, 3, 1, 5, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0}  // Black distribution
  };
  
  // Set up the test state with white to move, dice 1 and 3
  std::vector<int> dice = {1, 3};
  std::vector<int> scores = {8, 0}; // White has 8 checkers borne off
  lnstate->SetState(kXPlayerId, false, dice, scores, test_board);
  
  // Get legal actions
  std::vector<Action> legal_actions = lnstate->LegalActions();
  
  // Verify that bearing off actions are available
  bool can_bear_off_with_1 = false;
  bool can_bear_off_with_3 = false;
  bool has_pass = false;
  
  std::cout << "Board state for bearing off test:\n" << lnstate->ToString() << std::endl;
  std::cout << "Legal actions count: " << legal_actions.size() << std::endl;
  
  for (Action action : legal_actions) {
    std::vector<CheckerMove> moves = lnstate->SpielMoveToCheckerMoves(kXPlayerId, action);
    
    for (const CheckerMove& move : moves) {
      // For White, to_pos < 0 means bearing off
      if (move.pos == 1 && move.to_pos < 0) {
        if (move.die == 1) {
          can_bear_off_with_1 = true;
          std::cout << "Found action to bear off with die value 1\n";
        } else if (move.die == 3) {
          can_bear_off_with_3 = true;
          std::cout << "Found action to bear off with die value 3\n";
        }
      }
    }
    
    // Check for pass action
    std::vector<CheckerMove> pass_moves = {kPassMove, kPassMove};
    Action pass_action = lnstate->CheckerMovesToSpielMove(pass_moves);
    if (action == pass_action) {
      has_pass = true;
    }
  }
  
  // Display the results
  std::cout << "Can bear off with 1: " << (can_bear_off_with_1 ? "YES" : "NO") << std::endl;
  std::cout << "Can bear off with 3: " << (can_bear_off_with_3 ? "YES" : "NO") << std::endl;
  std::cout << "Has pass action: " << (has_pass ? "YES" : "NO") << std::endl;
  
  // We expect to be able to bear off with both dice
  bool test_passed = can_bear_off_with_1 && can_bear_off_with_3 && !has_pass;
  
  if (test_passed) {
    std::cout << "✓ Bearing off test PASSED - can bear off with any die value when all checkers are in home\n";
    return 0;
  } else {
    std::cout << "❌ Bearing off test FAILED - should be able to bear off with any die value when all checkers are in home\n";
    return 1;
  }
}
