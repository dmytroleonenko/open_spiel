#include <iostream>
#include "open_spiel/spiel.h"
#include "open_spiel/games/long_narde/long_narde.h"

using namespace open_spiel;

int main() {
  std::cout << "Testing Long Narde validation fix" << std::endl;
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  
  auto lnstate = dynamic_cast<long_narde::LongNardeState*>(state.get());
  if (!lnstate) {
    std::cerr << "Failed to cast state to LongNardeState" << std::endl;
    return 1;
  }

  // Test IsValidCheckerMove logic
  std::cout << "Testing validation functions..." << std::endl;
  
  // We'll create a simple test where we try to move to a position occupied by opponent
  
  // Simulate a dice roll for player 0
  if (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    // Find the action for rolling double 1s
    for (const auto& outcome : outcomes) {
      if (outcome.first == 1) {  // Action 1 is double 1s
        std::cout << "Rolling dice: double 1s" << std::endl;
        state->ApplyAction(outcome.first);
        break;
      }
    }
  }
  
  // Get the board state
  std::cout << "Board state:\n" << state->ToString() << std::endl;
  
  // Test IsValidCheckerMove with various scenarios
  int player = 0;  // Player 0
  int opponent = 1;  // Player 1
  
  // First, find a checker for player 0 that we can test with
  int from_pos = -1;
  for (int pos = 0; pos < long_narde::kNumPoints; pos++) {
    if (lnstate->board(player, pos) > 0) {
      from_pos = pos;
      break;
    }
  }
  
  if (from_pos == -1) {
    std::cerr << "Could not find a checker for player 0" << std::endl;
    return 1;
  }
  
  int to_pos = lnstate->GetToPos(player, from_pos, 1);  // Move 1 space
  std::cout << "Testing move from " << from_pos << " to " << to_pos << std::endl;
  
  // Check if the move is valid
  bool is_valid = lnstate->IsValidCheckerMove(player, from_pos, to_pos, 1, true);
  std::cout << "Move valid (without opponent): " << (is_valid ? "YES" : "NO") << std::endl;
  
  // Now try the buggy scenario:
  // Create a test board state with opposing checkers at destination
  
  // First print all legal moves
  std::cout << "Legal moves for player " << player << ":" << std::endl;
  auto legal_moves = lnstate->LegalCheckerMoves(player);
  for (const auto& move : legal_moves) {
    std::cout << "  From " << move.pos << " to " << move.to_pos << " using die " << move.die << std::endl;
  }
  
  std::cout << "Test completed!" << std::endl;
  return 0;
} 