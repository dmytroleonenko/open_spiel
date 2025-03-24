#include <iostream>
#include "open_spiel/spiel.h"
#include "open_spiel/games/long_narde/long_narde.h"

using namespace open_spiel;

int main() {
  std::cout << "Loading Long Narde game..." << std::endl;
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
  std::cout << "Creating initial state..." << std::endl;
  std::unique_ptr<State> state = game->NewInitialState();
  
  std::cout << "Initial state:\n" << state->ToString() << std::endl;
  
  // Attempt to create a problematic board state manually
  auto lnstate = dynamic_cast<long_narde::LongNardeState*>(state.get());
  if (!lnstate) {
    std::cerr << "Failed to cast state to LongNardeState" << std::endl;
    return 1;
  }
  
  std::cout << "Current player: " << lnstate->CurrentPlayer() << std::endl;
  
  // Test validation functions
  int player = long_narde::kXPlayerId; // White player
  int from_pos = 20;
  int to_pos = 19;
  int die_value = 1;
  
  // Make sure player has a checker at from_pos
  std::cout << "Testing IsValidCheckerMove(" << player << ", " << from_pos << ", " << to_pos << ", " << die_value << ", true)" << std::endl;
  
  // Direct access to board_ is not allowed, use reflection via spiel state
  std::string state_str = state->ToString();
  std::cout << "Current board:\n" << state_str << std::endl;
  
  // Use existing checker positions instead
  // Find a position that has checkers and test with that
  for (int pos = 0; pos < long_narde::kNumPoints; pos++) {
    if (lnstate->board(player, pos) > 0) {
      from_pos = pos;
      to_pos = lnstate->GetToPos(player, from_pos, die_value);
      std::cout << "Found checker at position " << from_pos << ", target position: " << to_pos << std::endl;
      break;
    }
  }
  
  bool is_valid = lnstate->IsValidCheckerMove(player, from_pos, to_pos, die_value, true);
  std::cout << "Is valid: " << (is_valid ? "YES" : "NO") << std::endl;
  
  // Create problematic scenario by applying a move to setup opponent checker
  // Apply a chance move to roll the dice
  if (state->IsChanceNode()) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    Action action = outcomes[0].first;  // Just pick the first one
    std::cout << "Applying chance action: " << action << std::endl;
    state->ApplyAction(action);
  }
  
  // Get legal actions and pick one to apply
  std::vector<Action> legal_actions = state->LegalActions();
  if (!legal_actions.empty()) {
    std::cout << "Applying action: " << legal_actions[0] << std::endl;
    state->ApplyAction(legal_actions[0]);
  }
  
  // If we're at player 1's turn, place a checker
  if (state->CurrentPlayer() == long_narde::kOPlayerId) {
    // We're at player 1's turn, ready to test
    if (lnstate->board(long_narde::kXPlayerId, to_pos) == 0) {
      // Create a checker move for testing ApplyCheckerMove directly
      std::cout << "Testing ApplyCheckerMove to a free position..." << std::endl;
      long_narde::CheckerMove move(from_pos, to_pos, die_value);
      try {
        lnstate->ApplyCheckerMove(player, move);
        std::cout << "Move applied successfully!" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "Caught unexpected exception: " << e.what() << std::endl;
      }
    } else {
      std::cout << "Position " << to_pos << " already has a checker, can't test." << std::endl;
    }
  }
  
  std::cout << "Test completed!" << std::endl;
  return 0;
} 