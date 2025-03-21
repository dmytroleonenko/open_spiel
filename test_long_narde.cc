#include <iostream>
#include "open_spiel/games/long_narde/long_narde.h"

using namespace open_spiel;
using namespace open_spiel::long_narde;

int main() {
  std::cout << "Testing Long Narde implementation\n";
  
  // Load the game
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::cout << "Game loaded successfully\n";
  
  // Create initial state
  std::unique_ptr<State> state = game->NewInitialState();
  std::cout << "Initial state created\n";
  
  // Check initial board setup
  LongNardeState* lnstate = static_cast<LongNardeState*>(state.get());
  
  // Check White's initial position (all 15 on point 24)
  if (lnstate->board(kXPlayerId, kWhiteHeadPos) == kNumCheckersPerPlayer) {
    std::cout << "White's initial position verified\n";
  } else {
    std::cout << "Error: White's initial position incorrect\n";
    return 1;
  }
  
  // Check Black's initial position (all 15 on point 12)
  if (lnstate->board(kOPlayerId, kBlackHeadPos) == kNumCheckersPerPlayer) {
    std::cout << "Black's initial position verified\n";
  } else {
    std::cout << "Error: Black's initial position incorrect\n";
    return 1;
  }
  
  std::cout << "All basic tests passed!\n";
  return 0;
} 