#include "open_spiel/games/long_narde/long_narde.h"

#include <algorithm>
#include <iostream>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

// Check that no checkers land on the opponent's checkers
void CheckNoHits(const State& state) {
  const LongNardeState& lnstate = static_cast<const LongNardeState&>(state);
  for (int pos = 0; pos < kNumPoints; ++pos) {
    // Check that no point has both black and white checkers
    if (lnstate.board(kXPlayerId, pos) > 0 && 
        lnstate.board(kOPlayerId, pos) > 0) {
      std::string board_str = lnstate.ToString();
      SpielFatalError(absl::StrCat(
          "Checkers at same point! pos: ", pos, ", board:\n", board_str));
    }
  }
}

void RunRandomSimTest() {
  std::cout << "=== Running RandomSimTest with verbose output ===" << std::endl;
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  
  // Set a very low number (1-2) for initial debugging
  const int num_sims = 1;
  const bool verbose = true;  // Enable verbose output
  const bool verify = true;   // Verify the move legality
  
  std::cout << "Starting RandomSimTest with " << num_sims << " simulations..." << std::endl;
  testing::RandomSimTest(*game, num_sims, verbose, verify, &CheckNoHits);
  std::cout << "RandomSimTest completed successfully!" << std::endl;
}

}  // namespace

void RunRandomSimTests() {
  std::cout << "Starting random sim tests..." << std::endl;
  RunRandomSimTest();
  std::cout << "All random sim tests completed!" << std::endl;
}

}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::long_narde::RunRandomSimTests();
  return 0;
} 