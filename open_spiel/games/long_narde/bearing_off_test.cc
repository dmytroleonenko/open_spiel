#include "open_spiel/games/long_narde/long_narde.h"

#include <iostream>
#include "open_spiel/spiel.h"
#include "open_spiel/games/long_narde/long_narde_test_common.h"

//StartFunction: main
//StartTest: test-bearingoff-main-1
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
  // Corrected definition: Use 4 elements, pad with 0s, higher die first
  std::vector<int> dice = {3, 1, 0, 0};
  // Corrected scores: W=1 on board -> score=14, B=15 on board -> score=0
  std::vector<int> scores = {14, 0};
  SetupBoardState(lnstate, long_narde::kXPlayerId, test_board, scores);
  // Corrected call: Pass 4-element vector, remove boolean flag
  SetupDice(lnstate, dice);

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
//EndTest: test-bearingoff-main-1
//EndFunction: main
}
