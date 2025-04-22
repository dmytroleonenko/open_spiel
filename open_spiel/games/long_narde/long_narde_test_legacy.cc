#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {

// Use anonymous namespace to avoid symbol conflicts
namespace {

//StartFunction: TestBasicMovementInternal
//StartTest: test-legacy-movement-internal
void TestBasicMovementInternal() {
  std::cout << "\n=== TestBasicMovement (forwarding to actual implementation) ===" << std::endl;
  TestMovementRules();
}
//EndTest: test-legacy-movement-internal
//EndFunction: TestBasicMovementInternal

} // namespace

//StartFunction: BasicLongNardeTests
//StartTest: test-legacy-basic-1
void BasicLongNardeTests() {
  std::cout << "\n=== Running legacy BasicLongNardeTests ===" << std::endl;

  std::shared_ptr<const Game> game = LoadGame("long_narde");

  // Run OpenSpiel's basic random simulation tests and undo tests.
  testing::RandomSimTest(*game, 10);
  testing::RandomSimTestWithUndo(*game, 10);

  // Verify core game type properties for legacy compatibility.
  SPIEL_CHECK_EQ(game->GetType().chance_mode, GameType::ChanceMode::kExplicitStochastic);
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game->GetType().information, GameType::Information::kPerfectInformation);
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(game->GetType().reward_model, GameType::RewardModel::kTerminal);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 21);

  std::cout << "\u2713 All legacy basic tests passed!" << std::endl;

  // Also run the legacy movement rules test for completeness.
  TestBasicMovementInternal();
}
//EndTest: test-legacy-basic-1
//EndFunction: BasicLongNardeTests

}  // namespace long_narde
}  // namespace open_spiel