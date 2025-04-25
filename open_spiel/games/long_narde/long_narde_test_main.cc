#include "open_spiel/games/long_narde/long_narde_test_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

int main(int argc, char **argv)
{
  // Ensure the game can be loaded before running test groups.
  open_spiel::testing::LoadGameTest("long_narde");

  // Run all major test groups for Long Narde.
  open_spiel::long_narde::TestMovementRules();
  // open_spiel::long_narde::TestBasicSetup();
  // open_spiel::long_narde::TestSimpleNonDoubleMove();
  // open_spiel::long_narde::BasicLongNardeTests(); // Legacy tests
  open_spiel::long_narde::TestActionEncoding();
  open_spiel::long_narde::TestEndgame();
  open_spiel::long_narde::TestPassMoveBehavior();

  std::cout << "\u2713 All tests passed\n";
  return 0;
}