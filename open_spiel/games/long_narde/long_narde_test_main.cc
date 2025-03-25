#include "open_spiel/games/long_narde/long_narde_test_common.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

int main(int argc, char** argv) {
  // Load game test ensures the game can be properly loaded
  open_spiel::testing::LoadGameTest("long_narde");
  
  // Run all test groups
  open_spiel::long_narde::TestBasicSetup();
  open_spiel::long_narde::TestMovementRules();
  open_spiel::long_narde::TestBridgeFormation();
  open_spiel::long_narde::TestActionEncoding();
  open_spiel::long_narde::TestEndgame();
  open_spiel::long_narde::TestHeadRule();
  open_spiel::long_narde::TestPassMoveBehavior();
  
  // For backward compatibility, the legacy test function that's called from
  // scripts like build_long_narde.sh
  if (argc > 1 && std::string(argv[1]) == "--legacy") {
    open_spiel::long_narde::BasicLongNardeTests();
    // TestBasicMovement is already run as part of TestMovementRules and BasicLongNardeTests
  }
  
  std::cout << "✓ All tests passed\n";
  return 0;
} 