#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

// Common constants for tests
// constexpr int kNumPoints = 24; // Defined in long_narde.h
// constexpr int kPassPos = -1;   // Defined in long_narde.h

// Common utility functions
bool ActionsContains(const std::vector<Action>& legal_actions, Action action);

// Exposed test functions - these are the main entry points for each test category
void TestBasicSetup();     // Basic setup and initialization tests
void TestMovementRules();  // Movement rules tests
void TestBridgeFormation(); // Bridge formation tests
void TestActionEncoding(); // Action encoding tests
void TestEndgame();        // Endgame tests including bearing off
void TestHeadRule();       // Head rule tests
void TestPassMoveBehavior(); // Test pass move behavior

// Original function for backward compatibility
void BasicLongNardeTests();

// TestBasicMovement is implemented in long_narde_test_movement.cc
void TestBasicMovement();

// Helper function to create a Long Narde game state
std::unique_ptr<LongNardeState> CreateStateFromString(const std::string& board_string);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_