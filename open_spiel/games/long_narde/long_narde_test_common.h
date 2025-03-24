// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {

// Additional constants for testing
constexpr int kBearOffPos = 0;  // Position used for bearing off
constexpr int kPassDie = 0;    // Die value used for pass moves

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

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_TEST_COMMON_H_ 