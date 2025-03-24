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

#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <algorithm>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {

// Use anonymous namespace to avoid symbol conflicts
namespace {

// Internal version of TestBasicMovement
void TestBasicMovementInternal() {
  std::cout << "\n=== TestBasicMovement (forwarding to actual implementation) ===" << std::endl;
  TestMovementRules();
}

} // namespace

void BasicLongNardeTests() {
  std::cout << "\n=== Running legacy BasicLongNardeTests ===" << std::endl;

  // Load the game
  std::shared_ptr<const Game> game = LoadGame("long_narde");

  // Run the basic tests from OpenSpiel
  testing::RandomSimTest(*game, 10);
  testing::RandomSimTestWithUndo(*game, 10);
  
  // TestClone is not available, skip it
  
  // Verify that the underlying game has the expected properties
  SPIEL_CHECK_EQ(game->GetType().chance_mode, GameType::ChanceMode::kExplicitStochastic);
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game->GetType().information, GameType::Information::kPerfectInformation);
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(game->GetType().reward_model, GameType::RewardModel::kTerminal);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 21);

  std::cout << "✓ All legacy basic tests passed!" << std::endl;
  
  // Run the movement tests as well, since they're part of the legacy tests
  TestBasicMovementInternal();
}

}  // namespace long_narde
}  // namespace open_spiel 