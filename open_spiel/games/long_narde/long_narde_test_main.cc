// Copyright 2025 DeepMind Technologies Limited
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

#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("long_narde");

  std::cout << "=== Running Movement Rules Tests ===\n";
  open_spiel::long_narde::TestMovementRules();
  std::cout << "=== Testing Action Encoding ===\n";
  open_spiel::long_narde::TestActionEncoding();
  std::cout << "=== Testing Bridge Rules ===\n";
  open_spiel::long_narde::TestBridgeFormation();
  std::cout << "=== Testing Endgame Rules ===\n";
  open_spiel::long_narde::TestEndgame();
  std::cout << "=== Testing Pass Move Behavior ===\n";
  open_spiel::long_narde::TestPassMoveBehavior();
  std::cout << "=== Testing Search Invariants ===\n";
  open_spiel::long_narde::TestSearchInvariants();

  std::cout << "✓ All tests passed\n";
  return 0;
}
