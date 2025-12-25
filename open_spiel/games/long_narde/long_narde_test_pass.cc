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
#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

void PassWhenNoMovesAvailable() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][23] = kNumCheckersPerPlayer;
  board_real[kOPlayerId][22] = 1;
  board_real[kOPlayerId][21] = 1;

  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {1, 2},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<Action> actions = lnstate->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 1);
  int order = 1;
  Action pass_action = testing::EncodeAction(kActionPassSrc, kActionPassSrc,
                                             order);
  SPIEL_CHECK_EQ(actions[0], pass_action);
}

}  // namespace

void TestPassMoveBehavior() {
  PassWhenNoMovesAvailable();
  std::cout << "✓ Pass move behavior tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
