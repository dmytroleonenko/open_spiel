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

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

void ActionEncodingTest() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<std::vector<int>> board_real(2,
                                           std::vector<int>(kNumPoints, 0));
  board_real[kXPlayerId][23] = 13;
  board_real[kXPlayerId][19] = 1;
  board_real[kXPlayerId][14] = 1;
  board_real[kOPlayerId][11] = kNumCheckersPerPlayer;

  testing::SetupStateFromRealBoard(lnstate, board_real, kXPlayerId, {5, 3},
                                   /*is_first_turn=*/false,
                                   /*head_moved_count=*/0,
                                   /*phase=*/0);

  std::vector<LongNardeCheckerMove> test_moves = {
      {14, lnstate->GetToPos(kXPlayerId, 14, 5), 5},
      {19, lnstate->GetToPos(kXPlayerId, 19, 3), 3}};
  Action action = lnstate->LongNardeCheckerMovesToSpielMove(test_moves);
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);

  std::vector<LongNardeCheckerMove> decoded =
      lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, action);
  SPIEL_CHECK_EQ(decoded.size(), 2);
  bool found_first = false;
  bool found_second = false;
  for (const auto& move : decoded) {
    if (move.pos == test_moves[0].pos && move.die == test_moves[0].die) {
      found_first = true;
    }
    if (move.pos == test_moves[1].pos && move.die == test_moves[1].die) {
      found_second = true;
    }
  }
  SPIEL_CHECK_TRUE(found_first);
  SPIEL_CHECK_TRUE(found_second);

  std::vector<LongNardeCheckerMove> pass_moves = {
      {kPassPos, kPassPos, 5},
      {kPassPos, kPassPos, 3}};
  Action pass_action = lnstate->LongNardeCheckerMovesToSpielMove(pass_moves);
  SPIEL_CHECK_GE(pass_action, 0);
  SPIEL_CHECK_LT(pass_action, kNumDistinctActions);
  std::vector<LongNardeCheckerMove> decoded_pass =
      lnstate->LongNardeSpielMoveToCheckerMoves(kXPlayerId, pass_action);
  SPIEL_CHECK_EQ(decoded_pass.size(), 2);
  SPIEL_CHECK_EQ(decoded_pass[0].pos, kPassPos);
  SPIEL_CHECK_EQ(decoded_pass[1].pos, kPassPos);
}

}  // namespace

void TestActionEncoding() {
  ActionEncodingTest();
  std::cout << "✓ Action encoding tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
