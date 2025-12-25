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

#include "open_spiel/games/long_narde/long_narde_internal.h"
#include "open_spiel/games/long_narde/long_narde_test_common.h"

#include <iostream>

#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace long_narde {
namespace {

void BridgeLegalWhenOpponentEmpty() {
  internal::Board board{};
  for (int i = 0; i < 6; ++i) {
    board[0][i] = 1;
  }
  auto opp_ctx = internal::OppNoAheadBits(board[1]);
  uint32_t occ_rot = internal::Rot12Bits(internal::OccBits(board[0]));
  SPIEL_CHECK_TRUE(
      internal::BridgeLegal(occ_rot, opp_ctx.first, opp_ctx.second));
}

void BridgeIllegalWhenOpponentTrapped() {
  internal::Board board{};
  for (int i = 0; i < 6; ++i) {
    board[0][i] = 1;
  }
  board[1][12] = 1;
  auto opp_ctx = internal::OppNoAheadBits(board[1]);
  uint32_t occ_rot = internal::Rot12Bits(internal::OccBits(board[0]));
  SPIEL_CHECK_FALSE(
      internal::BridgeLegal(occ_rot, opp_ctx.first, opp_ctx.second));
}

void BridgeLegalWhenOpponentAhead() {
  internal::Board board{};
  for (int i = 0; i < 6; ++i) {
    board[0][i] = 1;
  }
  board[1][6] = 1;
  auto opp_ctx = internal::OppNoAheadBits(board[1]);
  uint32_t occ_rot = internal::Rot12Bits(internal::OccBits(board[0]));
  SPIEL_CHECK_TRUE(
      internal::BridgeLegal(occ_rot, opp_ctx.first, opp_ctx.second));
}

}  // namespace

void TestBridgeFormation() {
  BridgeLegalWhenOpponentEmpty();
  BridgeIllegalWhenOpponentTrapped();
  BridgeLegalWhenOpponentAhead();
  std::cout << "✓ Bridge rule tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
