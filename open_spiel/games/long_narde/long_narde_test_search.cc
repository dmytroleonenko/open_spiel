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
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace long_narde {
namespace {

std::vector<int> RandomComposition(int total, int parts,
                                   std::mt19937* rng) {
  SPIEL_CHECK_GT(parts, 0);
  std::vector<int> out(parts, 1);
  int remaining = total - parts;
  for (int i = 0; i < remaining; ++i) {
    int idx = static_cast<int>((*rng)() % parts);
    out[idx] += 1;
  }
  return out;
}

std::array<std::array<int, kNumPoints>, 2> RandomCanonicalBoard(
    std::mt19937* rng) {
  std::array<std::array<int, kNumPoints>, 2> board{};
  for (auto& row : board) {
    row.fill(0);
  }

  std::array<int, kNumPoints> points{};
  std::iota(points.begin(), points.end(), 0);
  std::shuffle(points.begin(), points.end(), *rng);

  int count0 = 6 + static_cast<int>((*rng)() % 7);
  count0 = std::min(count0, kNumPoints - 1);
  std::vector<int> counts0 = RandomComposition(kNumCheckersPerPlayer, count0,
                                               rng);
  for (int i = 0; i < count0; ++i) {
    board[0][points[i]] = counts0[i];
  }

  int remaining = kNumPoints - count0;
  int count1 = 6 + static_cast<int>((*rng)() % 7);
  count1 = std::min(count1, remaining);
  count1 = std::max(count1, 1);
  std::vector<int> counts1 = RandomComposition(kNumCheckersPerPlayer, count1,
                                               rng);
  for (int i = 0; i < count1; ++i) {
    board[1][points[count0 + i]] = counts1[i];
  }

  return board;
}

void SetupDecisionState(LongNardeState* state,
                        const std::array<std::array<int, kNumPoints>, 2>& board,
                        const std::array<int, 2>& dice) {
  state->SetStateForTesting(board, kXPlayerId, dice, /*awaiting_roll=*/false,
                            /*initial_roll=*/false, /*is_first_turn=*/false,
                            /*head_moved_count=*/0, /*phase=*/0);
}

uint64_t HashMix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

uint64_t HashState(const LongNardeState& state) {
  uint64_t h = 0xcbf29ce484222325ULL;
  const auto& board = state.board();
  for (int side = 0; side < kNumPlayers; ++side) {
    for (int point = 0; point < kNumPoints; ++point) {
      h = HashMix(h, static_cast<uint64_t>(board[side][point]));
    }
  }
  h = HashMix(h, static_cast<uint64_t>(state.current_player_id()));
  h = HashMix(h, static_cast<uint64_t>(state.head_moved_count()));
  h = HashMix(h, static_cast<uint64_t>(state.phase()));
  h = HashMix(h, static_cast<uint64_t>(state.is_first_turn()));
  h = HashMix(h, static_cast<uint64_t>(state.initial_roll()));
  h = HashMix(h, static_cast<uint64_t>(state.is_doubles()));
  h = HashMix(h, static_cast<uint64_t>(state.awaiting_roll()));
  if (!state.awaiting_roll()) {
    int d0 = state.dice(0);
    int d1 = state.dice(1);
    if (d0 > d1) {
      std::swap(d0, d1);
    }
    h = HashMix(h, static_cast<uint64_t>(d0));
    h = HashMix(h, static_cast<uint64_t>(d1));
  }
  return h;
}

std::set<uint64_t> NextPreRollKeys(LongNardeState* state) {
  std::set<uint64_t> keys;
  std::vector<Action> actions = state->LegalActions();
  for (Action action : actions) {
    std::unique_ptr<State> child = state->Child(action);
    auto* lnchild = static_cast<LongNardeState*>(child.get());
    SPIEL_CHECK_TRUE(lnchild->awaiting_roll());
    keys.insert(HashState(*lnchild));
  }
  return keys;
}

void DiceSymmetryLegalActions() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::mt19937 rng(19);
  for (int i = 0; i < 20; ++i) {
    auto board = RandomCanonicalBoard(&rng);
    for (int j = 0; j < 8; ++j) {
      int a = static_cast<int>(rng() % 6) + 1;
      int b = static_cast<int>(rng() % 6) + 1;
      if (a == b) {
        continue;
      }
      std::unique_ptr<State> state_ab = game->NewInitialState();
      std::unique_ptr<State> state_ba = game->NewInitialState();
      auto* ln_ab = static_cast<LongNardeState*>(state_ab.get());
      auto* ln_ba = static_cast<LongNardeState*>(state_ba.get());
      SetupDecisionState(ln_ab, board, {a, b});
      SetupDecisionState(ln_ba, board, {b, a});
      std::set<uint64_t> keys_ab = NextPreRollKeys(ln_ab);
      std::set<uint64_t> keys_ba = NextPreRollKeys(ln_ba);
      SPIEL_CHECK_EQ(keys_ab.size(), keys_ba.size());
      SPIEL_CHECK_TRUE(keys_ab == keys_ba);
    }
  }
}

void SearchDeterminism() {
  std::shared_ptr<const Game> game = LoadGame("long_narde");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());
  std::mt19937 rng(7);
  auto board = RandomCanonicalBoard(&rng);
  SetupDecisionState(lnstate, board, {2, 5});

  nnue::NnueModel model;
  nnue::NnueEvaluator evaluator(&model);
  SearchConfig config;
  config.max_depth = 2;
  config.enable_pruning = true;
  config.use_tt = true;
  config.use_undo = true;

  ExpectiminimaxSearch search(&evaluator, config);
  SearchResult first = search.Search(lnstate);
  SearchResult second = search.Search(lnstate);

  SPIEL_CHECK_EQ(first.best_action, second.best_action);
  SPIEL_CHECK_EQ(first.value, second.value);
}

}  // namespace

void TestSearchInvariants() {
  DiceSymmetryLegalActions();
  SearchDeterminism();
  std::cout << "✓ Search invariant tests passed\n";
}

}  // namespace long_narde
}  // namespace open_spiel
