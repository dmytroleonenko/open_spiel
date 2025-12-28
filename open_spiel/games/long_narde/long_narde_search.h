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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SEARCH_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SEARCH_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nnue_cache.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"

namespace open_spiel {
namespace long_narde {

struct SearchConfig {
  int max_depth = 3;
  int top_k = 16;
  int min_k = 4;
  double delta = 0.25;
  bool enable_root_iterative = true;
  bool enable_panic_widen = true;
  double panic_margin = 0.75;
  int panic_min_k = 8;
  int panic_top_k = 32;
  double panic_delta = 0.5;
  bool enable_pruning = true;
  bool use_tt = true;
  int max_tt_entries = 200000;
  bool use_undo = true;
  bool use_nnue_cache = true;
  int chance_samples = 0;
  int chance_sample_depth = 1;
  uint64_t chance_seed = 0;
};

struct SearchResult {
  double value = 0.0;
  Action best_action = kInvalidAction;
};

class ExpectiminimaxSearch {
 public:
  ExpectiminimaxSearch(const nnue::NnueEvaluator* evaluator,
                       const SearchConfig& config);
  ~ExpectiminimaxSearch();

  SearchResult Search(LongNardeState* state);
  std::vector<std::pair<Action, double>> EvaluateDecisionActions(
      LongNardeState* state);
  void ClearCache();
  const std::vector<double>& last_root_times_ms() const {
    return last_root_times_ms_;
  }

 private:
  struct TTEntry {
    int depth = -1;
    double value = 0.0;
    Action best_action = kInvalidAction;
  };

  double SearchState(LongNardeState* state, int depth,
                     Player maximizing_player, Action* best_action);
  double EvaluatePreRoll(const LongNardeState& state,
                         Player maximizing_player) const;
  std::vector<Action> OrderedActions(
      LongNardeState* state, int depth, Player maximizing_player,
      std::vector<std::pair<Action, double>>* scores_out);
  std::vector<std::pair<Action, double>> ScoreActions(
      LongNardeState* state, Player maximizing_player);
  std::vector<Action> PruneActions(
      const std::vector<std::pair<Action, double>>& scores, int min_k,
      int top_k, double delta) const;
  double EvaluateActionList(LongNardeState* state,
                            const std::vector<Action>& actions, int depth,
                            Player maximizing_player, Action* best_action);
  static uint64_t HashMix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
  uint64_t HashState(const LongNardeState& state) const {
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

  const nnue::NnueEvaluator* evaluator_;
  SearchConfig config_;
  std::unordered_map<uint64_t, TTEntry> table_;
  std::unique_ptr<NnueCacheStack> cache_stack_;
  std::vector<double> last_root_times_ms_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SEARCH_H_
