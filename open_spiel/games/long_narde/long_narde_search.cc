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

#include "open_spiel/games/long_narde/long_narde_search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "open_spiel/spiel_utils.h"
namespace open_spiel {
namespace long_narde {
ExpectiminimaxSearch::ExpectiminimaxSearch(
    const nnue::NnueEvaluator* evaluator, const SearchConfig& config) {
  evaluator_ = evaluator;
  config_ = config;
  cache_stack_ = std::make_unique<NnueCacheStack>();
  if (config_.max_tt_entries > 0) {
    table_.reserve(config_.max_tt_entries);
  }
}

ExpectiminimaxSearch::~ExpectiminimaxSearch() = default;
void ExpectiminimaxSearch::ClearCache() {
  table_.clear();
  if (config_.max_tt_entries > 0) {
    table_.reserve(config_.max_tt_entries);
  } else {
    table_.rehash(0);
  }
  if (cache_stack_ != nullptr) {
    cache_stack_->Clear();
  }
}
SearchResult ExpectiminimaxSearch::Search(LongNardeState* state) {
  SearchResult result;
  if (state == nullptr) {
    return result;
  }
  last_root_times_ms_.clear();
  if (cache_stack_ != nullptr) {
    const nnue::NnueNetwork* network =
        (config_.use_nnue_cache && evaluator_ != nullptr)
            ? evaluator_->network()
            : nullptr;
    cache_stack_->Reset(*state, network);
  }
  Player maximizing_player = state->current_player_id();
  if (config_.enable_root_iterative && !state->IsChanceNode() &&
      !state->IsTerminal() && config_.max_depth > 0) {
    std::vector<std::pair<Action, double>> root_scores =
        ScoreActions(state, maximizing_player);
    if (root_scores.empty()) {
      result.value = SearchState(state, config_.max_depth, maximizing_player,
                                 &result.best_action);
    } else {
      int max_depth = config_.max_depth;
      int max_top_k = std::max(config_.top_k, config_.panic_top_k);
      int max_min_k = std::max(config_.min_k, config_.panic_min_k);
      double max_delta = std::max(config_.delta, config_.panic_delta);
      int denom = std::max(1, max_depth - 1);
      auto start_time = std::chrono::steady_clock::now();
      for (int depth = 1; depth <= max_depth; ++depth) {
        int step = depth - 1;
        int top_k = config_.top_k +
                    (max_top_k - config_.top_k) * step / denom;
        int min_k = config_.min_k +
                    (max_min_k - config_.min_k) * step / denom;
        double delta =
            config_.delta + (max_delta - config_.delta) * step / denom;
        top_k = std::max(top_k, min_k);
        std::vector<Action> actions =
            PruneActions(root_scores, min_k, top_k, delta);
        result.value = EvaluateActionList(
            state, actions, depth, maximizing_player, &result.best_action);
        auto now = std::chrono::steady_clock::now();
        using MilliDuration = std::chrono::duration<double, std::milli>;
        auto elapsed = std::chrono::duration_cast<MilliDuration>(
            now - start_time);
        double elapsed_ms = elapsed.count();
        last_root_times_ms_.push_back(elapsed_ms);
      }
    }
  } else {
    result.value = SearchState(state, config_.max_depth, maximizing_player,
                               &result.best_action);
  }
  if (cache_stack_ != nullptr) {
    cache_stack_->Clear();
  }
  return result;
}

std::vector<std::pair<Action, double>>
ExpectiminimaxSearch::EvaluateDecisionActions(LongNardeState* state) {
  std::vector<std::pair<Action, double>> results;
  if (state == nullptr || state->IsTerminal() || state->IsChanceNode()) {
    return results;
  }
  if (cache_stack_ != nullptr) {
    const nnue::NnueNetwork* network =
        (config_.use_nnue_cache && evaluator_ != nullptr)
            ? evaluator_->network()
            : nullptr;
    cache_stack_->Reset(*state, network);
  }
  Player maximizing_player = state->current_player_id();
  Player player = state->current_player_id();
  const std::array<int, 2> dice = state->dice();
  std::vector<Action> actions = state->LegalActions();
  results.reserve(actions.size());
  for (Action action : actions) {
    double child_value = 0.0;
    if (config_.use_undo) {
      state->ApplyAction(action);
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*state, action, dice, player);
      }
      int child_depth = config_.max_depth;
      if (state->awaiting_roll() && config_.max_depth > 0) {
        child_depth = config_.max_depth - 1;
      }
      child_value =
          SearchState(state, child_depth, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
      state->UndoAction(player, action);
    } else {
      std::unique_ptr<State> child = state->Child(action);
      auto* lnchild = static_cast<LongNardeState*>(child.get());
      int child_depth = config_.max_depth;
      if (lnchild->awaiting_roll() && config_.max_depth > 0) {
        child_depth = config_.max_depth - 1;
      }
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*lnchild, action, dice, player);
      }
      child_value =
          SearchState(lnchild, child_depth, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
    }
    results.push_back({action, child_value});
  }

  std::stable_sort(results.begin(), results.end(),
                   [player, maximizing_player](
                       const std::pair<Action, double>& a,
                       const std::pair<Action, double>& b) {
                     if (a.second == b.second) {
                       return a.first < b.first;
                     }
                     bool maximizing = player == maximizing_player;
                     return maximizing ? (a.second > b.second)
                                       : (a.second < b.second);
                   });
  if (cache_stack_ != nullptr) {
    cache_stack_->Clear();
  }
  return results;
}

double ExpectiminimaxSearch::SearchState(LongNardeState* state, int depth,
                                         Player maximizing_player,
                                         Action* best_action) {
  SPIEL_CHECK_GE(depth, 0);
  if (state->IsTerminal()) {
    return state->PlayerReturn(maximizing_player);
  }

  if (!state->IsChanceNode() && config_.use_tt) {
    uint64_t key = HashState(*state);
    auto it = table_.find(key);
    if (it != table_.end() && it->second.depth >= depth) {
      if (best_action != nullptr) {
        *best_action = it->second.best_action;
      }
      return it->second.value;
    }
  }

  if (state->IsChanceNode()) {
    if (depth == 0) {
      return EvaluatePreRoll(*state, maximizing_player);
    }
    const auto& outcomes = state->ChanceOutcomes();
    auto eval_outcome = [&](Action outcome) {
      double child_value = 0.0;
      if (config_.use_undo) {
        state->ApplyAction(outcome);
        if (cache_stack_ != nullptr) {
          cache_stack_->PushChance();
        }
        child_value = SearchState(state, depth, maximizing_player, nullptr);
        if (cache_stack_ != nullptr) {
          cache_stack_->Pop();
        }
        state->UndoAction(kChancePlayerId, outcome);
      } else {
        std::unique_ptr<State> child = state->Child(outcome);
        auto* lnchild = static_cast<LongNardeState*>(child.get());
        if (cache_stack_ != nullptr) {
          cache_stack_->PushChance();
        }
        child_value = SearchState(lnchild, depth, maximizing_player, nullptr);
        if (cache_stack_ != nullptr) {
          cache_stack_->Pop();
        }
      }
      return child_value;
    };
    int sample_count = config_.chance_samples;
    bool sample =
        sample_count > 0 &&
        sample_count < static_cast<int>(outcomes.size()) &&
        (config_.chance_sample_depth <= 0 ||
         depth <= config_.chance_sample_depth);
    if (sample) {
      std::vector<double> cumulative;
      cumulative.reserve(outcomes.size());
      double total_prob = 0.0;
      for (const auto& outcome : outcomes) {
        total_prob += outcome.second;
        cumulative.push_back(total_prob);
      }
      std::mt19937_64 rng(config_.chance_seed ^ HashState(*state) ^
                          static_cast<uint64_t>(depth));
      std::uniform_real_distribution<double> dist(0.0, total_prob);
      double value = 0.0;
      for (int i = 0; i < sample_count; ++i) {
        double r = dist(rng);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        std::size_t idx = static_cast<std::size_t>(
            std::distance(cumulative.begin(), it));
        if (idx >= outcomes.size()) {
          idx = outcomes.size() - 1;
        }
        value += eval_outcome(outcomes[idx].first);
      }
      return total_prob * (value / static_cast<double>(sample_count));
    }
    double value = 0.0;
    for (const auto& outcome : outcomes) {
      value += outcome.second * eval_outcome(outcome.first);
    }
    return value;
  }

  Player player = state->current_player_id();
  bool maximizing = player == maximizing_player;
  Action best = kInvalidAction;

  std::vector<std::pair<Action, double>> shallow_scores;
  bool collect_scores =
      config_.enable_panic_widen && config_.enable_pruning &&
      depth == config_.max_depth;
  std::vector<Action> actions = OrderedActions(
      state, depth, maximizing_player,
      collect_scores ? &shallow_scores : nullptr);

  double best_value =
      EvaluateActionList(state, actions, depth, maximizing_player, &best);

  if (!shallow_scores.empty()) {
    double best_shallow = shallow_scores.front().second;
    double drop = maximizing ? (best_shallow - best_value)
                             : (best_value - best_shallow);
    if (drop > config_.panic_margin) {
      int panic_top_k = config_.panic_top_k;
      if (panic_top_k <= 0) {
        panic_top_k = static_cast<int>(shallow_scores.size());
      }
      int panic_min_k = config_.panic_min_k;
      if (panic_min_k <= 0) {
        panic_min_k = config_.min_k;
      }
      double panic_delta = config_.panic_delta;
      if (panic_delta <= 0.0) {
        panic_delta = config_.delta;
      }
      panic_top_k = std::max(panic_top_k, config_.top_k);
      panic_min_k = std::max(panic_min_k, config_.min_k);
      panic_delta = std::max(panic_delta, config_.delta);
      std::vector<Action> widened =
          PruneActions(shallow_scores, panic_min_k, panic_top_k, panic_delta);
      if (widened.size() > actions.size()) {
        best_value = EvaluateActionList(state, widened, depth,
                                        maximizing_player, &best);
      }
    }
  }

  if (config_.use_tt) {
    uint64_t key = HashState(*state);
    if (config_.max_tt_entries > 0 &&
        table_.size() >= static_cast<size_t>(config_.max_tt_entries)) {
      table_.clear();
      table_.reserve(config_.max_tt_entries);
    }
    table_[key] = TTEntry{depth, best_value, best};
  }
  if (best_action != nullptr) {
    *best_action = best;
  }
  return best_value;
}

double ExpectiminimaxSearch::EvaluateActionList(
    LongNardeState* state, const std::vector<Action>& actions, int depth,
    Player maximizing_player, Action* best_action) {
  if (actions.empty()) {
    if (best_action != nullptr) {
      *best_action = kInvalidAction;
    }
    return 0.0;
  }
  Player player = state->current_player_id();
  const std::array<int, 2> dice = state->dice();
  bool maximizing = player == maximizing_player;
  double best_value = maximizing ? -std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::infinity();
  Action best = kInvalidAction;
  for (Action action : actions) {
    double child_value = 0.0;
    if (config_.use_undo) {
      state->ApplyAction(action);
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*state, action, dice, player);
      }
      int child_depth = depth;
      if (state->awaiting_roll() && depth > 0) {
        child_depth = depth - 1;
      }
      child_value =
          SearchState(state, child_depth, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
      state->UndoAction(player, action);
    } else {
      std::unique_ptr<State> child = state->Child(action);
      LongNardeState* child_state = static_cast<LongNardeState*>(child.get());
      int child_depth = depth;
      if (child_state->awaiting_roll() && depth > 0) {
        child_depth = depth - 1;
      }
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*child_state, action, dice, player);
      }
      child_value =
          SearchState(child_state, child_depth, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
    }
    if ((maximizing && child_value > best_value) ||
        (!maximizing && child_value < best_value) ||
        best == kInvalidAction) {
      best_value = child_value;
      best = action;
    }
  }
  if (best_action != nullptr) {
    *best_action = best;
  }
  return best_value;
}

double ExpectiminimaxSearch::EvaluatePreRoll(
    const LongNardeState& state, Player maximizing_player) const {
  SPIEL_CHECK_TRUE(state.awaiting_roll());
  if (evaluator_ == nullptr) {
    return 0.0;
  }
  nnue::NnueEval eval;
  const nnue::NnueCache* cache =
      cache_stack_ != nullptr ? cache_stack_->Current() : nullptr;
  const nnue::NnueNetwork* network =
      (cache != nullptr && evaluator_ != nullptr) ? evaluator_->network()
                                                  : nullptr;
  if (cache != nullptr && network != nullptr) {
    eval = nnue::EvaluateFromAccumulator(*network, cache->acc);
  } else {
    eval = evaluator_->EvaluateState(state);
  }
  double value = eval.ev;
  return (state.current_player_id() == maximizing_player) ? value : -value;
}

std::vector<std::pair<Action, double>> ExpectiminimaxSearch::ScoreActions(
    LongNardeState* state, Player maximizing_player) {
  std::vector<Action> actions = state->LegalActions();
  std::vector<std::pair<Action, double>> scores;
  scores.reserve(actions.size());
  Player player = state->current_player_id();
  const std::array<int, 2> dice = state->dice();
  for (Action action : actions) {
    double score = 0.0;
    if (config_.use_undo) {
      state->ApplyAction(action);
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*state, action, dice, player);
      }
      score = SearchState(state, 0, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
      state->UndoAction(player, action);
    } else {
      std::unique_ptr<State> child = state->Child(action);
      auto* lnchild = static_cast<LongNardeState*>(child.get());
      if (cache_stack_ != nullptr) {
        cache_stack_->PushAction(*lnchild, action, dice, player);
      }
      score = SearchState(lnchild, 0, maximizing_player, nullptr);
      if (cache_stack_ != nullptr) {
        cache_stack_->Pop();
      }
    }
    scores.push_back({action, score});
  }

  bool maximizing = player == maximizing_player;
  std::stable_sort(scores.begin(), scores.end(),
                   [maximizing](const std::pair<Action, double>& a,
                                const std::pair<Action, double>& b) {
                     if (a.second == b.second) {
                       return a.first < b.first;
                     }
                     return maximizing ? (a.second > b.second)
                                       : (a.second < b.second);
                   });
  return scores;
}

std::vector<Action> ExpectiminimaxSearch::PruneActions(
    const std::vector<std::pair<Action, double>>& scores, int min_k,
    int top_k, double delta) const {
  std::vector<Action> pruned;
  if (scores.empty()) {
    return pruned;
  }
  int effective_top_k = top_k <= 0 ? static_cast<int>(scores.size()) : top_k;
  double best_score = scores.front().second;
  pruned.reserve(scores.size());
  for (const auto& entry : scores) {
    if (static_cast<int>(pruned.size()) < min_k ||
        static_cast<int>(pruned.size()) < effective_top_k ||
        std::fabs(entry.second - best_score) <= delta) {
      pruned.push_back(entry.first);
    }
  }
  if (pruned.empty()) {
    pruned.push_back(scores.front().first);
  }
  return pruned;
}

std::vector<Action> ExpectiminimaxSearch::OrderedActions(
    LongNardeState* state, int depth, Player maximizing_player,
    std::vector<std::pair<Action, double>>* scores_out) {
  std::vector<Action> actions = state->LegalActions();
  if (!config_.enable_pruning || config_.top_k <= 0 ||
      static_cast<int>(actions.size()) <= config_.top_k) {
    std::sort(actions.begin(), actions.end());
    if (scores_out != nullptr) {
      scores_out->clear();
    }
    return actions;
  }

  std::vector<std::pair<Action, double>> scores =
      ScoreActions(state, maximizing_player);
  std::vector<Action> pruned =
      PruneActions(scores, config_.min_k, config_.top_k, config_.delta);
  if (scores_out != nullptr) {
    *scores_out = std::move(scores);
  }
  return pruned;
}

}  // namespace long_narde
}  // namespace open_spiel
