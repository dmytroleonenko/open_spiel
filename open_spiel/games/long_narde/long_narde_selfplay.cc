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

#include "open_spiel/games/long_narde/long_narde_selfplay.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct PendingSample {
  SelfPlaySample sample;
};

Action SampleChanceOutcome(
    const std::vector<std::pair<Action, double>>& outcomes,
    std::mt19937* rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(*rng);
  double acc = 0.0;
  for (const auto& outcome : outcomes) {
    acc += outcome.second;
    if (r <= acc) {
      return outcome.first;
    }
  }
  return outcomes.back().first;
}

Action SampleSoftmax(const std::vector<std::pair<Action, double>>& actions,
                     double temperature, std::mt19937* rng) {
  SPIEL_CHECK_FALSE(actions.empty());
  if (temperature <= 1e-6 || actions.size() == 1) {
    return actions.front().first;
  }
  double max_value = actions.front().second;
  for (const auto& entry : actions) {
    max_value = std::max(max_value, entry.second);
  }

  std::vector<double> weights(actions.size());
  double sum = 0.0;
  for (size_t i = 0; i < actions.size(); ++i) {
    double w = std::exp((actions[i].second - max_value) / temperature);
    weights[i] = w;
    sum += w;
  }
  if (sum <= 0.0) {
    return actions.front().first;
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  double r = dist(*rng);
  double acc = 0.0;
  for (size_t i = 0; i < actions.size(); ++i) {
    acc += weights[i];
    if (r <= acc) {
      return actions[i].first;
    }
  }
  return actions.back().first;
}

void FinalizeSamples(const std::vector<double>& returns,
                     const SelfPlayConfig& config,
                     std::vector<PendingSample>* pending,
                     std::vector<SelfPlaySample>* out) {
  for (auto& entry : *pending) {
    Player p = entry.sample.player;
    SPIEL_CHECK_GE(p, 0);
    SPIEL_CHECK_LT(p, kNumPlayers);
    double outcome = returns[p];
    entry.sample.outcome_value = outcome;
    entry.sample.target_value =
        config.alpha * entry.sample.search_value +
        (1.0 - config.alpha) * entry.sample.outcome_value;
    out->push_back(std::move(entry.sample));
  }
  pending->clear();
}

SelfPlayBatch RunWorker(std::shared_ptr<const Game> game,
                        const nnue::NnueEvaluator& evaluator,
                        const SearchConfig& search_config,
                        const SelfPlayConfig& config, int games,
                        uint64_t seed) {
  SelfPlayBatch batch;
  if (games <= 0) {
    return batch;
  }
  std::mt19937 rng(seed);
  ExpectiminimaxSearch search(&evaluator, search_config);

  for (int g = 0; g < games; ++g) {
    search.ClearCache();
    std::unique_ptr<State> state = game->NewInitialState();
    auto* lnstate = static_cast<LongNardeState*>(state.get());

    std::vector<PendingSample> pending;
    int moves = 0;

    while (!lnstate->IsTerminal() && moves < config.max_moves) {
      if (lnstate->IsChanceNode()) {
        if (!lnstate->initial_roll()) {
          PendingSample entry;
          entry.sample.player = lnstate->current_player_id();
          nnue::CollectActiveFeatureIndices(*lnstate,
                                            &entry.sample.active_features);
          entry.sample.search_value = search.Search(lnstate).value;
          pending.push_back(std::move(entry));
        }
        Action chance_action = SampleChanceOutcome(lnstate->ChanceOutcomes(),
                                                   &rng);
        lnstate->ApplyAction(chance_action);
      } else {
        std::vector<std::pair<Action, double>> scored =
            search.EvaluateDecisionActions(lnstate);
        Action action = SampleSoftmax(scored, config.temperature, &rng);
        lnstate->ApplyAction(action);
        moves += 1;
      }
    }

    if (lnstate->IsTerminal()) {
      std::vector<double> returns = lnstate->Returns();
      FinalizeSamples(returns, config, &pending, &batch.samples);
    }
    batch.stats.games += 1;
    batch.stats.total_moves += moves;
  }

  return batch;
}

}  // namespace

SelfPlayBatch RunSelfPlay(std::shared_ptr<const Game> game,
                          const nnue::NnueEvaluator& evaluator,
                          const SearchConfig& search_config,
                          const SelfPlayConfig& selfplay_config) {
  SPIEL_CHECK_TRUE(game != nullptr);
  SelfPlayBatch batch;
  int workers = selfplay_config.num_workers;
  if (workers <= 0) {
    unsigned int hc = std::thread::hardware_concurrency();
    workers = hc == 0 ? 1 : static_cast<int>(hc);
  }
  int total_games = std::max(selfplay_config.num_games, 0);
  workers = std::min(workers, std::max(total_games, 1));

  std::vector<SelfPlayBatch> partials(workers);
  std::vector<Thread> threads;
  threads.reserve(workers);

  int base = total_games / workers;
  int extra = total_games % workers;
  for (int i = 0; i < workers; ++i) {
    int count = base + (i < extra ? 1 : 0);
    uint64_t seed = selfplay_config.seed + static_cast<uint64_t>(i) * 9973u;
    threads.emplace_back([&, i, count, seed]() {
      partials[i] =
          RunWorker(game, evaluator, search_config, selfplay_config, count,
                    seed);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (auto& part : partials) {
    batch.stats.games += part.stats.games;
    batch.stats.total_moves += part.stats.total_moves;
    batch.samples.insert(batch.samples.end(),
                         std::make_move_iterator(part.samples.begin()),
                         std::make_move_iterator(part.samples.end()));
  }

  return batch;
}

}  // namespace long_narde
}  // namespace open_spiel
