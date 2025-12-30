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

#include "open_spiel/games/long_narde/long_narde_nats_worker_play.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_stats.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct PendingSample {
  SelfPlaySample sample;
};

double TemperatureForPly(const WorkerConfig& config, int ply) {
  double end = config.temperature_end;
  if (end < 0.0) {
    end = config.temperature;
  }
  if (config.temperature_decay_plies <= 0 || end == config.temperature) {
    return config.temperature;
  }
  double t = static_cast<double>(ply) /
             static_cast<double>(config.temperature_decay_plies);
  if (t >= 1.0) {
    return end;
  }
  return config.temperature + (end - config.temperature) * t;
}

Action SampleChanceOutcome(const ActionsAndProbs& outcomes,
                           std::mt19937* rng) {
  double sum = 0.0;
  for (const auto& outcome : outcomes) {
    sum += outcome.second;
  }
  if (sum <= 0.0) {
    return outcomes.front().first;
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
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
  if (actions.empty()) {
    return kInvalidAction;
  }
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
                     const WorkerConfig& config,
                     std::vector<PendingSample>* pending,
                     std::vector<SelfPlaySample>* out) {
  for (auto& entry : *pending) {
    Player p = entry.sample.player;
    double outcome = returns[p];
    entry.sample.outcome_value = outcome;
    entry.sample.target_value =
        config.alpha * entry.sample.search_value +
        (1.0 - config.alpha) * entry.sample.outcome_value;
    out->push_back(std::move(entry.sample));
  }
  pending->clear();
}

}  // namespace

std::vector<SelfPlaySample> PlayOneGame(std::shared_ptr<const Game> game,
                                        ExpectiminimaxSearch* search,
                                        const WorkerConfig& config,
                                        uint64_t game_id, std::mt19937* rng,
                                        WorkerStats* stats) {
  std::vector<SelfPlaySample> samples;
  std::unique_ptr<State> state = game->NewInitialState();
  auto* lnstate = static_cast<LongNardeState*>(state.get());

  std::vector<PendingSample> pending;
  int moves = 0;
  int ply = 0;

  search->ClearCache();
  while (!lnstate->IsTerminal() && moves < config.max_moves) {
    if (lnstate->IsChanceNode()) {
      if (!lnstate->initial_roll()) {
        PendingSample entry;
        entry.sample.player = lnstate->current_player_id();
        entry.sample.game_id =
            (static_cast<uint64_t>(config.seed) << 32) ^ game_id;
        entry.sample.ply = static_cast<uint16_t>(ply);
        nnue::CollectActiveFeatureIndices(*lnstate,
                                          &entry.sample.active_features);
        SearchResult result = search->Search(lnstate);
        entry.sample.search_value = result.value;
        if (stats != nullptr) {
          stats->AddSearchTimings(search->last_root_times_ms());
        }
        pending.push_back(std::move(entry));
        if (stats != nullptr) {
          stats->AddSamples(1);
        }
      }
      Action chance_action =
          SampleChanceOutcome(lnstate->ChanceOutcomes(), rng);
      lnstate->ApplyAction(chance_action);
    } else {
      std::vector<std::pair<Action, double>> scored =
          search->EvaluateDecisionActions(lnstate);
      double temperature = TemperatureForPly(config, ply);
      Action action = SampleSoftmax(scored, temperature, rng);
      lnstate->ApplyAction(action);
      moves += 1;
      ply += 1;
    }
  }

  if (lnstate->IsTerminal()) {
    std::vector<double> returns = lnstate->Returns();
    FinalizeSamples(returns, config, &pending, &samples);
  }
  return samples;
}

}  // namespace long_narde
}  // namespace open_spiel
