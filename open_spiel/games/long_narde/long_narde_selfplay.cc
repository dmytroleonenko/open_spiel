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
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue_features.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct PendingSample {
  SelfPlaySample sample;
};

struct StreamWriterState {
  LnueStreamWriter* writer = nullptr;
  std::mutex* mutex = nullptr;
  int flush_every_games = 1;
  int games_since_flush = 0;
  int64_t games_written = 0;
  int64_t samples_written = 0;
  std::function<void(int64_t, int64_t)> on_flush;
};

double TemperatureForPly(const SelfPlayConfig& config, int ply) {
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

bool WriteSamplesToStream(StreamWriterState* state,
                          std::vector<SelfPlaySample>* samples) {
  if (state == nullptr || state->writer == nullptr || state->mutex == nullptr ||
      samples == nullptr) {
    return false;
  }
  if (samples->empty()) {
    return true;
  }
  std::lock_guard<std::mutex> lock(*state->mutex);
  if (!state->writer->AddSamples(*samples)) {
    return false;
  }
  state->samples_written += static_cast<int64_t>(samples->size());
  state->games_written += 1;
  state->games_since_flush += 1;
  if (state->flush_every_games > 0 &&
      state->games_since_flush >= state->flush_every_games) {
    if (!state->writer->Flush()) {
      return false;
    }
    state->games_since_flush = 0;
    if (state->on_flush) {
      state->on_flush(state->games_written, state->samples_written);
    }
  }
  return true;
}

SelfPlayBatch RunWorker(std::shared_ptr<const Game> game,
                        const nnue::NnueEvaluator& evaluator,
                        const SearchConfig& search_config,
                        const SelfPlayConfig& config, int total_games,
                        std::atomic<int>* next_game,
                        std::atomic<int>* games_done,
                        const std::chrono::steady_clock::time_point* start_time,
                        std::mutex* cout_mutex,
                        int start_game,
                        StreamWriterState* writer_state) {
  SelfPlayBatch batch;
  if (total_games <= 0 || next_game == nullptr) {
    return batch;
  }
  ExpectiminimaxSearch search(&evaluator, search_config);

  while (true) {
    int game_id = next_game->fetch_add(1);
    if (game_id >= total_games) {
      break;
    }

    uint64_t game_seed =
        config.seed + static_cast<uint64_t>(game_id) * 9973u;
    std::mt19937 rng(static_cast<uint32_t>(game_seed));
    search.ClearCache();
    std::unique_ptr<State> state = game->NewInitialState();
    auto* lnstate = static_cast<LongNardeState*>(state.get());

    std::vector<PendingSample> pending;
    std::vector<SelfPlaySample> game_samples;
    int moves = 0;
    int ply = 0;

    while (!lnstate->IsTerminal() && moves < config.max_moves) {
      if (lnstate->IsChanceNode()) {
        if (!lnstate->initial_roll()) {
          PendingSample entry;
          entry.sample.player = lnstate->current_player_id();
          entry.sample.game_id =
              (static_cast<uint64_t>(config.seed) << 32) ^
              static_cast<uint64_t>(game_id);
          entry.sample.ply = static_cast<uint16_t>(ply);
          internal::Board opp_board = lnstate->board();
          internal::FlipBoard(&opp_board);
          entry.sample.mobility = nnue::MobilityValueFromBoard(opp_board);
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
        double temperature = TemperatureForPly(config, ply);
        Action action = SampleSoftmax(scored, temperature, &rng);
        lnstate->ApplyAction(action);
        moves += 1;
        ply += 1;
      }
    }

    if (lnstate->IsTerminal()) {
      std::vector<double> returns = lnstate->Returns();
      if (writer_state == nullptr) {
        FinalizeSamples(returns, config, &pending, &batch.samples);
      } else {
        FinalizeSamples(returns, config, &pending, &game_samples);
        if (!WriteSamplesToStream(writer_state, &game_samples)) {
          if (next_game != nullptr) {
            next_game->store(total_games);
          }
          break;
        }
      }
    }
    batch.stats.games += 1;
    batch.stats.total_moves += moves;

    if (games_done != nullptr && config.progress && config.report_every > 0) {
      int done_total = games_done->fetch_add(1) + 1;
      if (done_total % config.report_every == 0 ||
          done_total == total_games) {
        double elapsed = 0.0;
        if (start_time != nullptr) {
          elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - *start_time)
                        .count();
        }
        int done_new = std::max(done_total - start_game, 0);
        double rate = elapsed > 0.0 ? done_new / elapsed : 0.0;
        int remaining = std::max(total_games - done_total, 0);
        double eta = rate > 0.0 ? remaining / rate : 0.0;
        if (cout_mutex != nullptr) {
          std::lock_guard<std::mutex> lock(*cout_mutex);
          double pct = 100.0 * static_cast<double>(done_total) / total_games;
          std::ostream& out = std::cerr;
          out << "[selfplay] " << done_total << "/" << total_games << " ("
              << std::fixed << std::setprecision(1) << pct
              << "%) " << std::setprecision(2) << rate
              << " games/s ETA " << std::setprecision(0) << eta
              << "s\n";
          out.flush();
        }
      }
    }
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
    workers = hc == 0 ? 1 : static_cast<int>(hc) + 1;
  }
  int total_games = std::max(selfplay_config.num_games, 0);
  workers = std::min(workers, std::max(total_games, 1));

  std::vector<SelfPlayBatch> partials(workers);
  std::vector<Thread> threads;
  threads.reserve(workers);
  std::atomic<int> games_done{0};
  std::atomic<int> next_game{0};
  std::mutex cout_mutex;
  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < workers; ++i) {
    threads.emplace_back([&, i]() {
      partials[i] =
          RunWorker(game, evaluator, search_config, selfplay_config,
                    total_games, &next_game, &games_done, &start_time,
                    &cout_mutex, /*start_game=*/0, nullptr);
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

SelfPlayStats RunSelfPlayStreaming(std::shared_ptr<const Game> game,
                                   const nnue::NnueEvaluator& evaluator,
                                   const SearchConfig& search_config,
                                   const SelfPlayConfig& selfplay_config,
                                   LnueStreamWriter* writer,
                                   const SelfPlayStreamConfig& stream_config) {
  SPIEL_CHECK_TRUE(game != nullptr);
  SelfPlayStats stats;
  if (writer == nullptr) {
    return stats;
  }
  int workers = selfplay_config.num_workers;
  if (workers <= 0) {
    unsigned int hc = std::thread::hardware_concurrency();
    workers = hc == 0 ? 1 : static_cast<int>(hc) + 1;
  }
  int total_games = std::max(selfplay_config.num_games, 0);
  workers = std::min(workers, std::max(total_games, 1));

  std::vector<SelfPlayBatch> partials(workers);
  std::vector<Thread> threads;
  threads.reserve(workers);
  std::atomic<int> games_done{static_cast<int>(stream_config.start_game)};
  std::atomic<int> next_game{static_cast<int>(stream_config.start_game)};
  std::mutex cout_mutex;
  std::mutex writer_mutex;
  auto start_time = std::chrono::steady_clock::now();

  StreamWriterState writer_state;
  writer_state.writer = writer;
  writer_state.mutex = &writer_mutex;
  writer_state.flush_every_games = stream_config.flush_every_games;
  writer_state.games_written = stream_config.start_game;
  writer_state.samples_written = stream_config.start_samples;
  writer_state.on_flush = stream_config.on_flush;

  for (int i = 0; i < workers; ++i) {
    threads.emplace_back([&, i]() {
      partials[i] =
          RunWorker(game, evaluator, search_config, selfplay_config,
                    total_games, &next_game, &games_done, &start_time,
                    &cout_mutex, stream_config.start_game, &writer_state);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  {
    std::lock_guard<std::mutex> lock(writer_mutex);
    if (writer_state.writer->Flush() && writer_state.on_flush) {
      writer_state.on_flush(writer_state.games_written,
                            writer_state.samples_written);
    }
  }

  for (auto& part : partials) {
    stats.games += part.stats.games;
    stats.total_moves += part.stats.total_moves;
  }

  return stats;
}

}  // namespace long_narde
}  // namespace open_spiel
