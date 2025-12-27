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

#include "open_spiel/games/long_narde/long_narde_nats_worker_stats.h"

#include <chrono>
#include <iostream>
#include <thread>

namespace open_spiel {
namespace long_narde {

void WorkerStats::AddGame(int64_t samples_in_game) {
  games.fetch_add(1, std::memory_order_relaxed);
  samples.fetch_add(samples_in_game, std::memory_order_relaxed);
}

void RunStatsReporter(int report_every_seconds, const WorkerStats* stats,
                      std::atomic<bool>* done) {
  if (stats == nullptr || done == nullptr) {
    return;
  }
  int interval = report_every_seconds > 0 ? report_every_seconds : 1;
  auto start_time = std::chrono::steady_clock::now();
  auto last_time = start_time;
  int64_t last_games = 0;
  int64_t last_samples = 0;
  while (!done->load()) {
    std::this_thread::sleep_for(std::chrono::seconds(interval));
    int64_t total_games = stats->games.load(std::memory_order_relaxed);
    int64_t total_samples = stats->samples.load(std::memory_order_relaxed);
    int64_t pending = stats->pending.load(std::memory_order_relaxed);
    int version = stats->weights_version.load(std::memory_order_relaxed);
    auto now = std::chrono::steady_clock::now();
    double total_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                  start_time)
            .count();
    double window_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(now -
                                                                  last_time)
            .count();
    int64_t window_games = total_games - last_games;
    int64_t window_samples = total_samples - last_samples;
    double overall_games_s =
        total_elapsed > 0.0 ? total_games / total_elapsed : 0.0;
    double overall_samples_s =
        total_elapsed > 0.0 ? total_samples / total_elapsed : 0.0;
    double window_games_s =
        window_elapsed > 0.0 ? window_games / window_elapsed : 0.0;
    double window_samples_s =
        window_elapsed > 0.0 ? window_samples / window_elapsed : 0.0;
    double avg_samples =
        total_games > 0 ? static_cast<double>(total_samples) / total_games
                        : 0.0;
    std::cerr << "[worker] games=" << total_games
              << " samples=" << total_samples << " weights=" << version
              << " pending=" << pending
              << " overall=" << overall_games_s << " g/s " << overall_samples_s
              << " s/s window=" << window_games_s << " g/s "
              << window_samples_s << " s/s avg=" << avg_samples << " s/g\n";
    last_time = now;
    last_games = total_games;
    last_samples = total_samples;
  }
}

}  // namespace long_narde
}  // namespace open_spiel
