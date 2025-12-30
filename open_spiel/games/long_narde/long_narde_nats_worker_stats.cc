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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace open_spiel {
namespace long_narde {

void WorkerStats::AddGame() {
  games.fetch_add(1, std::memory_order_relaxed);
}

void WorkerStats::AddSamples(int64_t count) {
  samples.fetch_add(count, std::memory_order_relaxed);
}

void WorkerStats::AddSearchTimings(const std::vector<double>& times_ms) {
  int limit = std::min(static_cast<int>(times_ms.size()), kMaxDepthStats);
  for (int i = 0; i < limit; ++i) {
    int64_t us = static_cast<int64_t>(times_ms[i] * 1000.0);
    depth_time_us[i].fetch_add(us, std::memory_order_relaxed);
    depth_calls[i].fetch_add(1, std::memory_order_relaxed);
  }
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
    std::string depth_summary;
    for (int i = 0; i < WorkerStats::kMaxDepthStats; ++i) {
      int64_t calls = stats->depth_calls[i].load(std::memory_order_relaxed);
      if (calls <= 0) {
        continue;
      }
      int64_t total_us =
          stats->depth_time_us[i].load(std::memory_order_relaxed);
      double avg_s =
          static_cast<double>(total_us) / 1e6 / static_cast<double>(calls);
      depth_summary += " d" + std::to_string(i + 1) + "=" +
                       std::to_string(avg_s);
    }
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
              << window_samples_s << " s/s avg=" << avg_samples << " s/g";
    if (!depth_summary.empty()) {
      std::cerr << " depth_s=" << depth_summary;
    }
    std::cerr << "\n";
    last_time = now;
    last_games = total_games;
    last_samples = total_samples;
  }
}

}  // namespace long_narde
}  // namespace open_spiel
