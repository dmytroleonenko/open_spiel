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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_STATS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_STATS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace open_spiel {
namespace long_narde {

struct WorkerStats {
  static constexpr int kMaxDepthStats = 8;
  std::atomic<int64_t> games{0};
  std::atomic<int64_t> samples{0};
  std::atomic<int64_t> pending{0};
  std::atomic<int> weights_version{0};
  std::array<std::atomic<int64_t>, kMaxDepthStats> depth_time_us{};
  std::array<std::atomic<int64_t>, kMaxDepthStats> depth_calls{};

  void AddGame();
  void AddSamples(int64_t count);
  void AddSearchTimings(const std::vector<double>& times_ms);
};

void RunStatsReporter(int report_every_seconds, const WorkerStats* stats,
                      std::atomic<bool>* done);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_STATS_H_
