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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nats.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_args.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_config.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_config_utils.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_play.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_stats.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct PendingPayload {
  std::string payload;
  int samples = 0;
};

struct WeightsStore {
  mutable std::mutex mu;
  std::string data;
  int version = 0;

  void Update(const std::string& payload) {
    std::lock_guard<std::mutex> lock(mu);
    data = payload;
    version += 1;
  }

  bool HasData() const {
    std::lock_guard<std::mutex> lock(mu);
    return !data.empty();
  }

  bool GetIfNew(int* version_out, std::string* payload_out) const {
    if (version_out == nullptr || payload_out == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (version == *version_out) {
      return false;
    }
    *version_out = version;
    *payload_out = data;
    return !payload_out->empty();
  }
};

std::string Timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm tm_snapshot{};
#if defined(_WIN32)
  localtime_s(&tm_snapshot, &now_c);
#else
  localtime_r(&now_c, &tm_snapshot);
#endif
  std::ostringstream out;
  out << std::put_time(&tm_snapshot, "%F %T");
  return out.str();
}

double ElapsedSeconds() {
  static const auto start_time = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             now - start_time)
      .count();
}

void RequestLatestWeights(const WorkerConfig& config, const WeightsStore* store,
                          const std::string& inbox_subject) {
  if (store == nullptr || inbox_subject.empty()) {
    return;
  }
  if (!config.request_weights) {
    return;
  }
  std::string subject = BuildSubject(config.request_subject, config.run_id);
  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    return;
  }
  int interval_ms = config.request_interval_ms;
  if (interval_ms < 50) {
    interval_ms = 50;
  }
  while (!store->HasData()) {
    conn.Publish(subject, inbox_subject);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(interval_ms));
  }
}


void WeightSubscriber(const WorkerConfig& config, WeightsStore* store) {
  if (store == nullptr) {
    return;
  }
  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    std::cerr << "Failed to connect to NATS at " << config.nats_url << "\n";
    return;
  }
  std::string weights_subject =
      BuildSubject(config.weights_subject, config.run_id);
  if (!conn.Subscribe(weights_subject, 1)) {
    std::cerr << "Failed to subscribe to " << weights_subject << "\n";
    return;
  }
  std::string inbox_subject;
  if (config.request_weights) {
    inbox_subject = BuildSubject(
        "nnue.inbox." + std::to_string(config.seed) + "." +
            std::to_string(
                static_cast<int64_t>(
                    std::chrono::steady_clock::now()
                        .time_since_epoch()
                        .count())),
        config.run_id);
    if (!conn.Subscribe(inbox_subject, 2)) {
      std::cerr << "Failed to subscribe to " << inbox_subject << "\n";
      return;
    }
    std::thread request_thread(RequestLatestWeights, config, store,
                               inbox_subject);
    request_thread.detach();
  }

  NatsMessage msg;
  while (conn.NextMessage(&msg)) {
    store->Update(msg.payload);
  }
}

void WorkerLoop(std::shared_ptr<const Game> game, const WorkerConfig& config,
                std::atomic<int64_t>* next_game,
                const WeightsStore* store, WorkerStats* stats,
                int worker_id) {
  NatsConnection conn;
  bool connected = conn.Connect(config.nats_url);
  if (!connected) {
    std::cerr << "Worker " << worker_id << " failed to connect to NATS.\n";
  }
  std::string subject = BuildSubject(config.traj_subject, config.run_id);

  auto model = std::make_unique<nnue::NnueModel>();
  if (!config.nnue_path.empty()) {
    model->Load(config.nnue_path);
    if (worker_id == 0) {
      std::cerr << "[" << Timestamp()
                << "] [worker] loaded weights from file "
                << config.nnue_path << "\n";
    }
  } else if (config.wait_for_weights && store != nullptr) {
    int local_version = 0;
    std::string payload;
    while (!store->GetIfNew(&local_version, &payload)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    model->LoadFromBytes(payload);
    if (worker_id == 0) {
      std::cerr << "[" << Timestamp()
                << "] [worker] loaded weights from NATS version="
                << local_version << " bytes=" << payload.size() << "\n";
    }
  }
  nnue::NnueEvaluator evaluator(model.get());

  SearchConfig search_config;
  search_config.max_depth = config.depth;
  search_config.max_tt_entries = config.tt_entries;
  search_config.root_full_depth_top_k = config.root_full_depth_top_k;
  search_config.root_reduced_depth = config.root_reduced_depth;
  search_config.chance_samples = config.chance_samples;
  search_config.chance_sample_depth = config.chance_sample_depth;
  search_config.chance_seed = config.chance_seed;
  ExpectiminimaxSearch search(&evaluator, search_config);

  LnueShardConfig shard_config;
  shard_config.samples_per_chunk = 4096;
  shard_config.run_block_threshold = nnue::kNnueRunBlockThreshold;

  int local_version = 0;
  std::deque<PendingPayload> pending;
  int retry_ms = 200;
  auto next_retry = std::chrono::steady_clock::now();
  auto mark_pending = [&](int count) {
    if (stats != nullptr && count != 0) {
      stats->pending.fetch_add(count, std::memory_order_relaxed);
    }
  };
  auto ensure_connected = [&]() -> bool {
    if (connected) {
      return true;
    }
    auto now = std::chrono::steady_clock::now();
    if (now < next_retry) {
      return false;
    }
    if (conn.Connect(config.nats_url)) {
      connected = true;
      retry_ms = 200;
      return true;
    }
    retry_ms = std::min(retry_ms * 2, 5000);
    next_retry = now + std::chrono::milliseconds(retry_ms);
    return false;
  };
  auto flush_pending = [&]() {
    if (!connected) {
      return;
    }
    while (!pending.empty()) {
      if (!conn.Publish(subject, pending.front().payload)) {
        conn.Close();
        connected = false;
        next_retry = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(retry_ms);
        break;
      }
      if (stats != nullptr) {
        int64_t total_games = stats->AddGame();
        if (config.report_every_games > 0 &&
            total_games % config.report_every_games == 0) {
          int64_t total_samples =
              stats->samples.load(std::memory_order_relaxed);
          int64_t pending_now =
              stats->pending.load(std::memory_order_relaxed);
          std::cerr << "[" << Timestamp()
                    << "] [worker] completed games=" << total_games
                    << " samples=" << total_samples
                    << " pending=" << pending_now
                    << " elapsed_s=" << ElapsedSeconds() << "\n";
        }
      }
      pending.pop_front();
      mark_pending(-1);
    }
  };
  while (true) {
    int64_t game_id = next_game->fetch_add(1);
    if (config.games > 0 && game_id >= config.games) {
      break;
    }

    if (store != nullptr) {
      std::string payload;
      if (store->GetIfNew(&local_version, &payload)) {
        model->LoadFromBytes(payload);
        if (stats != nullptr) {
          stats->weights_version.store(local_version,
                                       std::memory_order_relaxed);
        }
        if (worker_id == 0) {
          std::cerr << "[" << Timestamp()
                    << "] [worker] updated weights from NATS version="
                    << local_version << " bytes=" << payload.size() << "\n";
        }
      }
    }

    if (!pending.empty()) {
      ensure_connected();
      flush_pending();
    }

    uint64_t game_seed = config.seed + static_cast<uint64_t>(game_id) * 9973u;
    std::mt19937 rng(static_cast<uint32_t>(game_seed));

    std::vector<SelfPlaySample> samples =
        PlayOneGame(game, &search, config,
                    static_cast<uint64_t>(game_id), &rng, stats);

    std::string payload;
    if (!SerializeLnueTrajectory(samples, shard_config, &payload)) {
      continue;
    }
    PendingPayload current;
    current.payload = std::move(payload);
    current.samples = static_cast<int>(samples.size());
    if (!connected && !ensure_connected()) {
      pending.push_back(std::move(current));
      mark_pending(1);
      continue;
    }
    if (!conn.Publish(subject, current.payload)) {
      conn.Close();
      connected = false;
      next_retry = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(retry_ms);
      pending.push_back(std::move(current));
      mark_pending(1);
      continue;
    }
    if (stats != nullptr) {
      int64_t total_games = stats->AddGame();
      if (config.report_every_games > 0 &&
          total_games % config.report_every_games == 0) {
        int64_t total_samples =
            stats->samples.load(std::memory_order_relaxed);
        int64_t pending_now =
            stats->pending.load(std::memory_order_relaxed);
        std::cerr << "[" << Timestamp()
                  << "] [worker] completed games=" << total_games
                  << " samples=" << total_samples
                  << " pending=" << pending_now
                  << " elapsed_s=" << ElapsedSeconds() << "\n";
      }
    }
  }
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin
            << " [--nats url] [--run_id id] [--traj_subject name]"
            << " [--weights_subject name] [--request_subject name]\n"
            << "             [--config_subject name]"
            << " [--config_request_subject name]\n"
            << "             [--nnue path] [--depth N] [--workers N]"
            << " [--games N]\n"
            << "             [--temperature T] [--temperature_end T]"
            << " [--temperature_decay_plies N]\n"
            << "             [--alpha A] [--seed N] [--wait_for_weights 0|1]\n"
            << "             [--request_weights 0|1]"
            << " [--request_interval_ms N]"
            << " [--request_config 0|1] [--config_timeout_ms N]\n"
            << "             [--report_every_seconds N]"
            << " [--report_every_games N] [--tt_entries N]\n"
            << "             [--root_full_depth_top_k N]"
            << " [--root_reduced_depth N]\n"
            << "             [--chance_samples N] [--chance_sample_depth N]"
            << " [--chance_seed N]\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::WorkerConfig;
  using open_spiel::long_narde::ApplyConfigArgs;
  using open_spiel::long_narde::FetchRemoteConfig;
  using open_spiel::long_narde::HasArg;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::ParseConfigPayload;
  using open_spiel::long_narde::GetDoubleArg;
  using open_spiel::long_narde::GetIntArg;
  using open_spiel::long_narde::GetInt64Arg;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::GetUint64Arg;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::RunStatsReporter;
  using open_spiel::long_narde::WeightSubscriber;
  using open_spiel::long_narde::WorkerLoop;
  using open_spiel::long_narde::WorkerStats;
  using open_spiel::long_narde::WeightsStore;
  using open_spiel::long_narde::nnue::NnueKernelName;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  WorkerConfig config;
  ApplyConfigArgs(args, &config);

  std::unordered_map<std::string, std::string> remote_args;
  std::string remote_payload;
  if (config.request_config &&
      FetchRemoteConfig(config, &remote_payload)) {
    remote_args = ParseConfigPayload(remote_payload);
    ApplyConfigArgs(remote_args, &config);
    std::cerr << "[" << open_spiel::long_narde::Timestamp()
              << "] [worker] loaded config from NATS subject "
              << open_spiel::long_narde::BuildSubject(
                     config.config_subject, config.run_id)
              << "\n";
  }
  ApplyConfigArgs(args, &config);
  bool wait_set =
      HasArg(args, "wait_for_weights") ||
      HasArg(remote_args, "wait_for_weights");
  if (!wait_set) {
    config.wait_for_weights = config.nnue_path.empty();
  }
  bool request_set =
      HasArg(args, "request_weights") ||
      HasArg(remote_args, "request_weights");
  if (!request_set) {
    config.request_weights = config.nnue_path.empty();
  }
  std::cerr << "[" << open_spiel::long_narde::Timestamp()
            << "] [worker] config: depth="
            << config.depth
            << " root_full_depth_top_k=" << config.root_full_depth_top_k
            << " root_reduced_depth=" << config.root_reduced_depth
            << " chance_samples=" << config.chance_samples
            << " chance_sample_depth=" << config.chance_sample_depth
            << " temperature=" << config.temperature
            << " temperature_end=" << config.temperature_end
            << " temperature_decay_plies=" << config.temperature_decay_plies
            << " alpha=" << config.alpha
            << " report_every_games=" << config.report_every_games
            << " report_every_seconds=" << config.report_every_seconds << "\n";

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("long_narde");

  int workers = config.workers;
  if (workers <= 0) {
    unsigned int hc = std::thread::hardware_concurrency();
    workers = hc == 0 ? 1 : static_cast<int>(hc) + 1;
  }
  config.workers = workers;
  std::cerr << "[worker] nnue_kernel=" << NnueKernelName()
            << " workers=" << config.workers << "\n";

  WeightsStore store;
  std::thread sub_thread(WeightSubscriber, config, &store);
  sub_thread.detach();

  WorkerStats stats;
  std::atomic<bool> reporter_done{false};
  std::thread reporter_thread;
  if (config.report_every_seconds > 0) {
    reporter_thread = std::thread(RunStatsReporter,
                                  config.report_every_seconds, &stats,
                                  &reporter_done);
  }

  std::atomic<int64_t> next_game{0};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int i = 0; i < workers; ++i) {
    threads.emplace_back(WorkerLoop, game, config, &next_game, &store, &stats,
                         i);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  reporter_done.store(true);
  if (reporter_thread.joinable()) {
    reporter_thread.join();
  }

  return 0;
}
