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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nats.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct WorkerConfig {
  std::string nats_url = "nats://127.0.0.1:4222";
  std::string run_id = "default";
  std::string traj_subject = "lnue.traj";
  std::string weights_subject = "nnue.weights";
  std::string nnue_path;
  int depth = 1;
  int max_moves = 1000;
  int workers = 0;
  double temperature = 1.0;
  double alpha = 0.5;
  int64_t games = 0;
  uint64_t seed = 7;
};

struct PendingSample {
  SelfPlaySample sample;
};

struct WeightsStore {
  std::mutex mu;
  std::string data;
  int version = 0;

  void Update(const std::string& payload) {
    std::lock_guard<std::mutex> lock(mu);
    data = payload;
    version += 1;
  }

  bool GetIfNew(int* version_out, std::string* payload_out) {
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

std::unordered_map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::unordered_map<std::string, std::string> out;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      continue;
    }
    arg = arg.substr(2);
    auto eq = arg.find('=');
    if (eq != std::string::npos) {
      out[arg.substr(0, eq)] = arg.substr(eq + 1);
    } else {
      std::string value;
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        value = argv[++i];
      }
      out[arg] = value;
    }
  }
  return out;
}

int GetIntArg(const std::unordered_map<std::string, std::string>& args,
              const std::string& key, int default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

int64_t GetInt64Arg(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key, int64_t default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoll(it->second);
}

uint64_t GetUint64Arg(const std::unordered_map<std::string, std::string>& args,
                      const std::string& key, uint64_t default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return static_cast<uint64_t>(std::stoull(it->second));
}

double GetDoubleArg(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key, double default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return std::stod(it->second);
}

std::string GetStringArg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, const std::string& default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return it->second;
}

std::string BuildSubject(const std::string& base, const std::string& run_id) {
  if (run_id.empty()) {
    return base;
  }
  return base + "." + run_id;
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

std::vector<SelfPlaySample> PlayOneGame(
    std::shared_ptr<const Game> game, ExpectiminimaxSearch* search,
    const WorkerConfig& config, uint64_t game_id, std::mt19937* rng) {
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
        entry.sample.search_value = search->Search(lnstate).value;
        pending.push_back(std::move(entry));
      }
        Action chance_action =
            SampleChanceOutcome(lnstate->ChanceOutcomes(), rng);
      lnstate->ApplyAction(chance_action);
    } else {
      std::vector<std::pair<Action, double>> scored =
          search->EvaluateDecisionActions(lnstate);
      Action action = SampleSoftmax(scored, config.temperature, rng);
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

void WeightSubscriber(const WorkerConfig& config, WeightsStore* store) {
  if (store == nullptr) {
    return;
  }
  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    std::cerr << "Failed to connect to NATS at " << config.nats_url << "\n";
    return;
  }
  std::string subject = BuildSubject(config.weights_subject, config.run_id);
  if (!conn.Subscribe(subject, 1)) {
    std::cerr << "Failed to subscribe to " << subject << "\n";
    return;
  }

  NatsMessage msg;
  while (conn.NextMessage(&msg)) {
    store->Update(msg.payload);
  }
}

void WorkerLoop(std::shared_ptr<const Game> game, const WorkerConfig& config,
                std::atomic<int64_t>* next_game,
                const WeightsStore* store, int worker_id) {
  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    std::cerr << "Worker " << worker_id << " failed to connect to NATS.\n";
    return;
  }
  std::string subject = BuildSubject(config.traj_subject, config.run_id);

  nnue::NnueModel model;
  if (!config.nnue_path.empty()) {
    model.Load(config.nnue_path);
  }
  nnue::NnueEvaluator evaluator(&model);

  SearchConfig search_config;
  search_config.max_depth = config.depth;
  ExpectiminimaxSearch search(&evaluator, search_config);

  LnueShardConfig shard_config;
  shard_config.samples_per_chunk = 4096;
  shard_config.run_block_threshold = nnue::kNnueRunBlockThreshold;

  int local_version = 0;
  while (true) {
    int64_t game_id = next_game->fetch_add(1);
    if (config.games > 0 && game_id >= config.games) {
      break;
    }

    if (store != nullptr) {
      std::string payload;
      if (store->GetIfNew(&local_version, &payload)) {
        model.LoadFromBytes(payload);
      }
    }

    uint64_t game_seed = config.seed + static_cast<uint64_t>(game_id) * 9973u;
    std::mt19937 rng(static_cast<uint32_t>(game_seed));

    std::vector<SelfPlaySample> samples =
        PlayOneGame(game, &search, config,
                    static_cast<uint64_t>(game_id), &rng);

    std::string payload;
    if (!SerializeLnueTrajectory(samples, shard_config, &payload)) {
      continue;
    }
    conn.Publish(subject, payload);
  }
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin
            << " [--nats url] [--run_id id] [--traj_subject name]"
            << " [--weights_subject name]\n"
            << "             [--nnue path] [--depth N] [--workers N]"
            << " [--games N]\n"
            << "             [--temperature T] [--alpha A] [--seed N]\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::WorkerConfig;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::GetDoubleArg;
  using open_spiel::long_narde::GetIntArg;
  using open_spiel::long_narde::GetInt64Arg;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::GetUint64Arg;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::WeightSubscriber;
  using open_spiel::long_narde::WorkerLoop;
  using open_spiel::long_narde::WeightsStore;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  WorkerConfig config;
  config.nats_url = GetStringArg(args, "nats", config.nats_url);
  config.run_id = GetStringArg(args, "run_id", config.run_id);
  config.traj_subject = GetStringArg(args, "traj_subject", config.traj_subject);
  config.weights_subject =
      GetStringArg(args, "weights_subject", config.weights_subject);
  config.nnue_path = GetStringArg(args, "nnue", config.nnue_path);
  config.depth = GetIntArg(args, "depth", config.depth);
  config.workers = GetIntArg(args, "workers", config.workers);
  config.temperature = GetDoubleArg(args, "temperature", config.temperature);
  config.alpha = GetDoubleArg(args, "alpha", config.alpha);
  config.games = GetInt64Arg(args, "games", config.games);
  config.seed = GetUint64Arg(args, "seed", config.seed);

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("long_narde");

  int workers = config.workers;
  if (workers <= 0) {
    unsigned int hc = std::thread::hardware_concurrency();
    workers = hc == 0 ? 1 : static_cast<int>(hc);
  }

  WeightsStore store;
  std::thread sub_thread(WeightSubscriber, config, &store);
  sub_thread.detach();

  std::atomic<int64_t> next_game{0};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int i = 0; i < workers; ++i) {
    threads.emplace_back(WorkerLoop, game, config, &next_game, &store, i);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  return 0;
}
