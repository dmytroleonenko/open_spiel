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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct EvalConfig {
  int games = 1000;
  int depth = 2;
  int max_moves = 1000;
  int report_every = 100;
  uint64_t seed = 7;
  int workers = 0;
  std::string nnue_a;
  std::string nnue_b;
  std::string out_path;
  bool progress = true;
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

uint64_t GetUint64Arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, uint64_t default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return static_cast<uint64_t>(std::stoull(it->second));
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

bool GetBoolArg(const std::unordered_map<std::string, std::string>& args,
                const std::string& key, bool default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  std::string value = it->second;
  return value == "1" || value == "true" || value == "yes";
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin
            << " [--games N] [--depth N] [--seed N] [--max_moves N]\n"
            << "             [--nnue_a path] [--nnue_b path]\n"
            << "             [--out path] [--progress 0|1] [--workers N]"
            << " [--report_every N]\n";
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

Action SampleRandomAction(const std::vector<Action>& actions,
                          std::mt19937* rng) {
  SPIEL_CHECK_FALSE(actions.empty());
  std::uniform_int_distribution<> dist(0, actions.size() - 1);
  return actions[dist(*rng)];
}

std::string LabelForPath(const std::string& path, const std::string& fallback) {
  if (path.empty()) {
    return fallback;
  }
  std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos || pos + 1 >= path.size()) {
    return path;
  }
  return path.substr(pos + 1);
}

struct EvalResult {
  int games = 0;
  int wins_a = 0;
  int wins_b = 0;
  int draws = 0;
  int mars_a = 0;
  int mars_b = 0;
  double sum_return_a = 0.0;
  double sum_return_b = 0.0;
};

EvalResult RunEvalWorker(std::shared_ptr<const Game> game,
                         const NnueEvaluator* eval_a,
                         const NnueEvaluator* eval_b,
                         const SearchConfig& search_config,
                         const EvalConfig& config, int total_games,
                         std::atomic<int>* next_game,
                         std::atomic<int>* games_done,
                         const std::chrono::steady_clock::time_point*
                             start_time,
                         std::mutex* cout_mutex) {
  EvalResult result;
  if (total_games <= 0 || next_game == nullptr) {
    return result;
  }
  ExpectiminimaxSearch search_a(eval_a, search_config);
  ExpectiminimaxSearch search_b(eval_b, search_config);

  while (true) {
    int game_id = next_game->fetch_add(1);
    if (game_id >= total_games) {
      break;
    }
    uint64_t game_seed =
        config.seed + static_cast<uint64_t>(game_id) * 9973u;
    std::mt19937 rng(static_cast<uint32_t>(game_seed));
    search_a.ClearCache();
    search_b.ClearCache();

    std::unique_ptr<State> state = game->NewInitialState();
    auto* lnstate = static_cast<LongNardeState*>(state.get());

    int moves = 0;
    while (!lnstate->IsTerminal() && moves < config.max_moves) {
      if (lnstate->IsChanceNode()) {
        Action chance_action =
            SampleChanceOutcome(lnstate->ChanceOutcomes(), &rng);
        lnstate->ApplyAction(chance_action);
        continue;
      }

      Player player = lnstate->current_player_id();
      bool use_nnue = (player == kXPlayerId) ? (eval_a != nullptr)
                                             : (eval_b != nullptr);
      Action action = kInvalidAction;
      if (use_nnue) {
        auto& search = (player == kXPlayerId) ? search_a : search_b;
        auto search_result = search.Search(lnstate);
        action = search_result.best_action;
      }
      if (action == kInvalidAction) {
        std::vector<Action> actions = lnstate->LegalActions();
        if (actions.empty()) {
          break;
        }
        action = SampleRandomAction(actions, &rng);
      }
      lnstate->ApplyAction(action);
      moves += 1;
    }

    if (lnstate->IsTerminal()) {
      std::vector<double> returns = lnstate->Returns();
      double ret_a = returns[kXPlayerId];
      double ret_b = returns[kOPlayerId];
      result.sum_return_a += ret_a;
      result.sum_return_b += ret_b;
      if (ret_a > ret_b) {
        result.wins_a += 1;
        if (ret_a >= 2.0) {
          result.mars_a += 1;
        }
      } else if (ret_b > ret_a) {
        result.wins_b += 1;
        if (ret_b >= 2.0) {
          result.mars_b += 1;
        }
      } else {
        result.draws += 1;
      }
    } else {
      result.draws += 1;
    }
    result.games += 1;

    if (games_done != nullptr && config.progress && config.report_every > 0) {
      int done = games_done->fetch_add(1) + 1;
      if (done % config.report_every == 0 || done == total_games) {
        double elapsed = 0.0;
        if (start_time != nullptr) {
          elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - *start_time)
                        .count();
        }
        double rate = elapsed > 0.0 ? done / elapsed : 0.0;
        int remaining = std::max(total_games - done, 0);
        double eta = rate > 0.0 ? remaining / rate : 0.0;
        if (cout_mutex != nullptr) {
          std::lock_guard<std::mutex> lock(*cout_mutex);
          double pct = 100.0 * static_cast<double>(done) / total_games;
          std::ostream& out = std::cerr;
          out << "[eval] " << done << "/" << total_games << " ("
              << std::fixed << std::setprecision(1) << pct
              << "%) " << std::setprecision(2) << rate
              << " games/s ETA " << std::setprecision(0) << eta
              << "s\n";
          out.flush();
        }
      }
    }
  }

  return result;
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::Action;
  using open_spiel::Player;
  using open_spiel::kInvalidAction;
  using open_spiel::long_narde::EvalConfig;
  using open_spiel::long_narde::EvalResult;
  using open_spiel::long_narde::ExpectiminimaxSearch;
  using open_spiel::long_narde::GetBoolArg;
  using open_spiel::long_narde::GetIntArg;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::GetUint64Arg;
  using open_spiel::long_narde::LabelForPath;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::SampleChanceOutcome;
  using open_spiel::long_narde::SampleRandomAction;
  using open_spiel::long_narde::SearchConfig;
  using open_spiel::long_narde::kOPlayerId;
  using open_spiel::long_narde::kXPlayerId;
  using open_spiel::long_narde::nnue::NnueEvaluator;
  using open_spiel::long_narde::nnue::NnueModel;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  EvalConfig config;
  config.games = GetIntArg(args, "games", config.games);
  config.depth = GetIntArg(args, "depth", config.depth);
  config.max_moves = GetIntArg(args, "max_moves", config.max_moves);
  config.report_every = GetIntArg(args, "report_every", config.report_every);
  config.seed = GetUint64Arg(args, "seed", config.seed);
  config.workers = GetIntArg(args, "workers", config.workers);
  config.nnue_a = GetStringArg(args, "nnue_a", "");
  config.nnue_b = GetStringArg(args, "nnue_b", "");
  config.out_path = GetStringArg(args, "out", "");
  config.progress = GetBoolArg(args, "progress", config.progress);

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("long_narde");

  NnueModel model_a;
  NnueModel model_b;
  bool use_a = !config.nnue_a.empty();
  bool use_b = !config.nnue_b.empty();
  if (use_a && !model_a.Load(config.nnue_a)) {
    std::cerr << "Failed to load NNUE A from " << config.nnue_a << "\n";
    return 1;
  }
  if (use_b && !model_b.Load(config.nnue_b)) {
    std::cerr << "Failed to load NNUE B from " << config.nnue_b << "\n";
    return 1;
  }

  NnueEvaluator eval_a(&model_a);
  NnueEvaluator eval_b(&model_b);

  SearchConfig search_config;
  search_config.max_depth = config.depth;

  EvalResult result;
  int total_games = std::max(config.games, 0);
  int workers = config.workers;
  if (workers <= 0) {
    unsigned int hc = std::thread::hardware_concurrency();
    workers = hc == 0 ? 1 : static_cast<int>(hc) + 1;
  }
  workers = std::min(workers, std::max(total_games, 1));

  std::vector<EvalResult> partials(workers);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  std::atomic<int> games_done{0};
  std::atomic<int> next_game{0};
  std::mutex cout_mutex;
  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < workers; ++i) {
    threads.emplace_back([&, i]() {
      partials[i] = RunEvalWorker(game, use_a ? &eval_a : nullptr,
                                  use_b ? &eval_b : nullptr, search_config,
                                  config, total_games, &next_game,
                                  &games_done, &start_time, &cout_mutex);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& part : partials) {
    result.games += part.games;
    result.wins_a += part.wins_a;
    result.wins_b += part.wins_b;
    result.draws += part.draws;
    result.mars_a += part.mars_a;
    result.mars_b += part.mars_b;
    result.sum_return_a += part.sum_return_a;
    result.sum_return_b += part.sum_return_b;
  }

  double avg_a = result.games > 0 ? result.sum_return_a / result.games : 0.0;
  double avg_b = result.games > 0 ? result.sum_return_b / result.games : 0.0;

  std::string label_a = LabelForPath(config.nnue_a, "random");
  std::string label_b = LabelForPath(config.nnue_b, "random");

  std::ostringstream line;
  line << "games=" << result.games << " depth=" << config.depth
       << " seed=" << config.seed << " agent_a=" << label_a
       << " agent_b=" << label_b << " wins_a=" << result.wins_a
       << " wins_b=" << result.wins_b << " draws=" << result.draws
       << " mars_a=" << result.mars_a << " mars_b=" << result.mars_b
       << " avg_return_a=" << std::fixed << std::setprecision(4) << avg_a
       << " avg_return_b=" << std::fixed << std::setprecision(4) << avg_b;

  std::cout << line.str() << "\n";

  if (!config.out_path.empty()) {
    std::ofstream out(config.out_path, std::ios::out | std::ios::app);
    if (!out.is_open()) {
      std::cerr << "Failed to open log file " << config.out_path << "\n";
      return 1;
    }
    out << line.str() << "\n";
  }

  return 0;
}
