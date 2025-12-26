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

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_selfplay.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {
namespace {

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

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin << " [--out path] [--games N] [--seed N]\n"
            << "             [--shard_id N]\n"
            << "             [--depth N] [--temperature T] [--alpha A]\n"
            << "             [--workers N] [--chunk N] [--nnue path]\n"
            << "             [--progress 0|1] [--report_every N]"
            << " [--tt_entries N]\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::LnueShardConfig;
  using open_spiel::long_narde::RunSelfPlay;
  using open_spiel::long_narde::SearchConfig;
  using open_spiel::long_narde::SelfPlayConfig;
  using open_spiel::long_narde::WriteLnueShard;
  using open_spiel::long_narde::nnue::kNnueRunBlockThreshold;
  using open_spiel::long_narde::nnue::NnueEvaluator;
  using open_spiel::long_narde::nnue::NnueModel;

  auto args = open_spiel::long_narde::ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    open_spiel::long_narde::PrintUsage(argv[0]);
    return 0;
  }

  std::string out_path =
      open_spiel::long_narde::GetStringArg(args, "out", "selfplay.lnue");
  int games = open_spiel::long_narde::GetIntArg(args, "games", 10);
  uint64_t seed = open_spiel::long_narde::GetUint64Arg(args, "seed", 7);
  uint32_t shard_id = static_cast<uint32_t>(
      open_spiel::long_narde::GetIntArg(args, "shard_id", 0));
  int depth = open_spiel::long_narde::GetIntArg(args, "depth", 2);
  int workers = open_spiel::long_narde::GetIntArg(args, "workers", 0);
  int chunk = open_spiel::long_narde::GetIntArg(args, "chunk", 4096);
  int tt_entries =
      open_spiel::long_narde::GetIntArg(args, "tt_entries", 200000);
  double temperature =
      open_spiel::long_narde::GetDoubleArg(args, "temperature", 1.0);
  double alpha = open_spiel::long_narde::GetDoubleArg(args, "alpha", 0.5);
  int progress =
      open_spiel::long_narde::GetIntArg(args, "progress", 0);
  int report_every =
      open_spiel::long_narde::GetIntArg(args, "report_every", 100);
  std::string nnue_path =
      open_spiel::long_narde::GetStringArg(args, "nnue", "");

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("long_narde");

  NnueModel model;
  if (!nnue_path.empty()) {
    if (!model.Load(nnue_path)) {
      std::cerr << "Failed to load NNUE from " << nnue_path << "\n";
    }
  }
  NnueEvaluator evaluator(&model);

  SearchConfig search_config;
  search_config.max_depth = depth;
  search_config.max_tt_entries = tt_entries;

  SelfPlayConfig selfplay_config;
  selfplay_config.num_games = games;
  selfplay_config.num_workers = workers;
  selfplay_config.temperature = temperature;
  selfplay_config.alpha = alpha;
  selfplay_config.seed = seed;
  selfplay_config.progress = (progress != 0);
  selfplay_config.report_every = report_every;

  LnueShardConfig shard_config;
  shard_config.samples_per_chunk = chunk;
  shard_config.seed = seed;
  shard_config.shard_id = shard_id;
  shard_config.run_block_threshold = kNnueRunBlockThreshold;

  auto batch = RunSelfPlay(game, evaluator, search_config, selfplay_config);
  if (!WriteLnueShard(out_path, batch, shard_config)) {
    std::cerr << "Failed to write LNUE shard to " << out_path << "\n";
    return 1;
  }

  std::cout << "Wrote " << batch.samples.size() << " samples from "
            << batch.stats.games << " games to " << out_path << "\n";
  return 0;
}
