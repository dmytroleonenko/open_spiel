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

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_selfplay.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/games/long_narde/long_narde_search.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

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
            << "             [--depth N] [--temperature T]"
            << " [--temperature_end T] [--temperature_decay_plies N]\n"
            << "             [--alpha A]\n"
            << "             [--workers N] [--chunk N] [--nnue path]\n"
            << "             [--progress 0|1] [--report_every N]"
            << " [--tt_entries N]\n"
            << "             [--root_full_depth_top_k N]"
            << " [--root_reduced_depth N]\n"
            << "             [--stream 0|1] [--resume 0|1]"
            << " [--flush_every_games N] [--lock path]"
            << " [--force_lock 0|1]\n"
            << "             [--chance_samples N] [--chance_sample_depth N]"
            << " [--chance_seed N]\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

namespace {

struct LockInfo {
  uint64_t seed = 0;
  uint32_t shard_id = 0;
  int64_t target_games = 0;
  int64_t games_done = 0;
  int64_t samples_done = 0;
  int flush_every_games = 1;
  int pid = 0;
  int64_t started_secs = 0;
  int64_t updated_secs = 0;
};

int64_t NowSeconds() {
  auto now = std::chrono::system_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::seconds>(
      now.time_since_epoch());
  return static_cast<int64_t>(secs.count());
}

bool WriteLockFile(const std::string& path, const LockInfo& info) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  out << "pid=" << info.pid << "\n";
  out << "seed=" << info.seed << "\n";
  out << "shard_id=" << info.shard_id << "\n";
  out << "target_games=" << info.target_games << "\n";
  out << "games_done=" << info.games_done << "\n";
  out << "samples_done=" << info.samples_done << "\n";
  out << "flush_every_games=" << info.flush_every_games << "\n";
  out << "started_secs=" << info.started_secs << "\n";
  out << "updated_secs=" << info.updated_secs << "\n";
  return out.good();
}

int ReadLockPid(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return -1;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("pid=", 0) == 0) {
      return std::stoi(line.substr(4));
    }
  }
  return -1;
}

bool IsProcessAlive(int pid) {
#ifdef _WIN32
  (void)pid;
  return false;
#else
  if (pid <= 0) {
    return false;
  }
  int result = kill(pid, 0);
  if (result == 0) {
    return true;
  }
  return errno == EPERM;
#endif
}

bool CreateLockFileExclusive(const std::string& path) {
#ifdef _WIN32
  if (open_spiel::file::Exists(path)) {
    return false;
  }
  std::ofstream out(path, std::ios::out);
  return out.is_open();
#else
  int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd < 0) {
    return false;
  }
  close(fd);
  return true;
#endif
}

bool AcquireLock(const std::string& path, bool resume, bool force,
                 std::string* error) {
  if (CreateLockFileExclusive(path)) {
    return true;
  }
  if (!resume && !force) {
    if (error != nullptr) {
      *error = "Lock file exists: " + path;
    }
    return false;
  }
  int pid = ReadLockPid(path);
  if (pid > 0 && IsProcessAlive(pid) && !force) {
    if (error != nullptr) {
      *error = "Lock file held by active pid: " + std::to_string(pid);
    }
    return false;
  }
  if (!open_spiel::file::Remove(path)) {
    if (error != nullptr) {
      *error = "Failed to remove stale lock: " + path;
    }
    return false;
  }
  if (!CreateLockFileExclusive(path)) {
    if (error != nullptr) {
      *error = "Failed to create lock file: " + path;
    }
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  using open_spiel::long_narde::LnueShardConfig;
  using open_spiel::long_narde::LnueResumeState;
  using open_spiel::long_narde::LnueStreamWriter;
  using open_spiel::long_narde::RunSelfPlay;
  using open_spiel::long_narde::RunSelfPlayStreaming;
  using open_spiel::long_narde::SearchConfig;
  using open_spiel::long_narde::SelfPlayConfig;
  using open_spiel::long_narde::SelfPlayStats;
  using open_spiel::long_narde::SelfPlayStreamConfig;
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
  int depth = open_spiel::long_narde::GetIntArg(args, "depth", 3);
  int workers = open_spiel::long_narde::GetIntArg(args, "workers", 0);
  int chunk = open_spiel::long_narde::GetIntArg(args, "chunk", 4096);
  int stream = open_spiel::long_narde::GetIntArg(args, "stream", 0);
  int resume = open_spiel::long_narde::GetIntArg(args, "resume", 0);
  int flush_every_games =
      open_spiel::long_narde::GetIntArg(args, "flush_every_games", 1);
  int force_lock =
      open_spiel::long_narde::GetIntArg(args, "force_lock", 0);
  std::string lock_path =
      open_spiel::long_narde::GetStringArg(args, "lock", "");
  int tt_entries =
      open_spiel::long_narde::GetIntArg(args, "tt_entries", 200000);
  int root_full_depth_top_k =
      open_spiel::long_narde::GetIntArg(args, "root_full_depth_top_k", 0);
  int root_reduced_depth =
      open_spiel::long_narde::GetIntArg(args, "root_reduced_depth", -1);
  int chance_samples =
      open_spiel::long_narde::GetIntArg(args, "chance_samples", 0);
  int chance_sample_depth =
      open_spiel::long_narde::GetIntArg(args, "chance_sample_depth", 1);
  uint64_t chance_seed =
      open_spiel::long_narde::GetUint64Arg(args, "chance_seed", 0);
  double temperature =
      open_spiel::long_narde::GetDoubleArg(args, "temperature", 1.0);
  double temperature_end =
      open_spiel::long_narde::GetDoubleArg(args, "temperature_end", -1.0);
  int temperature_decay_plies = open_spiel::long_narde::GetIntArg(
      args, "temperature_decay_plies", 0);
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
  search_config.root_full_depth_top_k = root_full_depth_top_k;
  search_config.root_reduced_depth = root_reduced_depth;
  search_config.chance_samples = chance_samples;
  search_config.chance_sample_depth = chance_sample_depth;
  search_config.chance_seed = chance_seed;

  SelfPlayConfig selfplay_config;
  selfplay_config.num_games = games;
  selfplay_config.num_workers = workers;
  selfplay_config.temperature = temperature;
  selfplay_config.temperature_end = temperature_end;
  selfplay_config.temperature_decay_plies = temperature_decay_plies;
  selfplay_config.alpha = alpha;
  selfplay_config.seed = seed;
  selfplay_config.progress = (progress != 0);
  selfplay_config.report_every = report_every;

  LnueShardConfig shard_config;
  shard_config.samples_per_chunk = chunk;
  shard_config.write_tmp = true;
  shard_config.seed = seed;
  shard_config.shard_id = shard_id;
  shard_config.run_block_threshold = kNnueRunBlockThreshold;

  bool use_stream = (stream != 0) || (resume != 0);
  if (use_stream) {
    shard_config.write_tmp = false;
    if (lock_path.empty()) {
      lock_path = out_path + ".lock";
    }
    std::string lock_error;
    if (!AcquireLock(lock_path, resume != 0, force_lock != 0, &lock_error)) {
      std::cerr << lock_error << "\n";
      return 1;
    }
    LnueStreamWriter writer;
    LnueResumeState resume_state{};
    int64_t start_game = 0;
    int64_t start_samples = 0;
    bool has_existing = open_spiel::file::Exists(out_path);
    if (resume != 0 && has_existing) {
      if (!writer.OpenAppend(out_path, shard_config, &resume_state)) {
        std::cerr << "Failed to resume LNUE shard from " << out_path << "\n";
        open_spiel::file::Remove(lock_path);
        return 1;
      }
      start_game = static_cast<int64_t>(resume_state.games_done);
      start_samples = static_cast<int64_t>(resume_state.samples_done);
      std::cerr << "Resuming shard from " << start_game << " games.\n";
    } else {
      if (!writer.Open(out_path, shard_config)) {
        std::cerr << "Failed to open LNUE shard " << out_path << "\n";
        open_spiel::file::Remove(lock_path);
        return 1;
      }
    }
    if (start_game >= games) {
      writer.Close();
      open_spiel::file::Remove(lock_path);
      std::cerr << "Shard already has " << start_game
                << " games; nothing to do.\n";
      return 0;
    }
    if (flush_every_games <= 0) {
      flush_every_games = 1;
    }
    LockInfo lock_info;
    lock_info.seed = seed;
    lock_info.shard_id = shard_id;
    lock_info.target_games = games;
    lock_info.games_done = start_game;
    lock_info.samples_done = start_samples;
    lock_info.flush_every_games = flush_every_games;
#ifdef _WIN32
    lock_info.pid = 0;
#else
    lock_info.pid = static_cast<int>(getpid());
#endif
    lock_info.started_secs = NowSeconds();
    lock_info.updated_secs = lock_info.started_secs;
    WriteLockFile(lock_path, lock_info);

    SelfPlayStreamConfig stream_config;
    stream_config.start_game = start_game;
    stream_config.start_samples = start_samples;
    stream_config.flush_every_games = flush_every_games;
    stream_config.on_flush = [&](int64_t games_done,
                                 int64_t samples_done) {
      lock_info.games_done = games_done;
      lock_info.samples_done = samples_done;
      lock_info.updated_secs = NowSeconds();
      WriteLockFile(lock_path, lock_info);
    };

    SelfPlayStats stats =
        RunSelfPlayStreaming(game, evaluator, search_config, selfplay_config,
                             &writer, stream_config);
    writer.Close();
    open_spiel::file::Remove(lock_path);
    int64_t total_games = start_game + stats.games;
    std::cout << "Wrote " << lock_info.samples_done << " samples from "
              << total_games << " games to " << out_path << "\n";
    return 0;
  }

  auto batch = RunSelfPlay(game, evaluator, search_config, selfplay_config);
  if (!WriteLnueShard(out_path, batch, shard_config)) {
    std::cerr << "Failed to write LNUE shard to " << out_path << "\n";
    return 1;
  }

  std::cout << "Wrote " << batch.samples.size() << " samples from "
            << batch.stats.games << " games to " << out_path << "\n";
  return 0;
}
