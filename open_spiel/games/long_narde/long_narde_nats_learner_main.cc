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
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nats.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace long_narde {
namespace {

std::string NextShardStatePath(const std::string& out_dir) {
  return out_dir + "/next_shard.txt";
}

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

int LoadNextShardId(const std::string& out_dir, bool* has_state) {
  if (has_state != nullptr) {
    *has_state = false;
  }
  const std::string path = NextShardStatePath(out_dir);
  if (!open_spiel::file::Exists(path)) {
    return 0;
  }
  if (has_state != nullptr) {
    *has_state = true;
  }
  std::string contents = open_spiel::file::ReadContentsFromFile(path, "r");
  std::istringstream iss(contents);
  int next_id = 0;
  if (!(iss >> next_id) || next_id < 0) {
    return 0;
  }
  return next_id;
}

void SaveNextShardId(const std::string& out_dir, int next_id) {
  const std::string path = NextShardStatePath(out_dir);
  open_spiel::file::WriteContentsToFile(path, "w",
                                        std::to_string(next_id) + "\n");
}

struct LearnerConfig {
  std::string nats_url = "nats://127.0.0.1:4222";
  std::string run_id = "default";
  std::string traj_subject = "lnue.traj";
  std::string out_dir = "results/nnue_stream";
  int games_per_shard = 1000;
  int samples_per_chunk = 4096;
  int report_every = 100;
  int max_games = 0;
  bool progress = true;
  uint64_t seed = 0;
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

uint64_t GetUint64Arg(const std::unordered_map<std::string, std::string>& args,
                      const std::string& key, uint64_t default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return static_cast<uint64_t>(std::stoull(it->second));
}

bool GetBoolArg(const std::unordered_map<std::string, std::string>& args,
                const std::string& key, bool default_value) {
  auto it = args.find(key);
  if (it == args.end()) {
    return default_value;
  }
  if (it->second.empty()) {
    return true;
  }
  std::string value = it->second;
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);
  return value == "1" || value == "true" || value == "yes";
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

std::string ShardPath(const std::string& out_dir, int shard_id) {
  std::ostringstream oss;
  oss << out_dir << "/shard_" << std::setw(4) << std::setfill('0')
      << shard_id << ".lnue";
  return oss.str();
}

bool OpenShard(const LearnerConfig& config, int shard_id,
               LnueStreamWriter* writer) {
  LnueShardConfig shard_config;
  shard_config.samples_per_chunk = config.samples_per_chunk;
  shard_config.write_tmp = true;
  shard_config.seed = config.seed;
  shard_config.shard_id = static_cast<uint32_t>(shard_id);
  shard_config.run_block_threshold = nnue::kNnueRunBlockThreshold;
  return writer->Open(ShardPath(config.out_dir, shard_id), shard_config);
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin
            << " [--nats url] [--run_id id] [--traj_subject name]"
            << " [--out_dir path]\n"
            << "             [--games_per_shard N] [--samples_per_chunk N]"
            << " [--report_every N] [--max_games N] [--seed N]"
            << " [--progress 0|1]\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::BuildSubject;
  using open_spiel::long_narde::DeserializeLnueTrajectory;
  using open_spiel::long_narde::GetBoolArg;
  using open_spiel::long_narde::GetIntArg;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::GetUint64Arg;
  using open_spiel::long_narde::LearnerConfig;
  using open_spiel::long_narde::LnueStreamWriter;
  using open_spiel::long_narde::LoadNextShardId;
  using open_spiel::long_narde::NatsConnection;
  using open_spiel::long_narde::NatsMessage;
  using open_spiel::long_narde::OpenShard;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::SaveNextShardId;
  using open_spiel::long_narde::SelfPlaySample;
  using open_spiel::long_narde::Timestamp;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  LearnerConfig config;
  config.nats_url = GetStringArg(args, "nats", config.nats_url);
  config.run_id = GetStringArg(args, "run_id", config.run_id);
  config.traj_subject = GetStringArg(args, "traj_subject", config.traj_subject);
  config.out_dir = GetStringArg(args, "out_dir", config.out_dir);
  config.games_per_shard = GetIntArg(args, "games_per_shard",
                                    config.games_per_shard);
  config.samples_per_chunk = GetIntArg(args, "samples_per_chunk",
                                      config.samples_per_chunk);
  config.report_every = GetIntArg(args, "report_every", config.report_every);
  config.max_games = GetIntArg(args, "max_games", config.max_games);
  config.seed = GetUint64Arg(args, "seed", config.seed);
  config.progress = GetBoolArg(args, "progress", config.progress);

  if (!open_spiel::file::Mkdirs(config.out_dir)) {
    std::cerr << "[" << Timestamp()
              << "] [learner] failed to create output dir " << config.out_dir
              << "\n";
    return 1;
  }

  bool has_state = false;
  int shard_id = LoadNextShardId(config.out_dir, &has_state);
  if (has_state) {
    std::cerr << "[" << Timestamp() << "] [learner] resuming at shard "
              << shard_id << "\n";
  }

  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    std::cerr << "[" << Timestamp()
              << "] [learner] failed to connect to NATS at " << config.nats_url
              << "\n";
    return 1;
  }

  std::string subject = BuildSubject(config.traj_subject, config.run_id);
  if (!conn.Subscribe(subject, 1)) {
    std::cerr << "[" << Timestamp() << "] [learner] failed to subscribe to "
              << subject << "\n";
    return 1;
  }

  LnueStreamWriter writer;
  int games_in_shard = 0;
  int total_games = 0;
  int total_samples = 0;
  int last_report_games = 0;
  int last_report_samples = 0;
  auto last_report_time = std::chrono::steady_clock::now();

  if (!OpenShard(config, shard_id, &writer)) {
    std::cerr << "[" << Timestamp() << "] [learner] failed to open shard "
              << shard_id << "\n";
    return 1;
  }
  SaveNextShardId(config.out_dir, shard_id + 1);

  NatsMessage msg;
  while (conn.NextMessage(&msg)) {
    std::vector<SelfPlaySample> samples;
    if (!DeserializeLnueTrajectory(msg.payload, &samples)) {
      std::cerr << "[" << Timestamp()
                << "] [learner] failed to parse trajectory payload.\n";
      continue;
    }
    if (!writer.AddSamples(samples)) {
      std::cerr << "[" << Timestamp()
                << "] [learner] failed to append samples to shard.\n";
      return 1;
    }
    games_in_shard += 1;
    total_games += 1;
    total_samples += static_cast<int>(samples.size());

    if (config.progress && config.report_every > 0 &&
        total_games % config.report_every == 0) {
      auto now = std::chrono::steady_clock::now();
      double window_elapsed =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              now - last_report_time)
              .count();
      int window_games = total_games - last_report_games;
      int window_samples = total_samples - last_report_samples;
      double window_games_s =
          window_elapsed > 0.0 ? window_games / window_elapsed : 0.0;
      double window_samples_s =
          window_elapsed > 0.0 ? window_samples / window_elapsed : 0.0;
      double avg_samples =
          total_games > 0 ? static_cast<double>(total_samples) / total_games
                          : 0.0;
      std::cerr << "[" << Timestamp() << "] [learner] games=" << total_games
                << " samples=" << total_samples << " shard=" << shard_id
                << " (" << games_in_shard << "/" << config.games_per_shard
                << ") rate=" << window_games_s << " g/s " << window_samples_s
                << " s/s window_s=" << window_elapsed << " avg=" << avg_samples
                << " s/g\n";
      last_report_time = now;
      last_report_games = total_games;
      last_report_samples = total_samples;
    }

    if (config.max_games > 0 && total_games >= config.max_games) {
      break;
    }

    if (games_in_shard >= std::max(1, config.games_per_shard)) {
      writer.Close();
      shard_id += 1;
      games_in_shard = 0;
      if (!OpenShard(config, shard_id, &writer)) {
        std::cerr << "[" << Timestamp() << "] [learner] failed to open shard "
                  << shard_id << "\n";
        return 1;
      }
      SaveNextShardId(config.out_dir, shard_id + 1);
    }
  }

  writer.Close();
  return 0;
}
