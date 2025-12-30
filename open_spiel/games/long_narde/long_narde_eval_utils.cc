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

#include "open_spiel/games/long_narde/long_narde_eval_utils.h"

#include <cctype>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

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
            << " [--report_every N] [--tt_entries N]\n"
            << "             [--root_full_depth_top_k N]"
            << " [--root_reduced_depth N]\n"
            << "             [--chance_samples N] [--chance_sample_depth N]"
            << " [--chance_seed N]\n";
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
  int best_iter = -1;
  std::size_t best_len = 0;
  std::size_t start = 0;
  while (start <= path.size()) {
    std::size_t end = path.find_first_of("/\\", start);
    if (end == std::string::npos) {
      end = path.size();
    }
    if (end > start) {
      std::string segment = path.substr(start, end - start);
      if (segment.rfind("iter_", 0) == 0 && segment.size() > 5) {
        bool all_digits = true;
        for (std::size_t i = 5; i < segment.size(); ++i) {
          if (!std::isdigit(static_cast<unsigned char>(segment[i]))) {
            all_digits = false;
            break;
          }
        }
        if (all_digits) {
          std::size_t len = segment.size() - 5;
          if (len > best_len) {
            best_len = len;
            best_iter = std::stoi(segment.substr(5));
          }
        }
      }
    }
    if (end == path.size()) {
      break;
    }
    start = end + 1;
  }
  if (best_iter >= 0) {
    return "nnue-" + std::to_string(best_iter);
  }
  std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos || pos + 1 >= path.size()) {
    return path;
  }
  return path.substr(pos + 1);
}

}  // namespace long_narde
}  // namespace open_spiel
