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
  std::size_t iter_pos = path.rfind("iter_");
  if (iter_pos != std::string::npos) {
    iter_pos += 5;
    std::size_t end = iter_pos;
    while (end < path.size() &&
           std::isdigit(static_cast<unsigned char>(path[end]))) {
      end += 1;
    }
    if (end > iter_pos) {
      int iter = std::stoi(path.substr(iter_pos, end - iter_pos));
      return "nnue-" + std::to_string(iter);
    }
  }
  std::size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos || pos + 1 >= path.size()) {
    return path;
  }
  return path.substr(pos + 1);
}

}  // namespace long_narde
}  // namespace open_spiel
