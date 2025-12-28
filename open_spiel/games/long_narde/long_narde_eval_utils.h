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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_EVAL_UTILS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_EVAL_UTILS_H_

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace long_narde {

struct EvalConfig {
  int games = 1000;
  int depth = 3;
  int max_moves = 1000;
  int report_every = 100;
  uint64_t seed = 7;
  int workers = 0;
  int tt_entries = 200000;
  int chance_samples = 0;
  int chance_sample_depth = 1;
  uint64_t chance_seed = 0;
  std::string nnue_a;
  std::string nnue_b;
  std::string out_path;
  bool progress = true;
};

struct EvalResult {
  struct RoleStats {
    int games = 0;
    int wins_a = 0;
    int wins_b = 0;
    int mars_a = 0;
    int mars_b = 0;
    double sum_return_a = 0.0;
    double sum_return_b = 0.0;
  };

  int games = 0;
  int wins_a = 0;
  int wins_b = 0;
  int mars_a = 0;
  int mars_b = 0;
  double sum_return_a = 0.0;
  double sum_return_b = 0.0;
  RoleStats a_starts;
  RoleStats b_starts;
};

std::unordered_map<std::string, std::string> ParseArgs(int argc, char** argv);
int GetIntArg(const std::unordered_map<std::string, std::string>& args,
              const std::string& key, int default_value);
uint64_t GetUint64Arg(const std::unordered_map<std::string, std::string>& args,
                      const std::string& key, uint64_t default_value);
std::string GetStringArg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, const std::string& default_value);
bool GetBoolArg(const std::unordered_map<std::string, std::string>& args,
                const std::string& key, bool default_value);
void PrintUsage(const char* bin);
Action SampleChanceOutcome(
    const std::vector<std::pair<Action, double>>& outcomes,
    std::mt19937* rng);
Action SampleRandomAction(const std::vector<Action>& actions,
                          std::mt19937* rng);
std::string LabelForPath(const std::string& path, const std::string& fallback);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_EVAL_UTILS_H_
