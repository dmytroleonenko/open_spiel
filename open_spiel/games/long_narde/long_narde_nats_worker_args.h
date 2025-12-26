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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_ARGS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_ARGS_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace open_spiel {
namespace long_narde {

std::unordered_map<std::string, std::string> ParseArgs(int argc, char** argv);

std::string GetStringArg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, const std::string& default_value);

int GetIntArg(const std::unordered_map<std::string, std::string>& args,
              const std::string& key, int default_value);

int64_t GetInt64Arg(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key, int64_t default_value);

uint64_t GetUint64Arg(const std::unordered_map<std::string, std::string>& args,
                      const std::string& key, uint64_t default_value);

double GetDoubleArg(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key, double default_value);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_ARGS_H_
