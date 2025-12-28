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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_UTILS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_UTILS_H_

#include <string>
#include <unordered_map>

#include "open_spiel/games/long_narde/long_narde_nats_worker_config.h"

namespace open_spiel {
namespace long_narde {

bool HasArg(const std::unordered_map<std::string, std::string>& args,
            const std::string& key);

std::unordered_map<std::string, std::string> ParseConfigPayload(
    const std::string& payload);

void ApplyConfigArgs(const std::unordered_map<std::string, std::string>& args,
                     WorkerConfig* config);

bool FetchRemoteConfig(const WorkerConfig& config, std::string* payload_out);

std::string BuildSubject(const std::string& base, const std::string& run_id);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_UTILS_H_
