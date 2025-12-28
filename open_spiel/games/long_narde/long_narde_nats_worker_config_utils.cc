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

#include "open_spiel/games/long_narde/long_narde_nats_worker_config_utils.h"

#include <cctype>
#include <chrono>
#include <sstream>
#include <string>
#include <unordered_map>

#include "open_spiel/games/long_narde/long_narde_nats.h"
#include "open_spiel/games/long_narde/long_narde_nats_worker_args.h"

namespace open_spiel {
namespace long_narde {
namespace {

std::string Trim(const std::string& value) {
  std::size_t start = 0;
  while (start < value.size() &&
         std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  std::size_t end = value.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(start, end - start);
}

}  // namespace

bool HasArg(const std::unordered_map<std::string, std::string>& args,
            const std::string& key) {
  return args.find(key) != args.end();
}

std::unordered_map<std::string, std::string> ParseConfigPayload(
    const std::string& payload) {
  std::unordered_map<std::string, std::string> out;
  std::istringstream stream(payload);
  std::string line;
  while (std::getline(stream, line)) {
    std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    if (line.rfind("--", 0) == 0) {
      line = line.substr(2);
    }
    std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    std::string key = Trim(line.substr(0, eq));
    std::string value = Trim(line.substr(eq + 1));
    if (!key.empty()) {
      out[key] = value;
    }
  }
  return out;
}

void ApplyConfigArgs(const std::unordered_map<std::string, std::string>& args,
                     WorkerConfig* config) {
  if (config == nullptr) {
    return;
  }
  config->nats_url = GetStringArg(args, "nats", config->nats_url);
  config->run_id = GetStringArg(args, "run_id", config->run_id);
  config->traj_subject =
      GetStringArg(args, "traj_subject", config->traj_subject);
  config->weights_subject =
      GetStringArg(args, "weights_subject", config->weights_subject);
  config->request_subject =
      GetStringArg(args, "request_subject", config->request_subject);
  config->config_subject =
      GetStringArg(args, "config_subject", config->config_subject);
  config->config_request_subject = GetStringArg(
      args, "config_request_subject", config->config_request_subject);
  config->request_config =
      GetIntArg(args, "request_config", config->request_config ? 1 : 0) != 0;
  config->config_timeout_ms =
      GetIntArg(args, "config_timeout_ms", config->config_timeout_ms);
  config->nnue_path = GetStringArg(args, "nnue", config->nnue_path);
  config->depth = GetIntArg(args, "depth", config->depth);
  config->workers = GetIntArg(args, "workers", config->workers);
  config->temperature = GetDoubleArg(args, "temperature", config->temperature);
  config->alpha = GetDoubleArg(args, "alpha", config->alpha);
  config->games = GetInt64Arg(args, "games", config->games);
  config->seed = GetUint64Arg(args, "seed", config->seed);
  config->wait_for_weights =
      GetIntArg(args, "wait_for_weights",
                config->wait_for_weights ? 1 : 0) != 0;
  config->request_weights =
      GetIntArg(args, "request_weights",
                config->request_weights ? 1 : 0) != 0;
  config->request_interval_ms =
      GetIntArg(args, "request_interval_ms", config->request_interval_ms);
  config->report_every_seconds =
      GetIntArg(args, "report_every_seconds", config->report_every_seconds);
  config->tt_entries = GetIntArg(args, "tt_entries", config->tt_entries);
}

std::string BuildSubject(const std::string& base, const std::string& run_id) {
  if (run_id.empty()) {
    return base;
  }
  return base + "." + run_id;
}

bool FetchRemoteConfig(const WorkerConfig& config, std::string* payload_out) {
  if (payload_out == nullptr) {
    return false;
  }
  if (config.config_subject.empty()) {
    return false;
  }
  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    return false;
  }
  if (config.config_timeout_ms > 0) {
    conn.SetReceiveTimeout(config.config_timeout_ms);
  }
  std::string config_subject =
      BuildSubject(config.config_subject, config.run_id);
  if (!conn.Subscribe(config_subject, 1)) {
    return false;
  }
  if (config.request_config && !config.config_request_subject.empty()) {
    std::string inbox =
        BuildSubject("nnue.config.inbox." + std::to_string(config.seed) + "." +
                         std::to_string(
                             static_cast<int64_t>(
                                 std::chrono::steady_clock::now()
                                     .time_since_epoch()
                                     .count())),
                     config.run_id);
    if (!conn.Subscribe(inbox, 2)) {
      return false;
    }
    std::string request_subject =
        BuildSubject(config.config_request_subject, config.run_id);
    conn.Publish(request_subject, inbox);
  }
  NatsMessage msg;
  if (!conn.NextMessage(&msg)) {
    return false;
  }
  *payload_out = msg.payload;
  return !payload_out->empty();
}

}  // namespace long_narde
}  // namespace open_spiel
