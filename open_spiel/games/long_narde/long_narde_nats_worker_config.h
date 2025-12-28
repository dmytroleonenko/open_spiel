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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_H_

#include <cstdint>
#include <string>

namespace open_spiel {
namespace long_narde {

struct WorkerConfig {
  std::string nats_url = "nats://127.0.0.1:4222";
  std::string run_id = "default";
  std::string traj_subject = "lnue.traj";
  std::string weights_subject = "nnue.weights";
  std::string request_subject = "nnue.request";
  std::string config_subject = "nnue.config";
  std::string config_request_subject = "nnue.config.request";
  bool request_config = true;
  int config_timeout_ms = 2000;
  std::string nnue_path;
  int depth = 3;
  int max_moves = 1000;
  int workers = 0;
  double temperature = 1.0;
  double alpha = 0.5;
  int64_t games = 0;
  uint64_t seed = 7;
  int tt_entries = 200000;
  int chance_samples = 0;
  int chance_sample_depth = 1;
  uint64_t chance_seed = 0;
  bool wait_for_weights = false;
  bool request_weights = false;
  int request_interval_ms = 1000;
  int report_every_seconds = 60;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_CONFIG_H_
