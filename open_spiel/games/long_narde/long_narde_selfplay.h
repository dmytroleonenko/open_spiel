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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_search.h"

namespace open_spiel {
namespace long_narde {

class LnueStreamWriter;

struct SelfPlayConfig {
  int num_games = 1;
  int max_moves = 1000;
  int num_workers = 0;
  double temperature = 1.0;
  double temperature_end = -1.0;
  int temperature_decay_plies = 0;
  double alpha = 0.5;
  uint64_t seed = 0;
  bool progress = false;
  int report_every = 100;
};

struct SelfPlaySample {
  std::vector<int> active_features;
  double search_value = 0.0;
  double outcome_value = 0.0;
  double target_value = 0.0;
  uint64_t game_id = 0;
  uint16_t ply = 0;
  Player player = kInvalidPlayer;
};

struct SelfPlayStats {
  int games = 0;
  int total_moves = 0;
};

struct SelfPlayBatch {
  std::vector<SelfPlaySample> samples;
  SelfPlayStats stats;
};

struct SelfPlayStreamConfig {
  int64_t start_game = 0;
  int64_t start_samples = 0;
  int flush_every_games = 1;
  std::function<void(int64_t, int64_t)> on_flush;
};

SelfPlayBatch RunSelfPlay(std::shared_ptr<const Game> game,
                          const nnue::NnueEvaluator& evaluator,
                          const SearchConfig& search_config,
                          const SelfPlayConfig& selfplay_config);
SelfPlayStats RunSelfPlayStreaming(std::shared_ptr<const Game> game,
                                   const nnue::NnueEvaluator& evaluator,
                                   const SearchConfig& search_config,
                                   const SelfPlayConfig& selfplay_config,
                                   LnueStreamWriter* writer,
                                   const SelfPlayStreamConfig& stream_config);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_SELFPLAY_H_
