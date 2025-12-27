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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_PLAY_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_PLAY_H_

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nats_worker_config.h"

namespace open_spiel {
class Game;

namespace long_narde {

class ExpectiminimaxSearch;
struct SelfPlaySample;

std::vector<SelfPlaySample> PlayOneGame(std::shared_ptr<const Game> game,
                                        ExpectiminimaxSearch* search,
                                        const WorkerConfig& config,
                                        uint64_t game_id,
                                        std::mt19937* rng);

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_WORKER_PLAY_H_
