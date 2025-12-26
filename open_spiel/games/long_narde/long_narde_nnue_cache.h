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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_CACHE_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_CACHE_H_

#include <array>
#include <memory>
#include <vector>

#include "open_spiel/games/long_narde/long_narde.h"
#include "open_spiel/games/long_narde/long_narde_nnue.h"

namespace open_spiel {
namespace long_narde {

class NnueCacheStack {
 public:
  NnueCacheStack() = default;
  void Reset(const LongNardeState& state, const nnue::NnueNetwork* network);
  void Clear();
  void PushChance();
  void PushAction(const LongNardeState& state, Action action,
                  const std::array<int, 2>& dice, Player prev_player);
  void Pop();
  const nnue::NnueCache* Current() const;

 private:
  const nnue::NnueNetwork* network_ = nullptr;
  std::vector<nnue::NnueCache> stack_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_CACHE_H_
