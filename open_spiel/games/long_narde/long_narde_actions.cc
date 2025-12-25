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

#include "open_spiel/games/long_narde/long_narde_internal.h"

namespace open_spiel {
namespace long_narde {
namespace internal {

DecodedAction DecodeAction(Action action) {
  DecodedAction out;
  out.order = (action >= kOrderActionCount) ? 1 : 0;
  int rem = action % kOrderActionCount;
  out.src1 = rem % 25;
  out.src2 = rem / 25;
  return out;
}

}  // namespace internal
}  // namespace long_narde
}  // namespace open_spiel
