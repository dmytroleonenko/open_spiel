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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_KERNELS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_KERNELS_H_

#include <cstdint>

namespace open_spiel {
namespace long_narde {
namespace nnue {

constexpr int kNnueFactor = 64;

using AddRowFn = void (*)(const int16_t* weights, int16_t* acc);
using SubRowFn = void (*)(const int16_t* weights, int16_t* acc);
using ComputeLayerFn = void (*)(const int8_t* input, const int8_t* weights,
                                const int8_t* bias, int in_dim, int out_dim,
                                int8_t* output);

inline int8_t ClampAcc(int16_t value) {
  if (value < 0) {
    return 0;
  }
  if (value > 127) {
    return 127;
  }
  return static_cast<int8_t>(value);
}

inline int8_t ClampLayer(int32_t value) {
  int32_t scaled = value / kNnueFactor;
  if (scaled < 0) {
    return 0;
  }
  if (scaled > 127) {
    return 127;
  }
  return static_cast<int8_t>(scaled);
}

AddRowFn GetAddRowKernel();
SubRowFn GetSubRowKernel();
ComputeLayerFn GetComputeLayerKernel();
const char* NnueKernelName();

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NNUE_KERNELS_H_
