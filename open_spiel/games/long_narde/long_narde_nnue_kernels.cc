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

#include "open_spiel/games/long_narde/long_narde_nnue_kernels.h"

#include "open_spiel/games/long_narde/long_narde_nnue.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace open_spiel {
namespace long_narde {
namespace nnue {
namespace {

void AddRowScalar(const int16_t* weights, int16_t* acc) {
  for (int i = 0; i < kNnueL1; ++i) {
    acc[i] += weights[i];
  }
}

void SubRowScalar(const int16_t* weights, int16_t* acc) {
  for (int i = 0; i < kNnueL1; ++i) {
    acc[i] -= weights[i];
  }
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void AddRowAvx2(const int16_t* weights, int16_t* acc) {
  for (int i = 0; i < kNnueL1; i += 16) {
    __m256i acc_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i wei_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
    acc_v = _mm256_add_epi16(acc_v, wei_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), acc_v);
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl")))
#endif
void AddRowAvx512(const int16_t* weights, int16_t* acc) {
#if defined(__AVX512F__)
  for (int i = 0; i < kNnueL1; i += 32) {
    __m512i acc_v = _mm512_loadu_si512(reinterpret_cast<const void*>(acc + i));
    __m512i wei_v =
        _mm512_loadu_si512(reinterpret_cast<const void*>(weights + i));
    acc_v = _mm512_add_epi16(acc_v, wei_v);
    _mm512_storeu_si512(reinterpret_cast<void*>(acc + i), acc_v);
  }
#else
  AddRowAvx2(weights, acc);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void SubRowAvx2(const int16_t* weights, int16_t* acc) {
  for (int i = 0; i < kNnueL1; i += 16) {
    __m256i acc_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i wei_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
    acc_v = _mm256_sub_epi16(acc_v, wei_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), acc_v);
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl")))
#endif
void SubRowAvx512(const int16_t* weights, int16_t* acc) {
#if defined(__AVX512F__)
  for (int i = 0; i < kNnueL1; i += 32) {
    __m512i acc_v = _mm512_loadu_si512(reinterpret_cast<const void*>(acc + i));
    __m512i wei_v =
        _mm512_loadu_si512(reinterpret_cast<const void*>(weights + i));
    acc_v = _mm512_sub_epi16(acc_v, wei_v);
    _mm512_storeu_si512(reinterpret_cast<void*>(acc + i), acc_v);
  }
#else
  SubRowAvx2(weights, acc);
#endif
}
#endif  // __x86_64__ || _M_X64

void ComputeLayerScalar(const int8_t* input, const int8_t* weights,
                        const int8_t* bias, int in_dim, int out_dim,
                        int8_t* output) {
  for (int o = 0; o < out_dim; ++o) {
    int32_t sum = bias[o] * kNnueFactor;
    const int8_t* row = weights + o * in_dim;
    for (int i = 0; i < in_dim; ++i) {
      sum += static_cast<int32_t>(input[i]) * row[i];
    }
    output[o] = ClampLayer(sum);
  }
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void ComputeLayerAvx2(const int8_t* input, const int8_t* weights,
                      const int8_t* bias, int in_dim, int out_dim,
                      int8_t* output) {
  const __m256i one = _mm256_set1_epi16(1);
  for (int o = 0; o < out_dim; ++o) {
    __m256i sum = _mm256_setzero_si256();
    const int8_t* row = weights + o * in_dim;
    for (int i = 0; i < in_dim; i += 32) {
      __m256i inp =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      __m256i wei =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
      __m256i dot = _mm256_madd_epi16(_mm256_maddubs_epi16(inp, wei), one);
      sum = _mm256_add_epi32(sum, dot);
    }
    __m128i sum_lo = _mm256_castsi256_si128(sum);
    __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
    sum128 = _mm_add_epi32(sum128,
                           _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum128 = _mm_add_epi32(sum128,
                           _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t total = _mm_cvtsi128_si32(sum128);
    total += bias[o] * kNnueFactor;
    output[o] = ClampLayer(total);
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,avxvnni")))
#endif
void ComputeLayerAvxVnni(const int8_t* input, const int8_t* weights,
                         const int8_t* bias, int in_dim, int out_dim,
                         int8_t* output) {
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
  for (int o = 0; o < out_dim; ++o) {
    __m256i sum = _mm256_setzero_si256();
    const int8_t* row = weights + o * in_dim;
    for (int i = 0; i < in_dim; i += 32) {
      __m256i inp =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      __m256i wei =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
      sum = _mm256_dpbusd_epi32(sum, inp, wei);
    }
    __m128i sum_lo = _mm256_castsi256_si128(sum);
    __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
    sum128 = _mm_add_epi32(sum128,
                           _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    sum128 = _mm_add_epi32(sum128,
                           _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t total = _mm_cvtsi128_si32(sum128);
    total += bias[o] * kNnueFactor;
    output[o] = ClampLayer(total);
  }
#else
  ComputeLayerAvx2(input, weights, bias, in_dim, out_dim, output);
#endif
}
#endif  // __x86_64__ || _M_X64

}  // namespace

AddRowFn GetAddRowKernel() {
  static AddRowFn fn = &AddRowScalar;
  static bool initialized = false;
  if (initialized) {
    return fn;
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx512f")) {
    fn = &AddRowAvx512;
  } else if (__builtin_cpu_supports("avx2")) {
    fn = &AddRowAvx2;
  }
#endif
#endif
  initialized = true;
  return fn;
}

SubRowFn GetSubRowKernel() {
  static SubRowFn fn = &SubRowScalar;
  static bool initialized = false;
  if (initialized) {
    return fn;
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx512f")) {
    fn = &SubRowAvx512;
  } else if (__builtin_cpu_supports("avx2")) {
    fn = &SubRowAvx2;
  }
#endif
#endif
  initialized = true;
  return fn;
}

ComputeLayerFn GetComputeLayerKernel() {
  static ComputeLayerFn fn = &ComputeLayerScalar;
  static bool initialized = false;
  if (initialized) {
    return fn;
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avxvnni")) {
    fn = &ComputeLayerAvxVnni;
  } else if (__builtin_cpu_supports("avx2")) {
    fn = &ComputeLayerAvx2;
  }
#endif
#endif
  initialized = true;
  return fn;
}

const char* NnueKernelName() {
  static const char* name = nullptr;
  if (name != nullptr) {
    return name;
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avxvnni")) {
    name = "avxvnni";
  } else if (__builtin_cpu_supports("avx512f")) {
    name = "avx512";
  } else if (__builtin_cpu_supports("avx2")) {
    name = "avx2";
  } else {
    name = "scalar";
  }
#else
  name = "scalar";
#endif
#else
  name = "scalar";
#endif
  return name;
}

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel
