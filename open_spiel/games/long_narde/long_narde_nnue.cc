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

// Portions of the NNUE inference layout are adapted from the Cerebrum project
// (MIT License, Copyright 2020-2025 David Carteau).

#include "open_spiel/games/long_narde/long_narde_nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace open_spiel {
namespace long_narde {
namespace nnue {
namespace {

constexpr int kNnueFactor = 64;
constexpr int kMaxActiveFeatures =
(kNnuePointsWithOff * 2) + (kNnueRunFeaturesPerSide * 2);
constexpr int kSigmoidTableSize = 4096;
constexpr float kSigmoidMin = -8.0f;
constexpr float kSigmoidMax = 8.0f;

struct ActiveFeatures {
  std::array<int, kMaxActiveFeatures> indices{};
  int count = 0;
};

using AddRowFn = void (*)(const int16_t* weights, int16_t* acc);
using ComputeLayerFn = void (*)(const int8_t* input, const int8_t* weights,
                                const int8_t* bias, int in_dim, int out_dim,
                                int8_t* output);

void AddRowScalar(const int16_t* weights, int16_t* acc) {
  for (int i = 0; i < kNnueL1; ++i) {
    acc[i] += weights[i];
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
#endif  // __x86_64__ || _M_X64

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
#endif

ComputeLayerFn GetComputeLayerKernel() {
  static ComputeLayerFn fn = &ComputeLayerScalar;
  static bool initialized = false;
  if (initialized) {
    return fn;
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx2")) {
    fn = &ComputeLayerAvx2;
  }
#endif
#endif
  initialized = true;
  return fn;
}

int8_t ClampAcc(int16_t value) {
  if (value < 0) {
    return 0;
  }
  if (value > 127) {
    return 127;
  }
  return static_cast<int8_t>(value);
}

int8_t ClampLayer(int32_t value) {
  int32_t scaled = value / kNnueFactor;
  if (scaled < 0) {
    return 0;
  }
  if (scaled > 127) {
    return 127;
  }
  return static_cast<int8_t>(scaled);
}

bool ReadExact(std::ifstream* file, void* dst, std::size_t size) {
  file->read(reinterpret_cast<char*>(dst), size);
  return file->good() && file->gcount() == static_cast<std::streamsize>(size);
}

bool ReadHeader(std::ifstream* file, NnueFileHeader* header) {
  if (!ReadExact(file, header, sizeof(*header))) {
    return false;
  }
  if (header->magic[0] != 'L' || header->magic[1] != 'N' ||
      header->magic[2] != 'N' || header->magic[3] != 'U') {
    return false;
  }
  if (header->version != kNnueFileVersion) {
    return false;
  }
  if (header->endian != kNnueEndianMarker) {
    return false;
  }
  if (header->feature_dim != kNnueFeatureDim || header->l1 != kNnueL1 ||
      header->l2 != kNnueL2 || header->l3 != kNnueL3) {
    return false;
  }
  return true;
}

bool LoadHeaderPayload(std::ifstream* file, const NnueFileHeader& header,
                       NnueNetwork* net) {
  if (header.run_features_enabled == 0 || header.run_block_threshold != 1) {
    return false;
  }
  if (header.w0_type != kNnueQuantInt16 || header.w1_type != kNnueQuantInt8 ||
      header.w2_type != kNnueQuantInt8 || header.b0_type != kNnueQuantInt16 ||
      header.b1_type != kNnueQuantInt8 || header.b2_type != kNnueQuantInt8) {
    return false;
  }
  const std::size_t expected_payload = sizeof(net->name) + sizeof(net->author) +
                                       sizeof(net->w0) + sizeof(net->b0) +
                                       sizeof(net->w1) + sizeof(net->b1) +
                                       sizeof(net->w2) + sizeof(net->b2);
  if (header.payload_size != expected_payload) {
    return false;
  }
  if (!ReadExact(file, net->name, sizeof(net->name))) {
    return false;
  }
  if (!ReadExact(file, net->author, sizeof(net->author))) {
    return false;
  }
  if (!ReadExact(file, net->w0.data(), sizeof(net->w0))) {
    return false;
  }
  if (!ReadExact(file, net->b0.data(), sizeof(net->b0))) {
    return false;
  }
  if (!ReadExact(file, net->w1.data(), sizeof(net->w1))) {
    return false;
  }
  if (!ReadExact(file, net->b1.data(), sizeof(net->b1))) {
    return false;
  }
  if (!ReadExact(file, net->w2.data(), sizeof(net->w2))) {
    return false;
  }
  if (!ReadExact(file, net->b2.data(), sizeof(net->b2))) {
    return false;
  }
  return true;
}

float SigmoidApprox(float x) {
  static const std::array<float, kSigmoidTableSize> table = []() {
    std::array<float, kSigmoidTableSize> vals{};
    for (int i = 0; i < kSigmoidTableSize; ++i) {
      float t =
          static_cast<float>(i) / static_cast<float>(kSigmoidTableSize - 1);
      float v = kSigmoidMin + t * (kSigmoidMax - kSigmoidMin);
      vals[i] = 1.0f / (1.0f + std::exp(-v));
    }
    return vals;
  }();

  if (x <= kSigmoidMin) {
    return table.front();
  }
  if (x >= kSigmoidMax) {
    return table.back();
  }
  float pos = (x - kSigmoidMin) / (kSigmoidMax - kSigmoidMin);
  float idx = pos * static_cast<float>(kSigmoidTableSize - 1);
  int i = static_cast<int>(idx);
  float frac = idx - static_cast<float>(i);
  return table[i] + frac * (table[i + 1] - table[i]);
}

uint32_t BlockedBits(const std::array<int, kNumPoints>& row) {
  uint32_t bits = 0;
  for (int i = 0; i < kNumPoints; ++i) {
    if (row[i] >= 1) {
      bits |= (1u << i);
    }
  }
  return bits;
}

const std::array<uint32_t, kNnueRunFeaturesPerSide>& RunMasks() {
  static const std::array<uint32_t, kNnueRunFeaturesPerSide> masks = []() {
    std::array<uint32_t, kNnueRunFeaturesPerSide> out{};
    int idx = 0;
    for (int len = kNnueRunMin; len <= kNnueRunMax; ++len) {
      int limit = kNumPoints - len;
      for (int start = 0; start <= limit; ++start) {
        uint32_t mask = 0;
        for (int i = 0; i < len; ++i) {
          mask |= (1u << (start + i));
        }
        out[idx++] = mask;
      }
    }
    return out;
  }();
  return masks;
}

void CollectActiveFeatures(const LongNardeState& state,
                           ActiveFeatures* active) {
  active->count = 0;
  const auto& board = state.board();
  for (int side = 0; side < kNumPlayers; ++side) {
    int base_offset = side * kNnueBaseFeaturesPerSide;
    int total = 0;
    for (int point = 0; point < kNumPoints; ++point) {
      int count = board[side][point];
      total += count;
      int bucket = std::min(std::max(count, 0), kNnueBucketCount - 1);
      int idx = base_offset + point * kNnueBucketCount + bucket;
      active->indices[active->count++] = idx;
    }
    int off = kNumCheckersPerPlayer - total;
    int off_bucket = std::min(std::max(off, 0), kNnueBucketCount - 1);
    int idx = base_offset + kNumPoints * kNnueBucketCount + off_bucket;
    active->indices[active->count++] = idx;
  }

  const auto& masks = RunMasks();
  for (int side = 0; side < kNumPlayers; ++side) {
    uint32_t blocked = BlockedBits(board[side]);
    int run_offset = kNnueBaseFeatures + side * kNnueRunFeaturesPerSide;
    for (int i = 0; i < kNnueRunFeaturesPerSide; ++i) {
      if ((blocked & masks[i]) == masks[i]) {
        active->indices[active->count++] = run_offset + i;
      }
    }
  }
}

void BuildAccumulator(const NnueNetwork& net, const ActiveFeatures& active,
                      std::array<int16_t, kNnueL1>* acc_out) {
  std::memcpy(acc_out->data(), net.b0.data(), kNnueL1 * sizeof(int16_t));
  AddRowFn add_row = GetAddRowKernel();
  for (int i = 0; i < active.count; ++i) {
    int feature = active.indices[i];
    const int16_t* row = &net.w0[feature * kNnueL1];
    add_row(row, acc_out->data());
  }
}

int32_t ComputeOutput(const int8_t* input, const int8_t* weights,
                      const int8_t* bias, int in_dim) {
  int32_t sum = bias[0] * kNnueFactor;
  for (int i = 0; i < in_dim; ++i) {
    sum += static_cast<int32_t>(input[i]) * weights[i];
  }
  return sum / kNnueFactor;
}

}  // namespace

bool NnueModel::Load(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    loaded_ = false;
    return false;
  }
  NnueFileHeader header{};
  if (ReadHeader(&file, &header)) {
    if (LoadHeaderPayload(&file, header, &network_)) {
      loaded_ = true;
      return true;
    }
    loaded_ = false;
    return false;
  }

  file.clear();
  file.seekg(0, std::ios::beg);
  if (!ReadExact(&file, &network_, sizeof(network_))) {
    loaded_ = false;
    return false;
  }
  loaded_ = true;
  return true;
}

NnueEval NnueEvaluator::EvaluateState(const LongNardeState& state) const {
  NnueEval eval;
  if (model_ == nullptr || !model_->IsLoaded()) {
    return eval;
  }

  ActiveFeatures active;
  CollectActiveFeatures(state, &active);

  std::array<int16_t, kNnueL1> acc;
  BuildAccumulator(model_->network(), active, &acc);

  std::array<int8_t, kNnueL1> l1;
  for (int i = 0; i < kNnueL1; ++i) {
    l1[i] = ClampAcc(acc[i]);
  }

  std::array<int8_t, kNnueL2> l2;
  ComputeLayerFn compute_layer = GetComputeLayerKernel();
  compute_layer(l1.data(), model_->network().w1.data(),
                model_->network().b1.data(), kNnueL1, kNnueL2, l2.data());

  int32_t out_win = ComputeOutput(l2.data(), model_->network().w2.data(),
                                  &model_->network().b2[0], kNnueL2);
  int32_t out_mars =
      ComputeOutput(l2.data(), model_->network().w2.data() + kNnueL2,
                    &model_->network().b2[1], kNnueL2);

  float logit_win = static_cast<float>(out_win);
  float logit_mars = static_cast<float>(out_mars);
  eval.p_win = SigmoidApprox(logit_win);
  eval.p_mars = SigmoidApprox(logit_mars);
  eval.ev = eval.p_win + eval.p_mars;
  return eval;
}

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel
