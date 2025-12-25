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

#include "open_spiel/games/long_narde/long_narde_nnue.h"

#include <cstring>
#include <fstream>
#include <string>

namespace open_spiel {
namespace long_narde {
namespace nnue {
namespace {

bool ReadExact(std::ifstream* file, void* dst, std::size_t size) {
  file->read(reinterpret_cast<char*>(dst), size);
  return file->good() && file->gcount() == static_cast<std::streamsize>(size);
}

bool ReadExact(const char* data, std::size_t size, std::size_t* offset,
               void* dst, std::size_t bytes) {
  if (offset == nullptr || dst == nullptr) {
    return false;
  }
  if (*offset + bytes > size) {
    return false;
  }
  std::memcpy(dst, data + *offset, bytes);
  *offset += bytes;
  return true;
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

bool ReadHeaderFromBuffer(const char* data, std::size_t size,
                          std::size_t* offset, NnueFileHeader* header) {
  if (!ReadExact(data, size, offset, header, sizeof(*header))) {
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

bool LoadHeaderPayloadFromBuffer(const char* data, std::size_t size,
                                 std::size_t* offset,
                                 const NnueFileHeader& header,
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
  if (!ReadExact(data, size, offset, net->name, sizeof(net->name))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->author, sizeof(net->author))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->w0.data(), sizeof(net->w0))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->b0.data(), sizeof(net->b0))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->w1.data(), sizeof(net->w1))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->b1.data(), sizeof(net->b1))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->w2.data(), sizeof(net->w2))) {
    return false;
  }
  if (!ReadExact(data, size, offset, net->b2.data(), sizeof(net->b2))) {
    return false;
  }
  return true;
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

bool NnueModel::LoadFromBytes(const std::string& payload) {
  loaded_ = false;
  if (payload.empty()) {
    return false;
  }
  const char* data = payload.data();
  std::size_t size = payload.size();
  std::size_t offset = 0;
  NnueFileHeader header{};
  if (ReadHeaderFromBuffer(data, size, &offset, &header)) {
    if (LoadHeaderPayloadFromBuffer(data, size, &offset, header, &network_)) {
      loaded_ = true;
      return true;
    }
    loaded_ = false;
    return false;
  }
  if (payload.size() == sizeof(network_)) {
    std::memcpy(&network_, payload.data(), sizeof(network_));
    loaded_ = true;
    return true;
  }
  return false;
}

}  // namespace nnue
}  // namespace long_narde
}  // namespace open_spiel
