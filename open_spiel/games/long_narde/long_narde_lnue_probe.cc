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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_nnue.h"
#include "open_spiel/games/long_narde/long_narde_selfplay_io.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace long_narde {

constexpr std::array<char, 4> kLnueMagic = {'L', 'N', 'U', 'E'};
constexpr std::array<char, 4> kLnueChunkMagic = {'C', 'H', 'N', 'K'};

struct LnueHeader {
  char magic[4];
  uint32_t version;
  uint32_t endian;
  uint32_t schema_id;
  uint32_t feature_dim;
  uint32_t flags;
  uint32_t feature_index_bytes;
  uint64_t seed;
  uint32_t shard_id;
  uint32_t run_block_threshold;
  uint32_t reserved[5];
};

struct LnueChunkHeader {
  char magic[4];
  uint32_t uncompressed_bytes;
  uint32_t compressed_bytes;
  uint32_t num_samples;
  uint32_t reserved[3];
};

bool ReadExact(std::ifstream* in, void* data, std::size_t size) {
  in->read(reinterpret_cast<char*>(data), size);
  return in->good() && in->gcount() == static_cast<std::streamsize>(size);
}

std::unordered_map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::unordered_map<std::string, std::string> out;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      continue;
    }
    arg = arg.substr(2);
    auto eq = arg.find('=');
    if (eq != std::string::npos) {
      out[arg.substr(0, eq)] = arg.substr(eq + 1);
    } else {
      std::string value;
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        value = argv[++i];
      }
      out[arg] = value;
    }
  }
  return out;
}

int GetIntArg(const std::unordered_map<std::string, std::string>& args,
              const std::string& key, int default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

std::string GetStringArg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, const std::string& default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return it->second;
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin << " --shard path --nnue path --out path"
            << " [--max_samples N]\n";
}

bool ReadLnueHeader(std::ifstream* in, LnueHeader* header) {
  if (!ReadExact(in, header->magic, sizeof(header->magic))) {
    return false;
  }
  if (!ReadExact(in, &header->version, sizeof(header->version)) ||
      !ReadExact(in, &header->endian, sizeof(header->endian)) ||
      !ReadExact(in, &header->schema_id, sizeof(header->schema_id)) ||
      !ReadExact(in, &header->feature_dim, sizeof(header->feature_dim)) ||
      !ReadExact(in, &header->flags, sizeof(header->flags)) ||
      !ReadExact(in, &header->feature_index_bytes,
                 sizeof(header->feature_index_bytes)) ||
      !ReadExact(in, &header->seed, sizeof(header->seed)) ||
      !ReadExact(in, &header->shard_id, sizeof(header->shard_id)) ||
      !ReadExact(in, &header->run_block_threshold,
                 sizeof(header->run_block_threshold)) ||
      !ReadExact(in, header->reserved, sizeof(header->reserved))) {
    return false;
  }
  if (std::memcmp(header->magic, kLnueMagic.data(), kLnueMagic.size()) != 0) {
    return false;
  }
  if (header->version != 1 && header->version != kLnueFormatVersion) {
    return false;
  }
  if (header->endian != kLnueEndianMarker) {
    return false;
  }
  if (header->feature_index_bytes != 2 && header->feature_index_bytes != 4) {
    return false;
  }
  if (header->feature_dim != nnue::kNnueFeatureDim) {
    return false;
  }
  return true;
}

bool ReadChunkHeader(std::ifstream* in, LnueChunkHeader* header) {
  if (!ReadExact(in, header->magic, sizeof(header->magic)) ||
      !ReadExact(in, &header->uncompressed_bytes,
                 sizeof(header->uncompressed_bytes)) ||
      !ReadExact(in, &header->compressed_bytes,
                 sizeof(header->compressed_bytes)) ||
      !ReadExact(in, &header->num_samples, sizeof(header->num_samples)) ||
      !ReadExact(in, header->reserved, sizeof(header->reserved))) {
    return false;
  }
  if (std::memcmp(header->magic, kLnueChunkMagic.data(),
                  kLnueChunkMagic.size()) != 0) {
    return false;
  }
  if (header->compressed_bytes != 0) {
    return false;
  }
  return true;
}

int RunLnueProbe(int argc, char** argv) {
  using open_spiel::long_narde::LnueChunkHeader;
  using open_spiel::long_narde::LnueHeader;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::ReadChunkHeader;
  using open_spiel::long_narde::ReadLnueHeader;
  using open_spiel::long_narde::nnue::EvaluateRawFromFeatures;
  using open_spiel::long_narde::nnue::NnueModel;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  std::string shard_path =
      open_spiel::long_narde::GetStringArg(args, "shard", "");
  std::string nnue_path =
      open_spiel::long_narde::GetStringArg(args, "nnue", "");
  std::string out_path =
      open_spiel::long_narde::GetStringArg(args, "out", "");
  int max_samples = open_spiel::long_narde::GetIntArg(args, "max_samples", 0);

  if (shard_path.empty() || nnue_path.empty() || out_path.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  NnueModel model;
  if (!model.Load(nnue_path)) {
    std::cerr << "Failed to load NNUE from " << nnue_path << "\n";
    return 1;
  }

  std::ifstream in(shard_path, std::ios::binary | std::ios::in);
  if (!in.is_open()) {
    std::cerr << "Failed to open shard " << shard_path << "\n";
    return 1;
  }

  LnueHeader header{};
  if (!ReadLnueHeader(&in, &header)) {
    std::cerr << "Invalid LNUE header in " << shard_path << "\n";
    return 1;
  }

  std::vector<int32_t> logits;
  logits.reserve(static_cast<size_t>(max_samples > 0 ? max_samples : 1024) * 2);

  int processed = 0;
  while (max_samples <= 0 || processed < max_samples) {
    LnueChunkHeader chunk{};
    if (!ReadChunkHeader(&in, &chunk)) {
      break;
    }

    int num_samples = static_cast<int>(chunk.num_samples);
    std::vector<uint32_t> offsets(num_samples + 1);
    if (!ReadExact(&in, offsets.data(),
                   offsets.size() * sizeof(uint32_t))) {
      std::cerr << "Failed to read LNUE offsets.\n";
      return 1;
    }
    uint32_t indices_len = offsets.back();

    std::vector<uint16_t> indices16;
    std::vector<uint32_t> indices32;
    if (header.feature_index_bytes == 2) {
      indices16.resize(indices_len);
      if (!ReadExact(&in, indices16.data(),
                     indices16.size() * sizeof(uint16_t))) {
        std::cerr << "Failed to read LNUE indices.\n";
        return 1;
      }
    } else {
      indices32.resize(indices_len);
      if (!ReadExact(&in, indices32.data(),
                     indices32.size() * sizeof(uint32_t))) {
        std::cerr << "Failed to read LNUE indices.\n";
        return 1;
      }
    }

    std::vector<float> v_search(num_samples);
    std::vector<int8_t> outcome(num_samples);
    std::vector<float> target(num_samples);
    std::vector<uint64_t> game_id(num_samples);
    std::vector<uint16_t> ply(num_samples);

    if (!ReadExact(&in, v_search.data(), num_samples * sizeof(float)) ||
        !ReadExact(&in, outcome.data(), num_samples * sizeof(int8_t)) ||
        !ReadExact(&in, target.data(), num_samples * sizeof(float)) ||
        !ReadExact(&in, game_id.data(), num_samples * sizeof(uint64_t)) ||
        !ReadExact(&in, ply.data(), num_samples * sizeof(uint16_t))) {
      std::cerr << "Failed to read LNUE columns.\n";
      return 1;
    }

    for (int i = 0; i < num_samples; ++i) {
      if (max_samples > 0 && processed >= max_samples) {
        break;
      }
      uint32_t start = offsets[i];
      uint32_t end = offsets[i + 1];
      std::vector<int> features;
      features.reserve(end - start);
      if (header.feature_index_bytes == 2) {
        for (uint32_t j = start; j < end; ++j) {
          features.push_back(static_cast<int>(indices16[j]));
        }
      } else {
        for (uint32_t j = start; j < end; ++j) {
          features.push_back(static_cast<int>(indices32[j]));
        }
      }

      auto raw = EvaluateRawFromFeatures(model.network(), features);
      logits.push_back(raw.logit_win);
      logits.push_back(raw.logit_mars);
      ++processed;
    }
  }

  std::ofstream out(out_path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Failed to write output " << out_path << "\n";
    return 1;
  }
  uint32_t count = static_cast<uint32_t>(processed);
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  out.write(reinterpret_cast<const char*>(logits.data()),
            logits.size() * sizeof(int32_t));
  out.close();
  return 0;
}

}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  return open_spiel::long_narde::RunLnueProbe(argc, argv);
}
