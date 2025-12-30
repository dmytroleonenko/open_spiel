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

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/long_narde/long_narde_lnue_tool_lib.h"

namespace open_spiel {
namespace long_narde {
namespace {

void PrintUsage(const char* bin) {
  std::cerr << "Usage:\n"
            << "  " << bin
            << " split --games_per_shard N --out_dir PATH [--out_prefix name]\n"
            << "        [--start_index N] [--samples_per_chunk N]\n"
            << "        input1.lnue [input2...]\n"
            << "  " << bin
            << " join --games_per_shard N --out_shards N --out_dir PATH\n"
            << "       [--out_prefix name] [--start_index N]\n"
            << "       [--samples_per_chunk N]\n"
            << "       input1.lnue [input2...]\n";
}

std::unordered_map<std::string, std::string> ParseArgs(
    int argc, char** argv, std::vector<std::string>* inputs) {
  std::unordered_map<std::string, std::string> out;
  if (inputs != nullptr) {
    inputs->clear();
  }
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      if (inputs != nullptr) {
        inputs->push_back(arg);
      }
      continue;
    }
    arg = arg.substr(2);
    auto eq = arg.find('=');
    if (eq != std::string::npos) {
      out[arg.substr(0, eq)] = arg.substr(eq + 1);
      continue;
    }
    std::string value;
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      value = argv[++i];
    }
    out[arg] = value;
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

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::GetIntArg;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::ParseShardName;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::ProcessInputs;
  using open_spiel::long_narde::ToolConfig;

  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }
  std::string mode = argv[1];
  if (mode != "split" && mode != "join") {
    PrintUsage(argv[0]);
    return 1;
  }

  std::vector<std::string> inputs;
  auto args = ParseArgs(argc, argv, &inputs);
  if (inputs.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  ToolConfig config;
  config.out_dir = GetStringArg(args, "out_dir", "");
  bool has_out_prefix = args.find("out_prefix") != args.end();
  bool has_start_index = args.find("start_index") != args.end();
  if (has_out_prefix) {
    config.out_prefix = GetStringArg(args, "out_prefix", config.out_prefix);
  }
  config.games_per_shard = GetIntArg(args, "games_per_shard", 0);
  config.samples_per_chunk = GetIntArg(args, "samples_per_chunk",
                                       config.samples_per_chunk);
  if (has_start_index) {
    config.start_index = GetIntArg(args, "start_index", 0);
  }
  config.out_shards = GetIntArg(args, "out_shards", 0);
  config.limit_output = (mode == "join");

  if (!has_out_prefix || !has_start_index) {
    std::string inferred_prefix;
    int inferred_index = 0;
    if (ParseShardName(inputs.front(), &inferred_prefix, &inferred_index)) {
      if (!has_out_prefix) {
        config.out_prefix = inferred_prefix;
      }
      if (!has_start_index) {
        config.start_index = inferred_index;
      }
    }
  }

  if (config.out_dir.empty() || config.games_per_shard <= 0) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (mode == "join" && config.out_shards <= 0) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (!ProcessInputs(inputs, config)) {
    return 1;
  }
  return 0;
}
