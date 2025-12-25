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

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>

#include "open_spiel/games/long_narde/long_narde_nats.h"

namespace open_spiel {
namespace long_narde {
namespace {

struct PublishConfig {
  std::string nats_url = "nats://127.0.0.1:4222";
  std::string run_id = "default";
  std::string weights_subject = "nnue.weights";
  std::string path;
};

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

std::string GetStringArg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key, const std::string& default_value) {
  auto it = args.find(key);
  if (it == args.end() || it->second.empty()) {
    return default_value;
  }
  return it->second;
}

std::string BuildSubject(const std::string& base, const std::string& run_id) {
  if (run_id.empty()) {
    return base;
  }
  return base + "." + run_id;
}

void PrintUsage(const char* bin) {
  std::cout << "Usage: " << bin
            << " [--nats url] [--run_id id] [--weights_subject name]"
            << " --file path\n";
}

}  // namespace
}  // namespace long_narde
}  // namespace open_spiel

int main(int argc, char** argv) {
  using open_spiel::long_narde::BuildSubject;
  using open_spiel::long_narde::GetStringArg;
  using open_spiel::long_narde::NatsConnection;
  using open_spiel::long_narde::ParseArgs;
  using open_spiel::long_narde::PrintUsage;
  using open_spiel::long_narde::PublishConfig;

  auto args = ParseArgs(argc, argv);
  if (args.find("help") != args.end()) {
    PrintUsage(argv[0]);
    return 0;
  }

  PublishConfig config;
  config.nats_url = GetStringArg(args, "nats", config.nats_url);
  config.run_id = GetStringArg(args, "run_id", config.run_id);
  config.weights_subject = GetStringArg(args, "weights_subject",
                                       config.weights_subject);
  config.path = GetStringArg(args, "file", config.path);

  if (config.path.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::ifstream file(config.path, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << config.path << "\n";
    return 1;
  }
  std::string payload((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  file.close();

  NatsConnection conn;
  if (!conn.Connect(config.nats_url)) {
    std::cerr << "Failed to connect to NATS at " << config.nats_url << "\n";
    return 1;
  }
  std::string subject = BuildSubject(config.weights_subject, config.run_id);
  if (!conn.Publish(subject, payload)) {
    std::cerr << "Failed to publish weights to " << subject << "\n";
    return 1;
  }
  std::cout << "Published " << payload.size() << " bytes to " << subject
            << "\n";
  return 0;
}
