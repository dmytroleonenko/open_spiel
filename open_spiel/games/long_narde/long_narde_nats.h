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

#ifndef OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_H_
#define OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace open_spiel {
namespace long_narde {

struct NatsMessage {
  std::string subject;
  std::string payload;
};

class NatsConnection {
 public:
  NatsConnection() = default;
  ~NatsConnection();

  NatsConnection(const NatsConnection&) = delete;
  NatsConnection& operator=(const NatsConnection&) = delete;

  bool Connect(const std::string& url);
  bool SetReceiveTimeout(int timeout_ms);
  bool Publish(const std::string& subject, const std::string& payload);
  bool Subscribe(const std::string& subject, int sid);
  bool NextMessage(NatsMessage* out);
  void Close();

 private:
  bool SendAll(const void* data, std::size_t size);
  bool ReadLine(std::string* line);
  bool ReadBytes(std::string* out, std::size_t size);
  bool EnsureAvailable(std::size_t size);
  bool ReadMore();
  bool WritePing();
  bool WritePong();

  int socket_ = -1;
  std::string read_buffer_;
};

}  // namespace long_narde
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LONG_NARDE_LONG_NARDE_NATS_H_
