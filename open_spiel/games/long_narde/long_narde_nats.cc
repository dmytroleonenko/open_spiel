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

#include "open_spiel/games/long_narde/long_narde_nats.h"

#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace open_spiel {
namespace long_narde {
namespace {

struct HostPort {
  std::string host;
  std::string port;
};

bool ParseUrl(const std::string& url, HostPort* out) {
  if (out == nullptr) {
    return false;
  }
  std::string trimmed = url;
  const std::string prefix = "nats://";
  if (trimmed.rfind(prefix, 0) == 0) {
    trimmed = trimmed.substr(prefix.size());
  }
  std::string host = trimmed;
  std::string port = "4222";
  std::size_t colon = trimmed.find(':');
  if (colon != std::string::npos) {
    host = trimmed.substr(0, colon);
    port = trimmed.substr(colon + 1);
  }
  if (host.empty()) {
    return false;
  }
  out->host = host;
  out->port = port;
  return true;
}

#if defined(_WIN32)
bool EnsureWsaInit() {
  static bool initialized = false;
  if (initialized) {
    return true;
  }
  WSADATA wsa_data;
  int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
  if (result != 0) {
    return false;
  }
  initialized = true;
  return true;
}
#endif

int CreateSocket(const HostPort& hp) {
#if defined(_WIN32)
  if (!EnsureWsaInit()) {
    return -1;
  }
#endif
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  if (getaddrinfo(hp.host.c_str(), hp.port.c_str(), &hints, &result) != 0) {
    return -1;
  }

  int sock = -1;
  for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    sock = static_cast<int>(::socket(rp->ai_family, rp->ai_socktype,
                                     rp->ai_protocol));
    if (sock < 0) {
      continue;
    }
    if (::connect(sock, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) {
      break;
    }
#if defined(_WIN32)
    closesocket(sock);
#else
    close(sock);
#endif
    sock = -1;
  }
  freeaddrinfo(result);
  return sock;
}

}  // namespace

NatsConnection::~NatsConnection() { Close(); }

bool NatsConnection::Connect(const std::string& url) {
  Close();
  HostPort hp;
  if (!ParseUrl(url, &hp)) {
    return false;
  }
  socket_ = CreateSocket(hp);
  if (socket_ < 0) {
    return false;
  }

  const std::string connect =
      "CONNECT {\"verbose\":false,\"pedantic\":false,"
      "\"lang\":\"cpp\",\"version\":\"1.0\"}\r\n";
  if (!SendAll(connect.data(), connect.size())) {
    Close();
    return false;
  }
  return true;
}

bool NatsConnection::Publish(const std::string& subject,
                             const std::string& payload) {
  if (socket_ < 0) {
    return false;
  }
  std::ostringstream oss;
  oss << "PUB " << subject << " " << payload.size() << "\r\n";
  std::string header = oss.str();
  if (!SendAll(header.data(), header.size())) {
    return false;
  }
  if (!payload.empty() && !SendAll(payload.data(), payload.size())) {
    return false;
  }
  return SendAll("\r\n", 2);
}

bool NatsConnection::Subscribe(const std::string& subject, int sid) {
  if (socket_ < 0) {
    return false;
  }
  std::ostringstream oss;
  oss << "SUB " << subject << " " << sid << "\r\n";
  std::string cmd = oss.str();
  return SendAll(cmd.data(), cmd.size());
}

bool NatsConnection::NextMessage(NatsMessage* out) {
  if (out == nullptr) {
    return false;
  }
  while (true) {
    std::string line;
    if (!ReadLine(&line)) {
      return false;
    }
    if (line == "PING") {
      if (!WritePong()) {
        return false;
      }
      continue;
    }
    if (line == "+OK" || line.empty()) {
      continue;
    }
    if (line.rfind("MSG ", 0) != 0) {
      continue;
    }

    std::istringstream iss(line);
    std::string tag;
    std::string subject;
    std::string sid;
    std::string reply;
    std::string size_str;
    iss >> tag >> subject >> sid;
    if (!(iss >> size_str)) {
      iss.clear();
      iss >> reply >> size_str;
    }
    if (size_str.empty()) {
      continue;
    }
    std::size_t size = static_cast<std::size_t>(std::stoul(size_str));
    std::string payload;
    if (!ReadBytes(&payload, size)) {
      return false;
    }
    if (!ReadBytes(&line, 2)) {
      return false;
    }
    out->subject = subject;
    out->payload = std::move(payload);
    return true;
  }
}

void NatsConnection::Close() {
  if (socket_ >= 0) {
#if defined(_WIN32)
    closesocket(socket_);
#else
    close(socket_);
#endif
    socket_ = -1;
  }
  read_buffer_.clear();
}

bool NatsConnection::SendAll(const void* data, std::size_t size) {
  const char* ptr = reinterpret_cast<const char*>(data);
  std::size_t remaining = size;
  while (remaining > 0) {
#if defined(_WIN32)
    int sent = ::send(socket_, ptr, static_cast<int>(remaining), 0);
#else
    ssize_t sent = ::send(socket_, ptr, remaining, 0);
#endif
    if (sent <= 0) {
      return false;
    }
    ptr += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return true;
}

bool NatsConnection::ReadLine(std::string* line) {
  while (true) {
    std::size_t pos = read_buffer_.find("\r\n");
    if (pos != std::string::npos) {
      *line = read_buffer_.substr(0, pos);
      read_buffer_.erase(0, pos + 2);
      return true;
    }
    if (!ReadMore()) {
      return false;
    }
  }
}

bool NatsConnection::ReadBytes(std::string* out, std::size_t size) {
  if (!EnsureAvailable(size)) {
    return false;
  }
  *out = read_buffer_.substr(0, size);
  read_buffer_.erase(0, size);
  return true;
}

bool NatsConnection::EnsureAvailable(std::size_t size) {
  while (read_buffer_.size() < size) {
    if (!ReadMore()) {
      return false;
    }
  }
  return true;
}

bool NatsConnection::ReadMore() {
  char buf[4096];
#if defined(_WIN32)
  int received = ::recv(socket_, buf, sizeof(buf), 0);
#else
  ssize_t received = ::recv(socket_, buf, sizeof(buf), 0);
#endif
  if (received <= 0) {
    return false;
  }
  read_buffer_.append(buf, static_cast<std::size_t>(received));
  return true;
}

bool NatsConnection::WritePing() { return SendAll("PING\r\n", 6); }

bool NatsConnection::WritePong() { return SendAll("PONG\r\n", 6); }

}  // namespace long_narde
}  // namespace open_spiel
