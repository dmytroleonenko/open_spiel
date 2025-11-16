#!/usr/bin/env bash
# Launch local MuZero gRPC services (replay, publisher) and print endpoints/log paths.
# Example:
#   ./scripts/launch_local_muzero_services.sh --replay 1 --publisher 1 --log-dir /tmp
set -euo pipefail

REPLAY_COUNT=1
PUBLISHER_COUNT=1
LOG_DIR="/tmp"
PYTHON_BIN=${PYTHON_BIN:-python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay) REPLAY_COUNT="$2"; shift 2 ;;
    --publisher) PUBLISHER_COUNT="$2"; shift 2 ;;
    --inference) echo "Inference launch removed; ignoring --inference." >&2; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --replay N --publisher N --inference N [--log-dir /tmp] [--python python]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOG_DIR}"

start_replay() {
  local idx=$1
  local log="${LOG_DIR}/mz_replay_${idx}.log"
  ${PYTHON_BIN} - <<'PY' >"${log}" 2>&1 &
from open_spiel.python.algorithms.muzero_jax.services.replay_service import GrpcReplayServer, InMemoryReplayService
backend = InMemoryReplayService(capacity=4000, alpha=0.6)
server = GrpcReplayServer(backend)
server.start()
print(server.endpoint, flush=True)
import time; time.sleep(10**9)
PY
  echo "$!" > "${log}.pid"
  local ep
  for _ in {1..30}; do
    ep=$(grep -m1 -E '127\.0\.0\.1:[0-9]+' "${log}" || true)
    [[ -n "${ep}" ]] && break
    sleep 0.1
  done
  echo "REPLAY_ENDPOINT_${idx}=${ep} LOG=${log} PID=$(cat ${log}.pid)"
}

start_publisher() {
  local idx=$1
  local log="${LOG_DIR}/mz_pub_${idx}.log"
  ${PYTHON_BIN} - <<'PY' >"${log}" 2>&1 &
from open_spiel.python.algorithms.muzero_jax.services.parameter_publisher import GrpcParameterPublisherServer, LocalParameterPublisher
backend = LocalParameterPublisher()
server = GrpcParameterPublisherServer(backend)
server.start()
print(server.endpoint, flush=True)
import time; time.sleep(10**9)
PY
  echo "$!" > "${log}.pid"
  local ep
  for _ in {1..30}; do
    ep=$(grep -m1 -E '127\.0\.0\.1:[0-9]+' "${log}" || true)
    [[ -n "${ep}" ]] && break
    sleep 0.1
  done
  echo "PUBLISHER_ENDPOINT_${idx}=${ep} LOG=${log} PID=$(cat ${log}.pid)"
}

echo "# Starting services (logs in ${LOG_DIR})"
for i in $(seq 1 ${REPLAY_COUNT}); do start_replay "${i}"; done
for i in $(seq 1 ${PUBLISHER_COUNT}); do start_publisher "${i}"; done

echo "# Example exports (first instance of each):"
echo "export REPLAY_ENDPOINT=$(grep -m1 -E '127\.0\.0\.1:[0-9]+' "${LOG_DIR}/mz_replay_1.log")"
echo "export PUBLISHER_ENDPOINT=$(grep -m1 -E '127\.0\.0\.1:[0-9]+' "${LOG_DIR}/mz_pub_1.log")"
