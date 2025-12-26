#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
ITERATIONS="${ITERATIONS:-200}"
DEPTH="${DEPTH:-2}"
GAMES_PER_ITER="${GAMES_PER_ITER:-200}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-50}"
EVAL_GAMES="${EVAL_GAMES:-300}"
EVAL_REPORT_EVERY="${EVAL_REPORT_EVERY:-25}"
EVAL_PROGRESS="${EVAL_PROGRESS:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-0}"
WORKERS="${WORKERS:-0}"
SEED="${SEED:-12345}"
INIT_NNUE="${INIT_NNUE:-results/nnue_closed_loop/iter_00/nnue_iter_00.nnue}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/nnue_closed_loop}"
SELFPLAY_ROOT="${SELFPLAY_ROOT:-results/nnue_stream}"

LEARNER_BIN="${LEARNER_BIN:-$REPO_ROOT/build/games/long_narde_nats_learner}"
WORKER_BIN="${WORKER_BIN:-$REPO_ROOT/build/games/long_narde_nats_worker}"

if [[ ! -x "$LEARNER_BIN" ]]; then
  echo "missing learner binary: $LEARNER_BIN" >&2
  exit 1
fi
if [[ ! -x "$WORKER_BIN" ]]; then
  echo "missing worker binary: $WORKER_BIN" >&2
  exit 1
fi
if [[ ! -f "$INIT_NNUE" ]]; then
  echo "missing init nnue: $INIT_NNUE" >&2
  exit 1
fi

CUR_NNUE="$INIT_NNUE"

for ((iter=0; iter<ITERATIONS; iter++)); do
  RUN_ID=$(printf "iter_%03d" "$iter")
  OUT_DIR="$SELFPLAY_ROOT/$RUN_ID"
  TRAIN_OUT="$OUTPUT_ROOT/$RUN_ID"

  mkdir -p "$OUT_DIR"
  mkdir -p "$TRAIN_OUT"

  echo "iter $iter: selfplay via NATS (run_id=$RUN_ID)" >&2

  "$LEARNER_BIN" \
    --nats "$NATS_URL" \
    --run_id "$RUN_ID" \
    --out_dir "$OUT_DIR" \
    --games_per_shard "$GAMES_PER_SHARD" \
    --max_games "$GAMES_PER_ITER" \
    --report_every "$EVAL_REPORT_EVERY" \
    --progress 1 &
  LEARNER_PID=$!

  sleep 1

  "$WORKER_BIN" \
    --nats "$NATS_URL" \
    --run_id "$RUN_ID" \
    --depth "$DEPTH" \
    --workers "$WORKERS" \
    --games "$GAMES_PER_ITER" \
    --nnue "$CUR_NNUE" &
  WORKER_PID=$!

  wait "$WORKER_PID"
  wait "$LEARNER_PID"

  echo "iter $iter: train/eval from $OUT_DIR" >&2

  ./venv/bin/python -m open_spiel.python.algorithms.long_narde_nnue.closed_loop \
    --iterations 1 \
    --skip_selfplay \
    --selfplay_dir "$OUT_DIR" \
    --output_dir "$TRAIN_OUT" \
    --eval_games "$EVAL_GAMES" \
    --eval_progress "$EVAL_PROGRESS" \
    --eval_report_every "$EVAL_REPORT_EVERY" \
    --eval_workers "$EVAL_WORKERS" \
    --depth "$DEPTH" \
    --device cpu \
    --init_nnue "$CUR_NNUE" \
    --seed "$SEED"

  CUR_NNUE="$TRAIN_OUT/iter_00/nnue_iter_00.nnue"
  if [[ ! -f "$CUR_NNUE" ]]; then
    echo "missing trained nnue: $CUR_NNUE" >&2
    exit 1
  fi

done
