#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
ITERATIONS="${ITERATIONS:-200}"
START_ITER="${START_ITER:-0}"
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
CUR_NNUE="${CUR_NNUE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/nnue_closed_loop}"
SELFPLAY_ROOT="${SELFPLAY_ROOT:-results/nnue_stream}"
SKIP_SELFPLAY="${SKIP_SELFPLAY:-0}"

LEARNER_BIN="${LEARNER_BIN:-}"
WORKER_BIN="${WORKER_BIN:-}"
EVAL_BIN="${EVAL_BIN:-}"

resolve_bin() {
  local name="$1"
  local override="$2"
  if [[ -n "$override" && -x "$override" ]]; then
    echo "$override"
    return 0
  fi
  local candidates=(
    "$REPO_ROOT/build-release/games/$name"
    "$REPO_ROOT/build-relwithdebinfo/games/$name"
    "$REPO_ROOT/build/games/$name"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if ! LEARNER_BIN="$(resolve_bin long_narde_nats_learner "$LEARNER_BIN")"; then
  echo "missing learner binary. Set LEARNER_BIN or build targets." >&2
  exit 1
fi
if ! WORKER_BIN="$(resolve_bin long_narde_nats_worker "$WORKER_BIN")"; then
  echo "missing worker binary. Set WORKER_BIN or build targets." >&2
  exit 1
fi
if ! EVAL_BIN="$(resolve_bin long_narde_eval "$EVAL_BIN")"; then
  echo "missing eval binary. Set EVAL_BIN or build targets." >&2
  exit 1
fi

echo "using learner_bin: $LEARNER_BIN" >&2
echo "using worker_bin: $WORKER_BIN" >&2
echo "using eval_bin: $EVAL_BIN" >&2
if [[ -z "$CUR_NNUE" ]]; then
  CUR_NNUE="$INIT_NNUE"
fi

if [[ "$START_ITER" -gt 0 && -z "${CUR_NNUE:-}" ]]; then
  prev_iter=$((START_ITER - 1))
  prev_path=$(printf "%s/iter_%03d/iter_00/nnue_iter_00.nnue" \
    "$OUTPUT_ROOT" "$prev_iter")
  if [[ -f "$prev_path" ]]; then
    CUR_NNUE="$prev_path"
  fi
fi

if [[ ! -f "$CUR_NNUE" ]]; then
  echo "missing init nnue: $CUR_NNUE" >&2
  exit 1
fi

for ((iter=START_ITER; iter<ITERATIONS; iter++)); do
  RUN_ID=$(printf "iter_%03d" "$iter")
  OUT_DIR="$SELFPLAY_ROOT/$RUN_ID"
  TRAIN_OUT="$OUTPUT_ROOT/$RUN_ID"

  mkdir -p "$OUT_DIR"
  mkdir -p "$TRAIN_OUT"

  use_existing=0
  if [[ "$SKIP_SELFPLAY" -eq 1 ]]; then
    if [[ -d "$OUT_DIR" ]] && compgen -G "$OUT_DIR/*.lnue" > /dev/null; then
      use_existing=1
    fi
  fi

  if [[ "$use_existing" -eq 1 ]]; then
    echo "iter $iter: using existing shards in $OUT_DIR" >&2
  else
    if [[ "$SKIP_SELFPLAY" -eq 1 ]]; then
      echo "iter $iter: no shards in $OUT_DIR, running NATS selfplay" >&2
    fi
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
  fi

  echo "iter $iter: train/eval from $OUT_DIR" >&2

  ./venv/bin/python -m open_spiel.python.algorithms.long_narde_nnue.closed_loop \
    --iterations 1 \
    --skip_selfplay \
    --selfplay_dir "$OUT_DIR" \
    --output_dir "$TRAIN_OUT" \
    --eval_games "$EVAL_GAMES" \
    --eval_bin "$EVAL_BIN" \
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
