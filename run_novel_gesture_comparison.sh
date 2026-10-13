#!/usr/bin/env bash
# Run the class-disjoint supervised-vs-ProtoNet comparison resumably.
# Example:
#   DATA_DIR=/data/cropped_images WORK_DIR=/data/output bash run_novel_gesture_comparison.sh
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d /home/soumik/soumik/misc/cropped_images ]]; then
    export DATA_DIR=/home/soumik/soumik/misc/cropped_images
  else
    export DATA_DIR="$PROJECT_DIR/gestures/cropped_images"
  fi
fi
export WORK_DIR="${WORK_DIR:-/home/soumik/soumik/misc/output}"
export SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-$WORK_DIR/split_manifest.json}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# The supervised trainer uses all visible GPUs through PyTorch DataParallel.
# This setting reduces CUDA allocator fragmentation on long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$WORK_DIR/logs"
LOG_FILE="$LOG_DIR/novel_gesture_comparison.log"
STATUS_FILE="$LOG_DIR/novel_gesture_comparison.status"
PID_FILE="$LOG_DIR/novel_gesture_comparison.pid"
mkdir -p "$LOG_DIR"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

write_status() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee "$STATUS_FILE"
}

run_phase() {
  local name="$1"; shift
  write_status "RUNNING: $name"
  echo "[$(date -Is)] START $name"
  if "$@"; then
    echo "[$(date -Is)] DONE $name"
    write_status "DONE: $name"
  else
    local code=$?
    echo "[$(date -Is)] FAILED $name (exit $code)" >&2
    write_status "FAILED: $name (exit $code)"
    return "$code"
  fi
}

run_all() {
  cd "$PROJECT_DIR"
  write_status "STARTED: class-disjoint novel-gesture comparison"
  echo "Project: $PROJECT_DIR"
  echo "Data:    $DATA_DIR"
  echo "Output:  $WORK_DIR"
  nvidia-smi || true

  run_phase supervised_train "$PYTHON_BIN" scripts/train_supervised_base.py \
    --steps "${SUPERVISED_STEPS:-1000}" \
    --batch-size "${SUPERVISED_BATCH_SIZE:-32}" \
    --workers "${SUPERVISED_WORKERS:-0}"
  run_phase supervised_base_test "$PYTHON_BIN" scripts/evaluate_supervised_base.py
  run_phase novel_gesture_evaluation "$PYTHON_BIN" scripts/evaluate_heldout_gesture_grid.py
  run_phase comparison_report "$PYTHON_BIN" scripts/compare_heldout_gestures.py
  write_status "COMPLETE: class-disjoint novel-gesture comparison"
}

if [[ "${1:-}" == "--foreground" ]]; then
  trap 'write_status "FAILED: interrupted or terminated"' INT TERM ERR
  run_all
else
  if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(<"$PID_FILE")"
    echo "Progress: $STATUS_FILE"
    echo "Log:      $LOG_FILE"
    exit 0
  fi
  nohup bash "$0" --foreground >>"$LOG_FILE" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  echo "Started PID: $pid"
  echo "Progress: $STATUS_FILE"
  echo "Log:      $LOG_FILE"
  echo "Follow:   tail -f $LOG_FILE"
fi
