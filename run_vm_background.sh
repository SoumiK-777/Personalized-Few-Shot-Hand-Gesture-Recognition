#!/usr/bin/env bash
# Run the complete experiment on a Linux VM in the background.
# Optional overrides, e.g.:
# DATA_DIR=/data/gestures/cropped_images WORK_DIR=/data/handgesture-output bash run_vm_background.sh
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Prefer an explicit DATA_DIR. The second path preserves compatibility with the
# original VM layout; the project-local path is the portable default.
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d /home/soumik/soumik/misc/cropped_images ]]; then
    export DATA_DIR=/home/soumik/soumik/misc/cropped_images
  else
    export DATA_DIR="$PROJECT_DIR/gestures/cropped_images"
  fi
fi
export WORK_DIR="${WORK_DIR:-$PROJECT_DIR/vm_output}"
export SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-$WORK_DIR/split_manifest.json}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Start again with: DATA_DIR=/absolute/path/to/cropped_images bash run_vm_background.sh" >&2
  exit 1
fi

run_phase() {
  local name="$1"; shift
  echo "[$(date -Is)] START $name"
  "$@"
  (cd "$WORK_DIR" && zip -rq output.zip results figures checkpoints 2>/dev/null || true)
  echo "[$(date -Is)] DONE $name"
}

run_all() {
  cd "$PROJECT_DIR"
  echo "Project: $PROJECT_DIR"
  echo "Data:    $DATA_DIR"
  echo "Output:  $WORK_DIR"
  nvidia-smi || true
  "$PYTHON_BIN" -m pip install --quiet timm scikit-learn seaborn

  run_phase split "$PYTHON_BIN" scripts/make_train_test_split.py
  run_phase protonet "$PYTHON_BIN" proto_net.py
  run_phase siamese "$PYTHON_BIN" siamese.py
  run_phase relationnet "$PYTHON_BIN" relation_net.py
  run_phase feat "$PYTHON_BIN" feat.py
  run_phase baselines "$PYTHON_BIN" baselines/baseline_finetune.py
  run_phase tables "$PYTHON_BIN" scripts/generate_tables.py
  run_phase sanity "$PYTHON_BIN" scripts/sanity_check_results.py
  run_phase tsne "$PYTHON_BIN" visualize/tsne_embeddings.py
  run_phase confusion "$PYTHON_BIN" visualize/confusion_matrix.py
  run_phase acc_plot "$PYTHON_BIN" visualize/acc_vs_kshot.py
  run_phase baseline_plot "$PYTHON_BIN" visualize/baseline_bar_chart.py
  run_phase curves "$PYTHON_BIN" visualize/training_curves.py
  echo "[$(date -Is)] ALL DONE: $WORK_DIR/output.zip"
}

# The --foreground mode is useful for systemd/tmux. Default mode backgrounds the
# work and prints its PID/log path immediately.
if [[ "${1:-}" == "--foreground" ]]; then
  run_all
else
  LOG_FILE="$WORK_DIR/logs/full_run_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$0" --foreground >"$LOG_FILE" 2>&1 < /dev/null &
  PID=$!
  echo "Started PID: $PID"
  echo "Log file: $LOG_FILE"
  echo "Follow progress: tail -f $LOG_FILE"
fi
