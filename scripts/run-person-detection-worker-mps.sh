#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="$repo_root/.venv-person-detection-mps/bin/python"
weights_path=${1:-${PERSON_DETECTION_WEIGHTS:-"$repo_root/yolo26m.pt"}}

if [ ! -x "$python_bin" ]; then
  echo "Run ./scripts/setup-person-detection-worker-mps.sh first."
  exit 1
fi

if [ ! -f "$weights_path" ]; then
  echo "Person-detection checkpoint does not exist: $weights_path" >&2
  exit 1
fi
weights_dir=$(CDPATH= cd -- "$(dirname -- "$weights_path")" && pwd)
weights_path="$weights_dir/$(basename -- "$weights_path")"
expected_checkpoint=$(basename -- "$weights_path")

if ready_payload=$(curl -fsS http://127.0.0.1:8096/health/ready 2>/dev/null); then
  actual_checkpoint=$(
    printf '%s' "$ready_payload" |
      "$python_bin" -c 'import json, sys; print(json.load(sys.stdin)["checkpoint"]["name"])'
  )
  if [ "$actual_checkpoint" = "$expected_checkpoint" ]; then
    echo "YOLO person MPS ($actual_checkpoint) is already ready on 127.0.0.1:8096."
    echo "Do not start a second copy; keep using the existing process."
    exit 0
  fi
  echo "YOLO person MPS is already running with $actual_checkpoint, but $expected_checkpoint was requested." >&2
  echo "Stop the existing worker with Ctrl-C, then run this command again." >&2
  exit 1
fi
if lsof -nP -iTCP:8096 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8096 is occupied by a process that is not a ready YOLO worker." >&2
  echo "Inspect it with: lsof -nP -iTCP:8096 -sTCP:LISTEN" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/services/person-detection-worker"
export PERSON_DETECTION_WEIGHTS="$weights_path"
export PERSON_DETECTION_DEVICE=mps
export PERSON_DETECTION_PRELOAD=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/replay-studio-person-matplotlib}

exec "$python_bin" -m uvicorn person_detection_worker_service.main:app \
  --host 127.0.0.1 \
  --port 8096 \
  --workers 1
