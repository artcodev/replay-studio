#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="$repo_root/.venv-calibration-mps/bin/python"

if [ ! -x "$python_bin" ]; then
  echo "Run ./scripts/setup-calibration-worker-mps.sh first."
  exit 1
fi

if curl -fsS http://127.0.0.1:8094/health/ready >/dev/null 2>&1; then
  echo "PnLCalib MPS is already ready on 127.0.0.1:8094."
  echo "Do not start a second copy; keep using the existing process."
  exit 0
fi
if lsof -nP -iTCP:8094 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8094 is occupied by a process that is not a ready PnLCalib worker." >&2
  echo "Inspect it with: lsof -nP -iTCP:8094 -sTCP:LISTEN" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/services/calibration-worker"
export SOCCERNET_ROOT="$repo_root/.references/sn-gamestate"
export PNLCALIB_KEYPOINT_WEIGHTS="$repo_root/services/calibration-worker/models/pnl_SV_kp"
export PNLCALIB_LINE_WEIGHTS="$repo_root/services/calibration-worker/models/pnl_SV_lines"
export PNLCALIB_DEVICE=mps
export PNLCALIB_PRELOAD=1
export PNLCALIB_BATCH_SIZE=${PNLCALIB_BATCH_SIZE:-1}
# Fail closed if an unsupported Metal operator appears; never hide a CPU
# fallback behind apparently successful MPS telemetry.
export PYTORCH_ENABLE_MPS_FALLBACK=0

exec "$python_bin" -m uvicorn calibration_worker_service.main:app \
  --host 127.0.0.1 \
  --port 8094 \
  --workers 1
