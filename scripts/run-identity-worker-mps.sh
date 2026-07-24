#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="$repo_root/.venv-identity-mps/bin/python"

if [ ! -x "$python_bin" ]; then
  echo "Run ./scripts/setup-identity-worker-mps.sh first."
  exit 1
fi

if curl -fsS http://127.0.0.1:8095/health/ready >/dev/null 2>&1; then
  echo "PRTReID MPS is already ready on 127.0.0.1:8095."
  echo "Do not start a second copy; keep using the existing process."
  exit 0
fi
if lsof -nP -iTCP:8095 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8095 is occupied by a process that is not a ready PRTReID worker." >&2
  echo "Inspect it with: lsof -nP -iTCP:8095 -sTCP:LISTEN" >&2
  exit 1
fi

export PYTHONPATH="$repo_root/services/identity-worker"
export PRTREID_WEIGHTS="$repo_root/services/identity-worker/models/prtreid-soccernet-baseline.pth.tar"
export PRTREID_HRNET_WEIGHTS="$repo_root/services/identity-worker/models/hrnetv2_w32_imagenet_pretrained.pth"
export REID_DEVICE=mps
export REID_PRELOAD=1
export REID_BATCH_SIZE=${REID_BATCH_SIZE:-8}
export REID_CACHE_MAX_ENTRIES=${REID_CACHE_MAX_ENTRIES:-4096}
export REID_CACHE_TTL_SECONDS=${REID_CACHE_TTL_SECONDS:-86400}
export REID_CACHE_WAIT_TIMEOUT_SECONDS=${REID_CACHE_WAIT_TIMEOUT_SECONDS:-900}
# Fail closed: an unsupported Metal operator must be visible instead of being
# silently executed by the CPU while telemetry claims MPS.
export PYTORCH_ENABLE_MPS_FALLBACK=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/replay-studio-identity-matplotlib}

exec "$python_bin" -m uvicorn identity_worker_service.main:app \
  --host 127.0.0.1 \
  --port 8095 \
  --workers 1
