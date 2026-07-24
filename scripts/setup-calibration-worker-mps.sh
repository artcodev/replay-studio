#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-/opt/homebrew/bin/python3.11}
venv_path="$repo_root/.venv-calibration-mps"

if [ ! -x "$python_bin" ]; then
  echo "Python 3.11 was not found at $python_bin."
  echo "Install it with: brew install python@3.11"
  exit 1
fi

"$python_bin" -c \
  'import platform; assert platform.machine() == "arm64", "Use native arm64 Python, not Rosetta"; print("Python architecture:", platform.machine())'

"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --upgrade pip
"$venv_path/bin/python" -m pip install \
  -r "$repo_root/services/calibration-worker/requirements-macos-mps.txt"

if [ ! -d "$repo_root/.references/sn-gamestate/plugins/calibration/pnlcalib" ]; then
  echo "Missing pinned SoccerNet source at .references/sn-gamestate."
  echo "Restore the repository reference before starting the worker."
  exit 1
fi

for weight in pnl_SV_kp pnl_SV_lines; do
  if [ ! -s "$repo_root/services/calibration-worker/models/$weight" ]; then
    echo "Missing PnLCalib weight: services/calibration-worker/models/$weight"
    exit 1
  fi
done

"$venv_path/bin/python" -c \
  'import torch; assert torch.backends.mps.is_built(), "PyTorch has no MPS backend"; assert torch.backends.mps.is_available(), "Metal/MPS is unavailable"; print("PyTorch", torch.__version__, "MPS available:", torch.backends.mps.is_available())'
