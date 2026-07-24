#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-/opt/homebrew/bin/python3.11}
venv_path="$repo_root/.venv-person-detection-mps"

if [ ! -x "$python_bin" ]; then
  echo "Python 3.11 was not found at $python_bin."
  echo "Install it with: brew install python@3.11"
  exit 1
fi

"$python_bin" -c \
  'import platform; assert platform.machine() == "arm64", "Use native arm64 Python, not Rosetta"; print("Python architecture:", platform.machine())'

if [ ! -s "$repo_root/yolo26m.pt" ]; then
  echo "Missing person detector checkpoint: $repo_root/yolo26m.pt"
  exit 1
fi

"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --upgrade pip
"$venv_path/bin/python" -m pip install \
  -r "$repo_root/services/person-detection-worker/requirements-macos-mps.txt"
"$venv_path/bin/python" -c \
  'import torch; assert torch.backends.mps.is_built(), "PyTorch has no MPS backend"; assert torch.backends.mps.is_available(), "Metal/MPS is unavailable"; print("PyTorch", torch.__version__, "MPS available:", torch.backends.mps.is_available())'
"$venv_path/bin/python" -m pip check
