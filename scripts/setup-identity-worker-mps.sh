#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-/opt/homebrew/bin/python3.11}
venv_path="$repo_root/.venv-identity-mps"
model_dir="$repo_root/services/identity-worker/models"

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
  -r "$repo_root/services/identity-worker/requirements-macos-mps.txt"

mkdir -p "$model_dir"
for weight in \
  prtreid-soccernet-baseline.pth.tar \
  hrnetv2_w32_imagenet_pretrained.pth
do
  if [ ! -s "$model_dir/$weight" ] && command -v docker >/dev/null 2>&1; then
    # Reuse the already checksummed model baked into the local Docker worker.
    docker compose -f "$repo_root/docker-compose.yml" \
      cp "identity-worker:/models/$weight" "$model_dir/$weight" \
      >/dev/null 2>&1 || true
  fi
done

REID_MODEL_DIRECTORY="$model_dir" \
  "$venv_path/bin/python" \
  "$repo_root/services/identity-worker/scripts/fetch_models.py"

"$venv_path/bin/python" -c \
  'import torch; assert torch.backends.mps.is_built(), "PyTorch has no MPS backend"; assert torch.backends.mps.is_available(), "Metal/MPS is unavailable"; print("PyTorch", torch.__version__, "MPS available:", torch.backends.mps.is_available())'
"$venv_path/bin/python" -m pip check
