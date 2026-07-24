#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
compose_files="-f $repo_root/docker-compose.yml -f $repo_root/docker-compose.mps.yml"
build_flag=

if [ "${1:-}" = "--build" ]; then
  build_flag=--build
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: ./scripts/start-mps-docker.sh [--build]" >&2
  exit 2
fi

# These containers belong only to the explicit docker-cpu-workers profile.
# Remove old instances left by earlier Compose revisions so Docker Desktop
# cannot restart a second copy after reboot.
# shellcheck disable=SC2086
docker compose $compose_files --profile docker-cpu-workers \
  rm -fsv calibration-worker identity-worker

# The MPS development layout serves Vue from native Vite, not nginx.
# shellcheck disable=SC2086
docker compose $compose_files --profile docker-web rm -fsv web

# shellcheck disable=SC2086
docker compose $compose_files up -d $build_flag \
  postgres redis jersey-ocr-worker migrate \
  api reconstruction-runner pipeline-runner

echo
echo "Docker backend is started. Verify it, then start native Vite:"
echo "  ./scripts/check-mps-stack.sh"
echo "  npm run dev"
