#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
compose_files="-f $repo_root/docker-compose.yml -f $repo_root/docker-compose.mps.yml"
require_vite=0

if [ "${1:-}" = "--require-vite" ]; then
  require_vite=1
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: ./scripts/check-mps-stack.sh [--require-vite]" >&2
  exit 2
fi

check_ready() {
  name=$1
  url=$2
  echo "Checking $name: $url"
  curl -fsS "$url" >/dev/null
}

check_ready "PnLCalib MPS" "http://127.0.0.1:8094/health/ready"
check_ready "PRTReID MPS" "http://127.0.0.1:8095/health/ready"
check_ready "YOLO person MPS" "http://127.0.0.1:8096/health/ready"

# Include stopped containers: an old `unless-stopped` duplicate could otherwise
# reappear after Docker Desktop or the computer restarts.
# shellcheck disable=SC2086
existing_services=$(docker compose $compose_files --profile docker-cpu-workers \
  --profile docker-web ps -a --services)
for duplicate in calibration-worker identity-worker; do
  if printf '%s\n' "$existing_services" | grep -Fx "$duplicate" >/dev/null; then
    echo "ERROR: duplicate Docker CPU worker container exists: $duplicate" >&2
    exit 1
  fi
done
if printf '%s\n' "$existing_services" | grep -Fx "web" >/dev/null; then
  echo "ERROR: Docker nginx web container exists; use native Vite instead" >&2
  exit 1
fi

health_file=$(mktemp "${TMPDIR:-/tmp}/replay-studio-health.XXXXXX")
trap 'rm -f "$health_file"' EXIT HUP INT TERM
curl -fsS "http://127.0.0.1:8000/api/health" >"$health_file"

python3 - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    health = json.load(stream)

expected = {
    "calibration_worker": "mps",
    "identity_worker": "mps",
    "person_detection_worker": "mps",
}
errors = []
for capability, device in expected.items():
    value = health.get(capability)
    if not isinstance(value, dict):
        errors.append(f"{capability}: missing health payload")
        continue
    if value.get("status") != "ready":
        errors.append(f"{capability}: status={value.get('status')!r}")
    if value.get("device") != device:
        errors.append(f"{capability}: device={value.get('device')!r}")

for capability in ("identity_worker", "person_detection_worker"):
    value = health.get(capability) or {}
    if value.get("mpsFallbackEnabled") is not False:
        errors.append(
            f"{capability}: mpsFallbackEnabled="
            f"{value.get('mpsFallbackEnabled')!r}"
        )

if errors:
    print("ERROR: API is not using the strict MPS providers:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("API routing:")
for capability in expected:
    value = health[capability]
    print(
        f"  {capability}: status={value['status']} "
        f"device={value['device']} model={value.get('modelVersion')}"
    )
print("Docker duplicate worker containers: none")
print("MPS fallback: disabled")
PY

if curl -fsS "http://127.0.0.1:5188/" >/dev/null 2>&1; then
  echo "Vite web: ready at http://127.0.0.1:5188"
elif [ "$require_vite" -eq 1 ]; then
  echo "ERROR: Vite web is not ready at http://127.0.0.1:5188" >&2
  exit 1
else
  echo "Vite web: not running (start with: npm run dev)"
fi
