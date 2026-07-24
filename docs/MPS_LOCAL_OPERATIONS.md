# Local macOS MPS stack: operator runbook

This is the canonical startup and recovery procedure for Replay Studio on an
Apple Silicon Mac. It keeps GPU-capable inference in three native macOS
processes, serves the Vue UI from native Vite, and keeps only the database,
API, runners, and jersey OCR in Docker.

Do not use plain `docker compose up` for this configuration. That is the
separate all-Docker CPU configuration. Always use the MPS override or the
wrapper scripts below.

## Process layout and terminal count

Use **four Terminal tabs**:

| Tab | Process | Port | Runtime |
| --- | --- | ---: | --- |
| 1 | PnLCalib points + lines | 8094 | native PyTorch MPS |
| 2 | PRTReID/BPBreID identity embeddings | 8095 | native PyTorch MPS |
| 3 | YOLO person detection | 8096 | native PyTorch MPS |
| 4 | Docker startup, then Vue/Vite | 8000/5188 | Docker + native Node |

Tabs 1–3 remain open because each worker runs in the foreground and keeps its
model resident in GPU memory. Tab 4 first starts the detached Docker backend
and then remains occupied by foreground Vite. A fifth tab is optional when you
want to follow Docker logs while all four foreground processes stay running.

The MPS Compose profile does **not** run `calibration-worker` or
`identity-worker` in Docker. `jersey-ocr-worker` remains a legitimate Docker
CPU service; it is not a duplicate of an MPS process. Person detection has no
Docker worker in this profile. The Docker nginx `web` container is also
inactive: the only web development server is Vite on port 5188.

## One-time initialization

Requirements:

- Apple Silicon Mac; `uname -m` must print `arm64`;
- Docker Desktop running;
- Homebrew Python 3.11 at `/opt/homebrew/bin/python3.11`;
- the checked-in/downloaded model files referenced by each setup script.

From the repository root, use Tab 4:

```sh
cd /Users/art/code/art-lab/footbal-match-3d
uname -m
/opt/homebrew/bin/python3.11 --version
docker compose version
npm install

./scripts/setup-calibration-worker-mps.sh
./scripts/setup-identity-worker-mps.sh
./scripts/setup-person-detection-worker-mps.sh
```

The setup scripts create three independent virtual environments:

- `.venv-calibration-mps`;
- `.venv-identity-mps`;
- `.venv-person-detection-mps`.

Run setup once. Run it again only after changing the corresponding
`requirements-macos-mps.txt`, deleting its virtual environment, changing
Python, or replacing model assets. A computer reboot does not require setup.

## First start after initialization

Open all four tabs in the repository root.

Tab 1:

```sh
cd /Users/art/code/art-lab/footbal-match-3d
./scripts/run-calibration-worker-mps.sh
```

Tab 2:

```sh
cd /Users/art/code/art-lab/footbal-match-3d
./scripts/run-identity-worker-mps.sh
```

Tab 3:

```sh
cd /Users/art/code/art-lab/footbal-match-3d
./scripts/run-person-detection-worker-mps.sh
```

This loads `yolo26m.pt`. If the reconstruction will use a different detector,
start Tab 3 with the same checkpoint selected in the editor:

```sh
./scripts/run-person-detection-worker-mps.sh /absolute/path/to/football.pt
```

Only one detector checkpoint stays resident at a time. Changing the model in
the editor therefore requires stopping Tab 3 with Ctrl-C and restarting it
with the corresponding file. Reconstruction fails before frame processing if
the requested and resident checkpoint names differ.

Wait until all three processes finish loading their models. Then, in Tab 4:

```sh
cd /Users/art/code/art-lab/footbal-match-3d
curl -fsS http://127.0.0.1:8094/health/ready
curl -fsS http://127.0.0.1:8095/health/ready
curl -fsS http://127.0.0.1:8096/health/ready

./scripts/start-mps-docker.sh --build
./scripts/check-mps-stack.sh
npm run dev
```

Use `--build` for the first start and after source/dependency changes that must
enter backend Docker images. Vite prints its ready URL and stays in the
foreground. Open the UI at
[http://localhost:5188](http://localhost:5188).

## Normal daily start

If the three virtual environments and Docker images already exist:

1. Start Docker Desktop and wait until it reports that the engine is running.
2. Start Tabs 1–3 with the same three `run-*-mps.sh` commands.
3. Wait for all three readiness endpoints.
4. In Tab 4 run:

```sh
./scripts/start-mps-docker.sh
./scripts/check-mps-stack.sh
npm run dev
```

The start script removes stale Docker CPU copies of PnLCalib and PRTReID before
starting the callers. It also removes a stale Docker nginx `web` container.
It does not delete database, Redis, or media volumes.

## After restarting the computer

No setup and normally no Docker rebuild are required.

1. Start Docker Desktop.
2. Start the three native workers in Tabs 1–3.
3. Wait for ports 8094, 8095, and 8096 to become ready.
4. Run `./scripts/start-mps-docker.sh` in Tab 4.
5. Run `./scripts/check-mps-stack.sh`.
6. Run `npm run dev` in Tab 4 and open port 5188.

Docker Desktop may restore previously running containers before the native
workers are ready. The configured providers fail closed, so inference cannot
silently move to CPU. The start script still must be run after the native
workers become ready: it removes old duplicate containers and recreates the
canonical MPS routing.

Before a planned reboot, first make sure there is no active calibration or
reconstruction. Stop Vite with Ctrl-C in Tab 4, then stop the Docker callers:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  stop api reconstruction-runner pipeline-runner
```

Then stop Tabs 1–3 with Ctrl-C. Do not use `docker compose down -v`: `-v`
deletes the named PostgreSQL, Redis, and media volumes.

## Restart scenarios

### Restart only the Docker application

Keep Tabs 1–3 running. Stop Vite with Ctrl-C in Tab 4, restart the backend,
then start Vite again:

```sh
./scripts/start-mps-docker.sh
./scripts/check-mps-stack.sh
npm run dev
```

Add `--build` after changes to API source, backend dependencies, or
Dockerfiles:

```sh
./scripts/start-mps-docker.sh --build
```

Vue/TypeScript/CSS source changes are handled by Vite HMR and do not require a
Docker rebuild.

### Restart only Vite

Press Ctrl-C in Tab 4 and run:

```sh
npm run dev
```

After changing `package.json` or `package-lock.json`, run `npm install` before
starting Vite again.

### Restart one native MPS worker

Do this only when no calibration or reconstruction is actively using it:

1. Press Ctrl-C in that worker's tab.
2. Run its `run-*-mps.sh` command again in the same tab.
3. Wait for its `/health/ready`.
4. Run `./scripts/check-mps-stack.sh`.

Docker does not need to be restarted because the host port remains unchanged.
An in-flight request is expected to fail visibly; it never falls back to the
Docker CPU implementation.

### Restart everything

1. Stop or cancel active jobs in the UI.
2. Stop Vite in Tab 4 and Tabs 1–3 with Ctrl-C.
3. Stop the Docker backend:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  stop api reconstruction-runner pipeline-runner
```

4. Start Tabs 1–3 again and wait for readiness.
5. Run:

```sh
./scripts/start-mps-docker.sh
./scripts/check-mps-stack.sh
npm run dev
```

## What the health check proves

Run:

```sh
./scripts/check-mps-stack.sh
```

It fails unless all of the following are true:

- native ports 8094, 8095, and 8096 answer their readiness endpoints;
- the API reports `device=mps` for calibration, identity, and person detection;
- identity and person detection report `mpsFallbackEnabled=false`;
- no Docker `calibration-worker` or `identity-worker` container exists, even
  in the stopped state;
- no Docker nginx `web` container exists.

The normal check treats Vite as optional so it can run immediately before
`npm run dev`. From an optional fifth tab, require the complete UI too:

```sh
./scripts/check-mps-stack.sh --require-vite
```

For manual inspection:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml ps
curl -fsS http://127.0.0.1:8000/api/health
```

The Docker list should contain:

- `api`, `reconstruction-runner`, `pipeline-runner`;
- `postgres`, `redis`, `migrate` when applicable;
- `jersey-ocr-worker`;
- **no** `web`;
- **no** `calibration-worker`;
- **no** `identity-worker`.

The API health response must contain:

```text
calibration_worker.status = ready
calibration_worker.device = mps
identity_worker.status = ready
identity_worker.device = mps
identity_worker.mpsFallbackEnabled = false
person_detection_worker.status = ready
person_detection_worker.device = mps
person_detection_worker.mpsFallbackEnabled = false
```

## Logs

Native model logs stay visible in Tabs 1–3 and Vite logs in Tab 4. Docker logs
can be followed from an optional fifth tab:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  logs -f api reconstruction-runner pipeline-runner
```

To inspect only current container state:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml ps
```

## Troubleshooting

### A native port is already occupied

Do not start a second copy. Find the owner first:

```sh
lsof -nP -iTCP:8094 -sTCP:LISTEN
lsof -nP -iTCP:8095 -sTCP:LISTEN
lsof -nP -iTCP:8096 -sTCP:LISTEN
```

If the process is one of the three expected workers and its readiness endpoint
works, keep using it. Otherwise stop that exact process deliberately before
starting the canonical script. The three `run-*-mps.sh` scripts perform this
check before loading a model: an already-ready worker is reused, while an
unknown listener fails immediately with the relevant `lsof` command.

### A duplicate Docker worker is reported

The normal start script removes it. The equivalent explicit cleanup is:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  --profile docker-cpu-workers \
  rm -fsv calibration-worker identity-worker
./scripts/check-mps-stack.sh
```

This removes only the two obsolete containers. It does not remove images,
models, or named data volumes.

The startup script also removes the obsolete nginx UI. Its explicit equivalent
is:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  --profile docker-web rm -fsv web
```

### Host readiness works but API health is unavailable

Check Docker and caller routing:

```sh
docker compose -f docker-compose.yml -f docker-compose.mps.yml ps
docker compose -f docker-compose.yml -f docker-compose.mps.yml \
  logs --tail 100 api
```

All three caller services use `host.docker.internal` in
`docker-compose.mps.yml`. Do not replace those URLs with `127.0.0.1`: inside a
container, `127.0.0.1` refers to that container itself.

### Vite does not start

Confirm that dependencies are installed and port 5188 is free:

```sh
npm install
lsof -nP -iTCP:5188 -sTCP:LISTEN
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000` by default. Check the backend
with `./scripts/check-mps-stack.sh` before debugging the UI proxy.

### Return intentionally to the all-Docker CPU stack

Stop Tabs 1–3, then start the base Compose file without the MPS override:

```sh
docker compose -f docker-compose.yml up -d --build
```

This is a separate explicit operating mode, not an automatic fallback. To
return to MPS, start all three native workers and run
`./scripts/start-mps-docker.sh`.
