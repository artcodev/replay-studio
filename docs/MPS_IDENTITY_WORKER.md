# Native PyTorch MPS identity worker

For the complete four-tab startup, restart, reboot, cleanup, and verification
procedure, use the canonical
[`MPS_LOCAL_OPERATIONS.md`](MPS_LOCAL_OPERATIONS.md) runbook.

The PRTReID/BPBreID model used to associate detections with a persistent
player identity can run on Apple GPU through PyTorch MPS. Docker Desktop does
not pass Metal through to Linux containers, so this worker runs as one native
macOS process and the Docker API/runners call it through
`host.docker.internal`.

The path is opt-in and fail-closed:

- `REID_DEVICE=mps` is validated at startup;
- the worker refuses readiness when MPS is unavailable;
- `PYTORCH_ENABLE_MPS_FALLBACK=0` prevents an unsupported operator from
  silently running on CPU;
- the model is loaded once and kept resident;
- readiness and reconstruction diagnostics record `device`, PyTorch version,
  batch size, request time, and provider inference time.

## One-time setup

```sh
uname -m
./scripts/setup-identity-worker-mps.sh
```

The setup creates `.venv-identity-mps`. It first tries to copy the already
checksummed official weights from the local Docker identity worker, then
downloads them from the pinned official source only when necessary.

## Start

Keep the calibration MPS worker running on port 8094. In another terminal:

```sh
./scripts/run-identity-worker-mps.sh
```

The first startup loads a 378 MB PRTReID checkpoint and a 158 MB HRNet
checkpoint. Wait for:

```sh
curl -fsS http://127.0.0.1:8095/health/ready
```

The response must contain:

```json
{
  "status": "ready",
  "device": "mps",
  "mpsFallbackEnabled": false
}
```

Then start the Docker callers with the canonical MPS wrapper:

```sh
./scripts/start-mps-docker.sh --build
./scripts/check-mps-stack.sh
npm run dev
```

Verify the route actually used by Docker:

```sh
curl -fsS http://127.0.0.1:8000/api/health
```

`identity_worker.device` must be `mps`. During a reconstruction,
the persisted identity diagnostics expose `workerRuntime.device`,
`providerInferenceSeconds`, and `requestSeconds`. Cached crops legitimately do
not invoke the GPU; clear or change the identity cache only when measuring a
cold benchmark.

## Stop or return to CPU

Stop the native process with Ctrl-C. Returning to CPU is a separate explicit
operating mode:

```sh
docker compose -f docker-compose.yml up -d --build
```

Start with `REID_BATCH_SIZE=8`. Larger batches can improve throughput but must
be measured against Metal memory pressure on the actual Mac.
