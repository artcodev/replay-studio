# Native PyTorch MPS calibration worker

For the complete four-tab startup, restart, reboot, cleanup, and verification
procedure, use the canonical
[`MPS_LOCAL_OPERATIONS.md`](MPS_LOCAL_OPERATIONS.md) runbook.

This is an opt-in macOS acceleration path for PnLCalib. The normal Docker
configuration remains CPU-only. The MPS worker runs as one native host process,
loads both neural networks once at startup, and the Docker API/runners call it
through `host.docker.internal`.

The change is isolated and fail-closed: if Metal is unavailable, the worker
does not become ready; unsupported MPS operators are not silently executed on
CPU.

## One-time setup

Use native arm64 Python 3.11, not a Rosetta/x86 shell:

```sh
uname -m
./scripts/setup-calibration-worker-mps.sh
```

The setup uses `.venv-calibration-mps` and the existing pinned SoccerNet source
and PnLCalib weights. It does not change the Docker worker.

## Start

Terminal 1:

```sh
./scripts/run-calibration-worker-mps.sh
```

Wait until startup finishes, then verify:

```sh
curl -fsS http://127.0.0.1:8094/health/ready
```

Terminal 4:

```sh
./scripts/start-mps-docker.sh --build
./scripts/check-mps-stack.sh
npm run dev
```

The `docker-compose.mps.yml` override changes the calibration-worker URL used
by the API and job runners. The Docker CPU calibration service is inactive in
this profile and the start script removes instances left by older revisions.
PnLCalib’s two models are preloaded once by the single native process, and
per-frame requests do not reload them.

## Stop or return to CPU

Stop the native worker with Ctrl-C. Returning to CPU is a separate explicit
operating mode:

```sh
docker compose -f docker-compose.yml up -d --build
```

## Notes

- Keep `PNLCALIB_BATCH_SIZE=1` first. Increase it only after measuring memory
  and throughput on the actual Mac.
- The first model load is expected to be slow. Subsequent frames should report
  inference timings without model-load time.
- MPS changes floating-point kernels, so calibration acceptance can differ
  slightly from CPU. Compare frame-level residual and acceptance diagnostics
  before making it the default deployment path.
