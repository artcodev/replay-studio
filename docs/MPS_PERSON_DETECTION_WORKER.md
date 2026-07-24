# Native MPS person-detection worker

For the complete four-tab startup, restart, reboot, cleanup, and verification
procedure, use the canonical
[`MPS_LOCAL_OPERATIONS.md`](MPS_LOCAL_OPERATIONS.md) runbook.

This worker runs one explicitly selected YOLO checkpoint on Apple GPU. It
returns raw boxes only; the API remains the sole owner of pitch-person
filtering, appearance features, annotations, tracking, calibration projection,
and cache persistence.

Frames cross the boundary as multipart binary JPEG, never base64 JSON. Worker
responses are versioned and fail closed on unknown fields. Once
`PERSON_DETECTION_WORKER_URL` is configured, an unavailable or malformed worker
is an error; reconstruction never silently switches back to CPU.

## One-time setup

```sh
./scripts/setup-person-detection-worker-mps.sh
```

## Start and verify

```sh
./scripts/run-person-detection-worker-mps.sh
curl -fsS http://127.0.0.1:8096/health/ready
```

The no-argument command selects `yolo26m.pt`. To use another model selected in
the reconstruction UI, restart this worker with that exact checkpoint:

```sh
./scripts/run-person-detection-worker-mps.sh /absolute/path/to/football.pt
```

The worker is intentionally fixed to one resident checkpoint for speed. The
API now compares the requested model with `checkpoint.name` at the start of
every reconstruction and fails clearly on a mismatch; it never silently runs
`yolo26m.pt` when the UI selected `football.pt` or `yolo26x.pt`.

The response must report `"device":"mps"` and
`"mpsFallbackEnabled":false`.
Startup performs one 1920×1080/1280px warm-up inference, so Metal graph
compilation is charged to readiness instead of the first reconstruction frame.

With the calibration worker on 8094 and identity worker on 8095 also running,
start Docker callers:

```sh
./scripts/start-mps-docker.sh --build
./scripts/check-mps-stack.sh
npm run dev
```

Check `person_detection_worker.device == "mps"`. Reconstruction persists
`providerInferenceSeconds`, `providerRequestSeconds`, provider box count, and
the exact checkpoint/runtime contract under person-detection cache
diagnostics. Cache hits correctly skip the GPU.
