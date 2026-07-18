# PnLCalib calibration worker

Server-side pitch calibration based on the official SoccerNet `sn-gamestate`
PnLCalib plugin. It predicts 57 semantic field keypoints and 23 semantic field
lines (plus one background channel for each model),
refines the image-to-pitch homography with both signals, and returns one metric
homography per successfully calibrated frame.

The response deliberately separates observed evidence from completed model
geometry. `keypointCount` / `detectedKeypointCount` and `inlierRatio` refer to
raw network detections; inferred points are reported separately as
`completedKeypointCount` and never increase the quality score. Every accepted
frame has at least six raw semantic inliers, an inlier ratio of at least 0.65,
and a representative error no greater than 18 px. `rawKeypoints` carries image
and pitch coordinates, inlier flags, and residuals for the Calibration QA
overlay. `confidence` is a named heuristic quality score, not a calibrated
probability.

`rawLines` exposes every detected semantic segment before PnLCalib mutates or
completes its geometry. Each item is
`{id, name, start: {x, y}, end: {x, y}, confidence, groundPlane}` in original
source-frame pixels. Goal-frame segments are retained as `groundPlane: false`
for visual/orientation evidence but excluded from the planar homography fit.
`detectedLineCount` counts all reported observations; `lineCount` counts the
ground-plane segments passed to the solver.

The Docker image pins SoccerNet Game State Reconstruction to commit
`1c958345067218297d221e45e1a6405f975f83e0`. Put the official `pnl_SV_kp`
and `pnl_SV_lines` checkpoints in `models/` before building. Their download
URLs and checksums are documented in `../../docs/CALIBRATION.md`.

From the repository root, build and verify the isolated service with:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build -d calibration-worker
curl --fail http://127.0.0.1:8090/health/live
curl --fail http://127.0.0.1:8090/health/ready
```

The explicit platform is required on Apple Silicon because the pinned
PnLCalib/PyTorch 1.13 runtime is built from the official Linux x86-64 CPU
wheel. The image intentionally installs `torch==1.13.1+cpu`, so it does not
pull the much larger CUDA runtime into a CPU-only worker.

The liveness endpoint is cheap. By default the worker preloads both models
before it starts accepting traffic, so readiness and the first calibration
request do not hide model-loading time. Set `PNLCALIB_PRELOAD=0` only when lazy
startup is explicitly preferred; readiness still loads and validates both
models and returns `503` if either checkpoint cannot be loaded.

## Internal boundaries

The HTTP root composes a narrow `CalibrationEngine` contract with five
independent capabilities: source-frame decoding, pinned PnLCalib runtime/model
loading, heatmap inference, geometry projection/quality gating, and a bounded
LRU result cache. `PnLCalibEngine` owns only batch deduplication, the
single-inference lock, and orchestration of those capabilities. Engine results
are immutable DTOs until the application service serializes them at the HTTP
boundary; no compatibility `engine.py` facade or mutable diagnostics argument
is retained.

## Latency, cache, and diagnostics

Successful and failed calibration results are cached in-process by the exact
uploaded-frame SHA-256 plus a model version derived from the inference schema,
input size, and both checkpoint identities. Repeated requests for the same
extracted frame therefore cannot reuse a result after a checkpoint change.
Identical uploads within one request are deduplicated before inference. The
cache is bounded LRU with a TTL and can be configured with:

- `PNLCALIB_CACHE_MAX_ENTRIES` (default `512`; `0` disables the cache)
- `PNLCALIB_CACHE_TTL_SECONDS` (default `3600`; `0` disables expiration)
- `PNLCALIB_PRELOAD` (default `1`)

Every `/v1/calibrate` response includes `diagnostics` with the model version,
cache hits/misses, deduplication and inference-batch counts, lock wait time,
decode and total request time, plus tensor assembly, keypoint inference, line
inference, heatmap decoding, and geometry timing. These diagnostics are also
attached to the API's per-frame calibration evidence for later inspection.

For a reproducible local single-frame benchmark, run from the repository root:

```bash
SOCCERNET_ROOT="$PWD/.references/sn-gamestate" \
PYTHONPATH=services/calibration-worker \
.venv/bin/python services/calibration-worker/scripts/benchmark_frame.py path/to/frame.jpg
```

The first run executes both models and subsequent runs demonstrate cache
latency. Add `--uncached` to execute inference on every run.
