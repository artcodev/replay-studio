# Ball detection and trajectory reconstruction

The ball pipeline is intentionally separate from person detection. A football
is a much smaller and faster target than a player, so reusing the player/
calibration COCO pass as the primary source loses observations and encourages false tracks. The
current pipeline decodes its own dense frames, keeps several detector
hypotheses per frame, and resolves one trajectory over the complete shot.

This document describes the implemented contract. It is not a claim that a
single broadcast camera recovers the ball's true 3D height: the published
position is a calibrated ground-plane game-state position with an explicit
rendering-height placeholder.

## Architecture

```text
private source clip + exact scene range
                 |
                 v
    dense source-resolution frame cache
       (up to 25 FPS, independent of people)
                 |
                 v
      selected detector backend
      + explicit fallback metadata
                 |
                 v
       top-K image-space candidates
        + detector provenance per frame
                 |
                 v
 bounded bracketing calibration interpolation
       + interpolated camera compensation
                 |
                 v
 global beam/Viterbi trajectory resolver
     observed / inferred / occluded states
                 |
                 v
 payload.ball.keyframes + diagnostics + QA
```

The implementation boundaries are:

- `apps/api/app/ball_frames.py` — deterministic dense-frame extraction and
  cache;
- `apps/api/app/ball_detection_contract.py` and
  `apps/api/app/ball_detector_factory.py` — provider-neutral contract and
  explicit backend selection;
- `apps/api/app/ball_detection_configuration.py` — immutable queued backend,
  checkpoint and failure-policy input;
- `apps/api/app/reconstruction_ball_detector_selection.py` — reconstruction
  detector/fallback construction from that immutable input;
- `apps/api/app/ultralytics_ball_detector.py` and
  `apps/api/app/wasb_ball_detector.py` — provider-specific inference adapters;
- `services/ball-worker/` — isolated pinned WASB-SBDT runtime;
- `apps/api/app/reconstruction_ball_detection.py` — dense source/cache,
  checkpoint and per-frame attempt orchestration;
- `apps/api/app/reconstruction_ball_roi.py` — adaptive ROI and reacquisition
  policy;
- `apps/api/app/reconstruction_ball_projection_contract.py` — immutable
  dense-frame projection input;
- `apps/api/app/reconstruction_bounded_homography.py` and
  `apps/api/app/reconstruction_dense_ball_projection_context.py` — bounded
  homography interpolation and auditable temporal camera/calibration choice;
- `apps/api/app/reconstruction_ball_candidate_projection.py` and
  `apps/api/app/reconstruction_ball_projection_status.py` — candidate metric
  materialization and final publication status;
- `apps/api/app/ball_tracking_contract.py` and
  `apps/api/app/ball_tracking_candidates.py` — strict resolver inputs and
  candidate normalization;
- `apps/api/app/ball_tracking_solver.py` — cohesive beam/Viterbi path solve;
- `apps/api/app/ball_trajectory_projection.py` and
  `apps/api/app/ball_trajectory_materialization.py` — metric projection,
  editor keyframes and diagnostics;
- `apps/api/app/ball_tracking.py` — thin normalize → solve → materialize
  orchestration;
- `apps/api/app/reconstruction_ball_phase.py` — reconstruction phase
  composition and publication payload;
- `apps/api/app/reconstruction_ball_trajectory.py` — pure automatic/manual
  scene trajectory rules;
- `apps/api/app/reconstruction_ball_trajectory_command.py` — artifact and
  scene-repository publication for an editor trajectory command;
- `apps/api/app/quality_measurements.py` — typed reconstruction measurements.
- `apps/api/app/quality_metric_report.py` — stable QA metric contract.
- `apps/api/app/quality_gate_report.py` — versioned QA gate assessment.

## Detector backends

The editor exposes the same three backend identifiers accepted by the API.

| UI label | API value | Intended use | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| Roboflow · tiled | `dedicated-ultralytics` | Default | One-class football checkpoint, native-resolution overlapping tiles, batched tile inference, duplicate suppression | Single-frame model; CPU inference is expensive; a visually ball-like object can still be proposed |
| WASB · temporal | `wasb-service` | Accuracy challenger | Official three-frame soccer-ball heatmap model, temporal evidence, isolated and checksum-verified runtime | Optional worker and checkpoint are required; CPU is slow; heatmap confidence is not interchangeable with YOLO confidence |
| COCO · fallback | `generic-ultralytics` | Explicit failure recovery and diagnostics only | Uses the already-loaded person model and requires no extra worker/checkpoint | Generic COCO class 32 is materially weaker for a tiny broadcast football and should not be the accuracy baseline |

### Dedicated Roboflow/Ultralytics backend

`dedicated-ultralytics` loads
`apps/api/models/football-ball-detection.pt`, accepts one-class detections
(class `0`), and evaluates overlapping 640 x 640 tiles in batches. Candidate
boxes are translated back into full-frame pixel coordinates, then global NMS
removes tile-boundary duplicates. The default confidence floor is deliberately
low (`0.05`): ambiguous detections are retained for the temporal resolver
instead of being accepted as the trajectory immediately.

The locally provisioned checkpoint has this SHA-256:

```text
678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912
```

Verify it with:

```bash
shasum -a 256 apps/api/models/football-ball-detection.pt
```

The detector does not download a missing model implicitly. A missing or
unreadable checkpoint therefore remains an observable configuration error.

Dense dedicated inference uses a deterministic adaptive schedule. Frames `0`,
the final frame, every fifth frame, and every declared camera-cut boundary run
the full tiled scan. Intermediate frames rerun up to three exact source tiles
that produced the strongest candidates on the previous frame. Reusing the
original tile is important: this checkpoint can produce a different result for
a centred crop even when that crop still contains the ball. Strongly
overlapping candidate tiles are deduplicated; seeds without tile provenance use
a shifted 640-pixel window that retains full context at image borders.

An empty ROI result triggers a full tiled scan on the same timestamp. A
non-empty false-positive ROI can still hide a true ball until the next periodic
scan, so the default worst-case blind window is four frames. Set
`BALL_DETECTION_FULL_SCAN_INTERVAL=1` to disable adaptive ROI and restore a full
scan on every frame for gold-set validation. This strategy applies only to
`dedicated-ultralytics`; generic COCO and WASB keep their original contracts.
Per-frame `scanMode` and top-level `adaptiveRoi` diagnostics report global/ROI
frame counts, crop counts, same-frame reacquisitions, the estimated all-global
baseline, and crop reduction ratio.

### WASB temporal challenger

`wasb-service` uses the offline window `[previous, current, next]` and requests
the centred output channel. At shot boundaries the missing neighbour is an
explicit repeat of the edge frame; the worker reports this as
`temporalPadding`, and padding is not counted as additional observed evidence.
The pinned implementation preserves the upstream three-frame HRNet
architecture, 512 x 288 affine preprocessing, ImageNet normalization, sigmoid
heatmaps, and weighted connected-component centers.

The worker separates liveness from readiness:

- `GET /health/live` proves that HTTP is serving;
- `GET /health/ready` verifies the checkpoint checksum, imports the pinned
  model source, strictly loads every state-dict key, and moves the model to the
  configured device;
- `POST /v1/detections` is the only inference contract. The API uploads image
  bytes as multipart files and sends a strict JSON manifest. Repeated manifest
  `fileIndex` values express temporal edge padding without duplicating image
  bytes.

The pinned WASB checkpoint SHA-256 is:

```text
d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c
```

See `services/ball-worker/README.md` for the complete worker contract.

### Explicit fallback behavior

Fallbacks are never presented as results from the requested model:

1. With `BALL_DETECTION_FAILURE_POLICY=fallback`, a failed WASB request crosses
   once to the dedicated Roboflow detector and records the reason. A circuit
   breaker keeps later frames on that explicit fallback instead of repeating
   the same worker timeout hundreds of times.
2. A failed dedicated detector may cross to generic COCO under the same policy.
3. If the configured fallback also fails, a sampled generic COCO observation is mapped
   to at most one nearest dense frame within half a dense-frame interval; it is
   never duplicated across the clip. It is tagged `generic-coco-fallback`.
4. If native dense frames cannot be decoded, the sampled reconstruction frames
   are used and `frameSource.source` becomes `sampled-frame-fallback`.

Every degraded route is visible in reconstruction warnings, `backendCounts`,
per-frame metadata, `fallbackFrameCount`, `failedFrameCount`, and the circuit
breaker record. Set `BALL_DETECTION_FAILURE_POLICY=raise` while validating a
model: the first detector failure then fails the reconstruction and preserves
the previous successful scene instead of substituting another detector.

## Dense source-frame cache

The ball decoder reads the private source clip over the exact scene
`sourceStart` / `sourceEnd` range. Its target cadence is:

```text
min(source video FPS, BALL_ANALYSIS_FRAME_RATE)
```

The default maximum is 25 FPS. Frames remain at source resolution and are
encoded as high-quality JPEGs; the player/calibration samples (now sampled at
the source frame rate up to 25 FPS) are a separate pass and are not changed.

The cache lives below the media asset:

```text
<MEDIA_ROOT>/<asset-id>/ball-frames/<cache-key>/
```

The key includes source name, byte size, modification time, scene range, and
requested FPS. A manifest and expected frame count make partial decoded-frame
caches invalid.
Extraction first writes to a temporary directory and renames it only after a
complete FFmpeg run, so an interrupted decode is not reused. Rebuilding the
same inputs reuses decoded frames. A persistent per-key `flock` serializes the
recheck/extract/publish boundary across API processes. Lock files remain in
place intentionally to avoid inode races.

Detector inference has a second, independent cache:

```text
<MEDIA_ROOT>/<asset-id>/ball-detections/<cache-key>.json
<MEDIA_ROOT>/<asset-id>/ball-detections/<cache-key>.partial.json
```

Its key contains the dense-frame cache key, cache schema, and the complete
detector input fingerprint, including checkpoint identity and tiling/confidence
settings. The `.json` artifact remains the publishable complete cache: only a
run produced entirely by the requested primary backend may populate it. A run
containing any failed or fallback frame is never published as complete, so a
temporary WASB outage cannot become sticky.

The separate `.partial.json` artifact is a resumable, clean contiguous prefix,
not a valid completed result. It is atomically refreshed every
`BALL_DETECTION_CHECKPOINT_INTERVAL` clean primary frames (four by default).
After cancellation or process interruption, an exact contract and timestamp
match resumes inference after that prefix. A failed/fallback frame stops the
prefix from growing, and successful publication of the complete cache removes
the partial artifact. Both artifacts store post-NMS image-space candidates, not
the resolved trajectory or pitch projection: calibration changes, annotations,
and resolver changes are always re-evaluated on rebuild. Writes use a temporary
file, `fsync`, and atomic replacement; corrupt or stale files are treated as
cache misses. Complete and partial publication also share a persistent
per-contract `flock`: concurrent retries may extend a prefix, but a late fenced
worker cannot replace it with fewer frames or recreate partial state after the
complete cache exists.

On the bundled 1920 x 1080 Shot 01 sample, the first 102-frame tiled CPU pass
takes roughly 4–5 minutes on the development machine. An identical rebuild
reuses raw candidates and completes the ball-detection phase in under a second.
Reducing the 25 FPS cadence or tile overlap would be faster, but changes recall
and should be done only against labelled validation data.

## Global temporal resolver

A high detector score on one frame is not sufficient evidence. The resolver
keeps the strongest candidates per frame and performs a beam-pruned Viterbi
search over the full shot. Its cost combines:

- negative log detector confidence;
- beginning, continuing, and ending an occlusion;
- motion and acceleration consistency;
- soft penalties for physically implausible speed or acceleration;
- reacquisition and long-gap penalties.

Motion coordinates are selected in this order:

1. calibrated pitch coordinates (`pitchX`, `pitchZ`);
2. camera-stabilized image coordinates;
3. raw image coordinates.

Metric evidence therefore wins when calibration exists, while difficult
frames still produce inspectable image-space diagnostics. The default soft
limits are 55 m/s and 180 m/s² in metric space, or 1800 px/s and 8000 px/s²
when only image coordinates are available. A violation increases path cost; it
does not delete the evidence.

The resolver requires at least two observed frames and a peak detection
confidence of at least `0.12`. If that evidence is absent, it returns
`no-stable-trajectory` instead of manufacturing a path.

### States

| State | Meaning | Published in `ball.keyframes` |
| --- | --- | --- |
| `observed` | A detector candidate selected by the global path | Yes |
| `inferred` | A gap bounded by observations on both sides and no longer than 0.8 seconds | Yes, with decayed confidence and increasing uncertainty |
| `occluded` | Before the first observation, after the last observation, or a gap too long to interpolate honestly | No; retained in `ball.diagnostics.path` |

An inferred keyframe has no `detectionConfidence`. Its provenance names the
left and right source candidates. An observed keyframe preserves both
`detectionConfidence` and the motion-adjusted `trajectoryConfidence`, plus the
selected candidate ID, rank, image position, detector metadata, calibration
frame, projection source, and uncertainty.

The public `y` value is currently `0.22` metres and is marked
`heightSource: rendering-placeholder`. Monocular broadcast video does not
provide reliable airborne height without an additional model or view.

## Calibration boundary

Detector output is image-space evidence. A trustworthy 3D replay additionally
requires an accepted image-to-pitch calibration. An interior dense frame uses
bounded interpolation only when both adjacent samples passed QA, the interval
is at most 0.25 seconds, frame sizes agree, and the intervening camera-motion
edge is reliable. Homography and camera-transform matrices are normalized and
revalidated. Every unsafe case becomes an explicit nearest-sample fallback;
its reason, sample indices, alpha, and uncertainty remain in provenance.

When calibration QA rejects a reconstruction, the system keeps detector
candidates and temporal diagnostics for inspection but publishes no new
world-space ball keyframes. This distinction is important when diagnosing a
ball on the wrong side of the pitch: inspect calibration and orientation before
tuning the ball model.

## Persisted evidence

The reconstruction stores detector-level information under:

```text
payload.videoAsset.reconstruction.ballDetection
```

Important fields include:

- `status`: `ready` or `degraded`;
- `requestedBackend` and the complete input/configuration fingerprint;
- `frameSource`: cache key, cadence, scene range, cache hit, fallback reason,
  detector-checkpoint hit, and resumed/checkpointed frame counts;
- `frameCount`, `candidateCount`, and `framesWithCandidates`;
- `backendCounts`, showing the effective backend on every frame;
- observed, inferred, and occluded counts and coverage;
- `frames[]`: per-frame backend, candidate count, image size, fallback reason,
  and backend metadata;
- `tracking`: a copy of the global resolver diagnostics.

The trajectory and its canonical diagnostics are stored under:

```text
payload.ball.keyframes
payload.ball.diagnostics
```

`payload.ball.diagnostics.path` is the most useful frame-by-frame audit: it
contains the selected state, candidate ID, candidate count, costs, motion
source, and violation flags. `selectedCandidateIds` can be joined back to the
detector provenance without relying on array position.

The beam search stores shared immutable backpointers and materializes the full
path once for the winner, avoiding quadratic path-prefix copying on long clips.

## QA metrics

The generic reconstruction QA report includes these ball-specific metrics:

| Metric | What it answers | Initial thresholds |
| --- | --- | --- |
| `ballObservedCoverage` | How much of the dense timeline is backed by detector evidence? | pass >= 0.65; review >= 0.35 |
| `ballPublishedCoverage` | How much is observed or bounded-interpolated? | pass >= 0.85; review >= 0.60 |
| `ballLongestUnresolvedGap` | What is the longest inferred/occluded run? | Diagnostic; no verdict threshold yet |
| `ballPathCostMargin` | How much better is the selected global path than the runner-up? | Diagnostic; larger is less ambiguous |
| `ballSpeedViolationRatio` | How many published observed motion segments exceed the QA limit? | QA limit 50 m/s; pass <= 0.01; review <= 0.05 |

Coverage gates are currently non-required because a real ball can be
out-of-frame or genuinely occluded. They are still essential for model
comparison and should be aggregated separately for visible-ball and
not-visible-ball ground truth. The QA speed threshold is intentionally stricter
than the resolver's soft 55 m/s transition limit.

For a useful gold set, label at least:

- ball-visible versus ball-not-visible for every frame;
- ball center when visible, including blur and partial occlusion;
- shot/camera cut boundaries;
- accepted calibration and visible pitch side;
- possession events, kicks, and goalmouth congestion;
- hard negatives such as socks, line intersections, heads, ad boards, and
  specular highlights.

Track detector recall/precision on visible frames separately from trajectory
coverage, center error in pixels, pitch error in metres, identity switches
between hypotheses, longest false interpolation, and runtime per source
second. Dataset metrics such as GS-HOTA remain complementary; the engineering
gates here protect runtime behavior rather than replacing benchmark metrics.

## Configuration

The API reads these variables from `.env` through `apps/api/app/config.py`.

| Variable | Default | Effect |
| --- | --- | --- |
| `BALL_DETECTION_BACKEND` | `dedicated-ultralytics` | Default backend for new/rebuilt scenes |
| `BALL_DETECTION_MODEL` | `./apps/api/models/football-ball-detection.pt` | Dedicated checkpoint path |
| `BALL_DETECTION_CONFIDENCE` | `0.05` | Dedicated candidate confidence floor |
| `BALL_DETECTION_IMAGE_SIZE` | `640` | Ultralytics inference size per tile |
| `BALL_DETECTION_TILE_SIZE` | `640` | Square tile size in source pixels |
| `BALL_DETECTION_TILE_OVERLAP` | `0.20` | Fractional overlap between adjacent tiles |
| `BALL_DETECTION_INFERENCE_BATCH_SIZE` | `8` | Tiles submitted per Ultralytics inference batch |
| `BALL_DETECTION_CHECKPOINT_INTERVAL` | `4` | Clean primary frames between resumable detector-prefix checkpoints |
| `BALL_DETECTION_FULL_SCAN_INTERVAL` | `5` | Dedicated-backend frames between mandatory full tiled scans; `1` disables ROI optimization |
| `BALL_DETECTION_ROI_REGION_COUNT` | `3` | Maximum previous-candidate tiles evaluated on an intermediate frame |
| `BALL_DETECTION_ROI_PADDING` | `320` | Half-size of the 640px fallback ROI for seeds without source-tile provenance |
| `BALL_DETECTION_NMS_IOU` | `0.10` | Full-frame duplicate suppression threshold |
| `BALL_DETECTION_MAX_CANDIDATES` | `12` | Maximum detector candidates retained per frame |
| `BALL_ANALYSIS_FRAME_RATE` | `25` | Maximum dense decode FPS |
| `BALL_WASB_WORKER_URL` | `http://127.0.0.1:8092/v1/detections` | Canonical WASB multipart endpoint |
| `BALL_WASB_TIMEOUT` | `120` | Per-request timeout in seconds |
| `BALL_DETECTION_FAILURE_POLICY` | `fallback` | Explicit fallback with a circuit breaker, or strict `raise` |
| `RECONSTRUCTION_DEVICE` | `cpu` | Device passed to the local dedicated/generic Ultralytics backends |

The WASB worker has its own isolated runtime configuration:

| Variable | Default | Effect |
| --- | --- | --- |
| `WASB_DEVICE` | `cpu` | `cpu`, `cuda:0`, or another supported Torch device |
| `WASB_PRELOAD` | `1` | Load and validate the model during startup |
| `WASB_WEIGHTS` | `models/wasb-soccer-best.pth.tar` | Checkpoint path |
| `WASB_WEIGHTS_SHA256` | pinned project checksum | Expected checkpoint checksum |
| `WASB_HRNET_SOURCE` | `.references/WASB-SBDT/src/models/hrnet.py` | Pinned upstream model source |
| `WASB_SCORE_THRESHOLD` | `0.5` | Heatmap/component threshold |

Worker request safety limits (`WASB_MAX_FRAME_BYTES`,
`WASB_MAX_FRAME_PIXELS`, `WASB_MAX_BATCH_FRAMES`, and
`WASB_MAX_CANDIDATES`) can be tightened for a shared deployment.

## Starting and selecting a backend

### Default dedicated detector

No ball worker is required. Verify that the dedicated checkpoint exists, then
start the normal API and web development servers:

```bash
. .venv/bin/activate
uvicorn app.main:app --app-dir apps/api
npm run dev
```

Open the reconstruction menu, choose **Ball detector → Roboflow · tiled**, and
run **Reconstruct scene**. The backend selection is stored with the reconstruction
request and contributes to the input fingerprint, so changing it schedules a
real rebuild rather than reusing incompatible output.

### Local WASB worker

From the repository root:

```bash
PYTHONPATH=services/ball-worker WASB_DEVICE=cpu \
  .venv/bin/uvicorn ball_worker_service.main:app --host 127.0.0.1 --port 8092
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8092/health/live
curl --fail http://127.0.0.1:8092/health/ready
```

Choose **Ball detector → WASB · temporal** and reconstruct the scene. The main API health
response also reports the optional worker readiness.

### Docker Compose

Start only the challenger worker with:

```bash
docker compose --profile challengers up --build -d ball-worker
curl --fail http://127.0.0.1:8092/health/ready
```

Or run the full stack with WASB selected:

```bash
BALL_DETECTION_BACKEND=wasb-service \
  docker compose --profile challengers up --build
```

The API does not depend on WASB health because WASB is an optional challenger.
This preserves the editor when the worker is unavailable while health and
reconstruction metadata expose the degraded state.

## Performance tuning

At 25 FPS, a 20-second shot contains roughly 500 dense frames. A 1920 x 1080
frame requires several overlapping 640-pixel tiles, so local CPU inference can
take minutes even though frame decoding is cached.

Tune in this order:

1. Use a CUDA-capable device for either backend when available. Set
   `RECONSTRUCTION_DEVICE` for Ultralytics or `WASB_DEVICE` for the worker.
2. Increase `BALL_DETECTION_INFERENCE_BATCH_SIZE` only while GPU/CPU memory has
   headroom. It batches tiles, not source frames.
3. Reduce `BALL_ANALYSIS_FRAME_RATE` to 15–20 for previews or low-motion clips;
   validate the recall change on visible-ball ground truth.
4. Reduce tile overlap only after measuring misses at tile borders.
5. Do not raise the confidence floor merely to make processing look cleaner;
   the resolver needs alternate low-confidence hypotheses in difficult frames.

The first request includes model load/warm-up. Report warm and cold latency
separately. A dense-frame cache hit eliminates FFmpeg decoding; a complete
detector-cache hit eliminates model inference, while a partial checkpoint skips
only its validated clean prefix. WASB CPU is suitable for correctness checks; a
GPU-backed service is the intended path for interactive challenger comparisons.

## Troubleshooting

| Symptom | Inspect | Action |
| --- | --- | --- |
| Progress stays on dense ball analysis | Current frame/total and effective backend in the phase detail | On CPU, lower `BALL_ANALYSIS_FRAME_RATE`; for dedicated inference lower tile batch size if memory is saturated; prefer GPU for normal use |
| `No stable ball trajectory was found` | `candidateCount`, `framesWithCandidates`, peak confidence, and `payload.ball.diagnostics.status` | Confirm the ball is visible in at least two frames; compare dedicated and WASB; avoid hiding evidence with a high confidence floor |
| WASB was selected but `backendCounts` shows dedicated | Per-frame metadata, `circuitBreaker`, and `/health/ready` | Verify mounts, checksum, device support, and `BALL_WASB_WORKER_URL`; use `raise` to stop on the first WASB error |
| `sampled-frame-fallback` | `frameSource.fallbackReason` | Restore the private source clip, install FFmpeg, and verify the scene source range |
| Candidates exist but there is no 3D ball | `worldProjectionStatus`, calibration verdict, line alignment, and reprojection metrics | Fix/confirm calibration; image-space detections are deliberately not published as fabricated world coordinates |
| Ball appears on the wrong pitch side | Calibration evidence and visible-side/attack-direction settings | Correct calibration or orientation first; detector pixel coordinates alone cannot determine pitch side |
| Sudden jumps between ball-like objects | `diagnostics.path`, motion source, speed/acceleration violations, and `pathCostMargin` | Check camera stabilization and calibration; compare backend candidates; add the clip to the hard-negative gold set |
| False positives near lines, socks, or faces | Per-frame candidates and selected candidate ID | Keep top-K evidence but label hard negatives for model evaluation; tune confidence/NMS only against measured recall and precision |
| Out-of-memory during tiled inference | Device memory and tile batch size | Lower `BALL_DETECTION_INFERENCE_BATCH_SIZE`; lowering FPS reduces total work but not per-batch memory |

## Verification

Run the focused API and worker suites after changing this pipeline:

```bash
.venv/bin/pytest -q apps/api/tests/test_ball_detection.py
.venv/bin/pytest -q apps/api/tests/test_ball_frames.py
.venv/bin/pytest -q apps/api/tests/test_ball_tracking.py
.venv/bin/pytest -q services/ball-worker/tests
```

For an end-to-end check, rebuild a short continuous shot, then confirm:

- `ballDetection.frameSource.source` is `dense-source-cache`;
- `backendCounts` matches the selected detector or explains every fallback;
- observed candidates retain `sourceCandidateId` and provenance;
- `worldProjectionStatus` is `published` only with accepted calibration;
- inferred runs are bounded by observations and remain at most 0.8 seconds;
- occluded runs are present in diagnostics but absent from published
  keyframes;
- QA contains observed/published coverage, path margin, unresolved gap, and
  ball-speed metrics.

## Licensing and deployment boundary

The current prototype is explicitly scoped to non-commercial enthusiast use.
The runtime uses Ultralytics, whose standard open-source distribution is
AGPL-3.0, and a checkpoint trained from Roboflow Sports material described by
the project as CC BY 4.0. Dataset terms do not automatically settle the license
for every exported checkpoint or hosted inference deployment. Preserve source
notices and attribution, and verify the exact model/export and Ultralytics
commercial terms before distributing a closed or commercial product.

The pinned WASB-SBDT implementation is MIT, but its training datasets and
checkpoint distribution terms are separate from the code license. Review those
terms for the intended deployment and do not infer permission to redistribute
broadcast video from a model or dataset license. Uploaded match footage also
has its own copyright and platform terms.

Upstream references:

- [Roboflow Sports](https://github.com/roboflow/sports)
- [WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- pinned local WASB source and license under `.references/WASB-SBDT/`
