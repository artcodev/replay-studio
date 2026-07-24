# Pitch calibration architecture

## Chosen solution

Replay Studio now treats calibration as a server-side vision task. The primary
backend is the official SoccerNet PnLCalib implementation from
[`sn-gamestate`](https://github.com/SoccerNet/sn-gamestate), pinned to commit
`1c958345067218297d221e45e1a6405f975f83e0` in an isolated worker.

PnLCalib predicts 57 semantic pitch keypoints plus a background channel and 23
semantic pitch lines plus a background channel. It
then estimates a homography with RANSAC and refines it against line evidence.
This is qualitatively different from the old Hough approach: a detected line is
not merely “some white line”, but part of a named left/right field structure.
The left and right penalty areas therefore no longer form an unresolved mirror
hypothesis.

The API sends all sampled frames of a continuous shot to the worker. Every
successful result includes:

- `imageToPitch`: original-frame pixels to centred 105 x 68 metre coordinates;
- `pitchSide`: semantic visible half (`left`, `right`, or unknown near midfield);
- keypoint, line, and inlier counts;
- reprojection error and calibrated confidence;
- method and source-frame index.
- `rawKeypoints` and `rawLines`, the source-frame observations used to audit
  what the networks actually detected before any inferred geometry is added.

Sampling cadence is an explicit calibration input. The editor shows the source
FPS, immutable-generation FPS, selected FPS and estimated frame count. Native
source FPS is the default; the operator may choose a lower preset before a full
calibration. The selected value is persisted and fingerprinted, so changing it
invalidates the old calibration gate and requires recalibration. Reconstruction
consumes the same cadence as its pinned calibration artifact.

Each `rawLines` item has the stable contract
`{id, name, start: {x, y}, end: {x, y}, confidence, groundPlane}`. Names follow
the official 23-class PnLCalib order. Goal crossbars/posts are returned as
`groundPlane: false`: they are valuable orientation evidence, but they are not
fed to the grass-plane homography solver. `detectedLineCount` counts all raw
segments while `lineCount` counts only the planar segments used by PnLCalib.

Person-foot and ball-centre observations are projected with the homography from
their own frame before camera stabilisation. Tracking is still performed in the
stabilised image plane, while final track coordinates use those per-frame metric
positions and temporal smoothing. A representative homography is never used to
silently fill missing frames. Missing/rejected observations may only be
recovered by the auditable temporal hypothesis graph below.

Each selected frame gets one cache-aware PnLCalib attempt. If that candidate is
missing, fails frame-local QA, or is later rejected as a shot-level residual-p95
outlier, the API runs at most two additional fresh batch rounds. Every pending
frame is still accepted or rejected independently; batching only removes the
per-frame HTTP/inference setup that made a full recalibration unnecessarily
serial. Both the API disk cache and the worker LRU are bypassed through
`POST /v1/recalibrate`, so a retry is fresh inference rather than another read
of the first result. The first candidate that passes the applicable QA gate is
kept; otherwise the frame proceeds to temporal recovery or explicit manual
review. `pnlcalibAttempts` records every attempt, its request kind, status,
rejection reasons, and selected attempt.

## Temporal hypothesis graph

Calibration is solved per continuous camera shot in two stages:

1. Collect direct point/line calibration candidates, people, and projective
   camera-motion edges for every sampled frame.
2. Resolve the whole sequence with forward and backward inference. A strong
   later direct anchor can therefore recover an earlier partial view, and an
   earlier anchor can predict a later frame.

Each motion edge maps image coordinates in frame `t` to frame `t - 1`. It is
estimated with forward/backward Lucas–Kanade tracks and a RANSAC projective
homography. Tracked-point coverage, inlier ratio, residual p50/p95, scene-change
score, and motion confidence are stored. `cut` and `unreliable` edges are hard
barriers; camera stabilisation also starts a new coordinate system after them.

For a frame without an accepted direct fit, the solver retains up to two direct
anchors on each side of time. Every candidate stores its anchor frame, motion
path, direction, score, homography, visible side, disagreement with competing
candidates, and engineering p95 position uncertainty. The rules are:

- a direct accepted observation is never overwritten;
- compatible earlier/later candidates form `temporal-bidirectional` consensus;
- similarly scored incompatible candidates stay `ambiguous` and publish no
  metric coordinates;
- a path never crosses a cut or an unestimated motion edge;
- uncertainty increases with temporal distance, edge residuals, weak motion,
  missing target-line evidence, and sparse person support;
- target-frame semantic lines or person-foot support can veto a propagated fit;
- a rejected direct observation remains visible as `direct-rejected`, but is
  never promoted to an anchor.

The frame API keeps observation and solution separate through
`observationStatus`, `solutionStatus`, `hypotheses`, `selectedHypothesisId`,
`temporal`, and `uncertainty`. The Vue QA timeline displays direct in green,
temporal in blue,
ambiguous, rejected, missing, and camera-cut states. The selected-frame panel
shows the ranked candidates, anchor(s), direction, temporal gap, motion metrics,
candidate disagreement, and uncertainty.

Shot-level QA reports direct and usable coverage separately, temporal recovery
count, ambiguity count, camera-motion reliability/cuts, and temporal uncertainty
p95. Recovered frames add a required uncertainty gate: p95 up to 2.5 m passes,
2.5–5 m requires review, and larger estimates are not published.

## Calibration sources

1. `pnlcalib-points-lines` is the only automatic calibration backend.
2. A manually saved four-anchor calibration is an explicit operator-authored
   direct anchor in its connected camera shot; it is not propagated through a
   cut.
3. QA-gated temporal recovery may transport an accepted PnLCalib or manual
   direct anchor to a nearby unresolved frame. It is a derived calibration, not
   another backend.

If PnLCalib is unavailable, automatic Calibration fails closed. Reconstruction
does not select or run another calibration model. Screen-relative 2.5D is a
separate, explicitly authorised non-metric reconstruction mode and is never
labelled as calibrated metric output.

## Manual edit workflow

Manual correction is a two-command workflow:

1. `POST .../pitch-calibration/drafts` validates and saves the selected
   frame's anchors plus their source evidence. It does **not** queue a job,
   change the published calibration artifact, or run PnLCalib.
2. `POST .../pitch-calibration/finalize` is the explicit commit command. It
   verifies that the draft still refers to the same immutable base artifact,
   applies the staged manual anchors, and re-solves only non-direct frames that
   could depend on them. Unchanged direct observations and their neural
   evidence are reused. No PnLCalib inference runs during incremental
   finalization.

The editor can prepare a draft from the stored frame homography, from the
nearest resolved previous/next frame, or by interpolating the two surrounding
resolved frames in the stabilised-reference coordinate system. Switching
between penalty-area, goal-area, and centre-circle anchor presets projects a
different semantic point set through the current homography; it does not invoke
the neural worker. A separate **Run PnLCalib** action makes fresh inference
explicit.

The timeline marks staged frames separately and Reconstruction remains blocked
until the staged session is finalized. Finalization logs the edited and
affected sample indices, reused-frame count, base artifact hash, and an explicit
zero PnLCalib inference count.

## Run

The full setup is the recommended path:

```bash
docker compose up --build
```

On Apple Silicon, an opt-in native PyTorch MPS worker is documented in
[`MPS_CALIBRATION_WORKER.md`](MPS_CALIBRATION_WORKER.md). The default Compose
path remains CPU-only.

The worker listens on container port `8090` and Compose publishes that port as
`127.0.0.1:8090` for a locally run API. The API container uses the internal URL
`CALIBRATION_WORKER_URL=http://calibration-worker:8090`; a local API defaults to
`http://127.0.0.1:8090`. Start just the server-side calibrator and wait for both
official models to load with:

```bash
docker compose up --build -d calibration-worker
curl --fail http://127.0.0.1:8090/health/live
curl --fail http://127.0.0.1:8090/health/ready
```

`/health/live` only proves that the worker process is serving HTTP.
`/health/ready` loads and validates both checkpoints and reports the backend,
device, and inference batch size. The API's `/api/health` response includes the
same dependency readiness without making the whole API unhealthy when the
worker is still warming up. PnLCalib is the only automatic calibration backend:
an unconfigured, unavailable, or interrupted worker fails the Calibration run
without publishing a replacement artifact. Manual anchors remain explicit
operator evidence; they are not an automatic inference fallback.

`CALIBRATION_WORKER_TIMEOUT=900` is the HTTP timeout for **each submitted frame
batch**, not a wall-clock deadline for the complete reconstruction job. A job
with many batches can therefore run longer. Reconstruction stays in the
background and does not hold an editor request open.

Calibration requests a direct PnLCalib candidate for every selected analysis
frame by default. The calibration workspace exposes an explicit per-scene
**Direct PnLCalib sampling** option; positive maximum-gap modes trade accuracy
for speed by letting the temporal solver fill frames between sparse anchors.
The selected policy is persisted, fingerprinted, and logged with the
calibration artifact. There is no process-global anchor-gap override. Manual
calibration anchors remain authoritative. **Run PnLCalib** submits exactly the
selected frame, while **Save frame correction** only stages the edited
parameters. Fresh retries are always single-frame requests regardless of the
initial batch size.

The official checkpoints are stored under
`services/calibration-worker/models/` and have these verified MD5 hashes:

```text
pnl_SV_kp     322d4a6c82d2966ea88b69963ba85f07
pnl_SV_lines  270b94527c9e817bc32edd54c8e47b62
```

Official sources:

- `https://zenodo.org/records/14046275/files/pnl_SV_kp?download=1`
- `https://zenodo.org/records/14046275/files/pnl_SV_lines?download=1`

## Verified result on the supplied video

Frames 1, 11, and 21 from
`Highlights_Spain_2-1_Belgium_World_Cup_2026.mp4` were processed end to end by
the worker on CPU. All three calibrated successfully as the right half, using
18–22 completed semantic keypoints, 7–9 line detections, and 4.03–5.55 px
reprojection error. The projected touchline, penalty area, goal area, penalty
arc, and centre-circle segment visually align with the source frame.

A full `Rebuild Tracks` was then run on the short left-goal broadcast scene
`moment-fd717af8fbdd-shot-07`. PnLCalib accepted 16 of 17 sampled frames
(94.1% coverage), selected the left pitch half, and stored a representative
confidence of 0.857 with 2.69 px reprojection error. It produced 275 metric
person observations and 19 accepted tracks. Across 238 saved player keyframes,
X spans -50.25…-17.36 metres and no sample is clamped to a pitch boundary.

These numbers prove that the service and the official weights run on the actual
input. They are not a full accuracy benchmark. We still need a labelled set of
left-goal, midfield, zoom, replay, and occlusion frames to report median pitch
error and mirror-error rate.

## Orientation semantics

Three concepts are stored separately:

- the model's semantic pitch X axis;
- which pitch half is visible in a frame;
- the scene's selected attacking goal.

An explicit editor choice of attacking goal remains authoritative match
metadata. It never mirrors calibration or track coordinates. Attack direction
is no longer inferred by asking which side of the screen a generic rectangle
happens to occupy. PnLCalib calibrations do not use the retired screen-side
canonicalisation heuristic. Recovered frames inherit an
anchor's visible-side label and therefore do not count as independent
orientation votes.

## Licensing and deployment

`sn-gamestate` is GPL-3.0 and the PnLCalib checkpoints on Zenodo are CC BY 4.0.
The existing project is explicitly non-commercial, so the isolated CPU worker is an
acceptable prototype choice. Before a commercial release, review distribution,
attribution, model/dataset rights, and whether the worker must be replaced by an
internally trained compatible service. The API/worker JSON boundary is designed
to make that replacement possible.
