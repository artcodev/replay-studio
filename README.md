# Replay Studio

Vue 3 + Three.js editor for reconstructing a short football moment, binding reconstructed tracks to real match metadata, and publishing an interactive replay.

## Included

- interactive 3D pitch with orbit, broadcast, tactical, and goal-line cameras;
- timeline playback, slow motion, ball trajectory, track confidence, and event markers;
- reversible manual ball trajectories with a dedicated keypoint timeline and pitch placement;
- persistent player-action intervals and multi-phase keypoints on a dedicated
  timeline for the selected canonical player, with a renderer-neutral 3D
  playback preview;
- player selection and keyframe editing directly on the pitch;
- provider-neutral match, lineup, player, and event binding through server-side API-Football and TheSportsDB adapters;
- team-name and matchday search with a persisted scene-to-match binding;
- FastAPI scene API with SQLite for zero-config development and PostgreSQL in Docker;
- Docker infrastructure for PostgreSQL, Redis, MinIO, FastAPI, the Vue application, and isolated SoccerNet calibration, PRTReID identity, and jersey OCR workers.
- video ingestion with upload validation, FFprobe metadata, H.264 proxy, poster, sampled detector frames, and a synchronized video/3D workspace.
- separate player and ball reconstruction: tiled Roboflow Sports detection or a temporal WASB worker, camera-compensated global trajectory resolution, team-color clustering, and pitch-line calibration.
- multi-angle analysis that reconstructs 2–6 selected views independently,
  keeps the strongest pass as the canonical 3D trajectory, and adds
  conflict-checked cross-view identity evidence only after accepted alignment.
- frame-level human correction: select any detected person or draw a missing box, assign a player/goalkeeper/referee/other role, bind a roster identity, or ignore a phantom.

API-Football is the preferred match-data provider and requires a server-side
API key. TheSportsDB v1 is registered alongside it and can use its free key
(`123`). The editor may select either provider per catalog/bind request;
`MATCH_DATA_PROVIDER` supplies the preferred default when a request omits that
choice. If that preferred adapter has no key, only a legacy request with no
provider resolves to another configured adapter. An explicit choice, an
upstream failure and a saved binding are never silently rerouted. Provider IDs
have separate namespaces, so the server never retries an API-Football fixture
ID against TheSportsDB or vice versa.
Match metadata remains separated from reconstruction data, and a saved project
continues to work from its persisted snapshot when the selected provider is
unavailable.

## Local development

Requirements: Node.js 22+, Python 3.11+, and FFmpeg/FFprobe.

```bash
npm install
python3 -m venv .venv
. .venv/bin/activate
pip install -e 'apps/api[test]'
cp .env.example .env
docker compose up --build -d calibration-worker
curl --fail http://127.0.0.1:8090/health/ready
uvicorn app.main:app --app-dir apps/api
```

Add your API-Football key to `.env` before using the match catalog:

```dotenv
MATCH_DATA_PROVIDER=api-football
API_FOOTBALL_API_KEY=your-dashboard-key
API_FOOTBALL_BASE_URL=https://v3.football.api-sports.io
```

The key is read only by the FastAPI service and is sent upstream in the
`x-apisports-key` header; it must never be exposed to Vue or committed. Both
adapters can be configured and used in the same API process. To make the
existing TheSportsDB adapter the default, set:

```dotenv
MATCH_DATA_PROVIDER=thesportsdb
SPORTSDB_API_KEY=123
SPORTSDB_BASE_URL=https://www.thesportsdb.com/api/v1/json
```

Restart the API only after changing environment configuration. Switching the
provider in the editor does not require a restart. Existing project snapshots
keep their original `source` and provider event IDs; refresh uses that saved
provider. Bind the project to a different provider through its catalog instead
of sending an ID created by the other provider.

`GET /api/catalog/providers` reports the preferred provider, the currently
resolved default and which adapters are configured, without returning either
secret.

The local API uses the PnLCalib worker at `http://127.0.0.1:8090` by default.
Measured latency, cache behavior, and the prioritized acceleration plan are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).
The first readiness request loads and validates both point and line models, so
it can take noticeably longer than the cheap `/health/live` liveness check.
`GET http://127.0.0.1:8000/api/health` reports whether the API can reach the
ready worker. To intentionally use the smaller in-process keypoint fallback,
set `CALIBRATION_WORKER_URL=` explicitly before starting the API.

Player ReID uses a second isolated worker because the official SoccerNet
PRTReID/BPBreID baseline is pinned to Python 3.9 and PyTorch 1.13.1. Its Docker
build downloads and checksum-verifies both official checkpoints automatically;
offline provisioning is documented in
[`services/identity-worker/README.md`](services/identity-worker/README.md).
Start and verify it independently:

```bash
docker compose up --build -d identity-worker
curl --fail http://127.0.0.1:8091/health/live
curl --fail http://127.0.0.1:8091/health/ready
```

The API uses `http://127.0.0.1:8091` in local development and the Compose
service name in the full stack. Missing weights or a failed model load keep the
worker explicitly unavailable in `/api/health`; no random or generic embedding
is substituted. The rest of the editor remains available while identity
reconstruction reports reduced evidence.

Jersey-number recognition is isolated for the same reason. The default worker
uses the SoccerNet MMOCR DBNet+SAR baseline; EasyOCR is an explicit challenger,
and the provider-neutral `jersey-ocr.v1` contract carries tracklet IDs so a
stronger PARSeq/tracklet implementation can replace either provider without
changing API callers. Start and verify it independently:

```bash
docker compose up --build -d jersey-ocr-worker
curl --fail http://127.0.0.1:8093/health/live
curl --fail http://127.0.0.1:8093/health/ready
```

Crop QA, low-confidence readings and competing numbers are returned as
auditable evidence. Reconstruction aggregates up to five quality-ranked,
temporally separated views per tracklet; a persisted split gets the same
bounded reservoir per prospective partition and final ownership is re-fused
after split/merge. ReID and OCR workers fingerprint decoded crop pixels, so
the same image cannot become multiple independent votes through different
IDs, batches or correction history. Only repeated reliable readings from
independent crops may help join compatible track fragments. OCR roster matches remain suggestions
and are never silently accepted as an external player identity. Runtime,
offline provisioning and the multipart batch contract are
documented in
[`services/jersey-ocr-worker/README.md`](services/jersey-ocr-worker/README.md).

For the complete identity pipeline, start both optional identity services and
verify the main API view of their exact model contracts:

```bash
docker compose up --build -d identity-worker jersey-ocr-worker
curl --fail http://127.0.0.1:8091/health/ready
curl --fail http://127.0.0.1:8093/health/ready
curl --fail http://127.0.0.1:8000/api/health
```

The editor's **Identity review** panel then shows quality-ranked crops,
worker readiness/rejection reasons and closed-set roster hypotheses. Names are
never accepted automatically; when the resolver abstains, any player from the
saved full roster remains available through the explicit manual picker. A
complete roster snapshot is required; a
truncated free-provider response remains manual-only. Legacy provider data can
be refreshed, or a strict JSON roster can be imported through
`POST /api/scenes/{sceneId}/match-binding/import`. A validated example for the
current Spain–Belgium clip lives at
[`data/matches/spain-belgium-2026-qf.json`](data/matches/spain-belgium-2026-qf.json).

Match settings are project-scoped. The root video scene owns the canonical
snapshot; segment and multi-pass scenes receive the same effective binding with
`scope`, `projectSceneId`, and `inherited` metadata. Bind/import/refresh requests
may use the active shot ID, but the API validates and updates the complete video
project atomically. Reconstruction remains scene-scoped: only an explicitly
requested single-pass shot may be queued, while a request from the root video
queues no child shots. Startup migration promotes the richest compatible legacy
snapshot but never creates hidden reconstruction jobs; affected shots are
rebuilt only when the user opens and explicitly reconstructs them.

The opt-in labelled-crop harness evaluates both workers without loading model
runtimes into the API or claiming accuracy from synthetic tests. Its manifest,
dataset fingerprint, thresholds, identity distance/role metrics and OCR
exact/abstention metrics are documented in
[`services/model-validation/README.md`](services/model-validation/README.md).

Ball detection defaults to the bundled one-class Roboflow Sports checkpoint
and runs on a dense, source-resolution frame cache. The reconstruction menu
can instead select the three-frame WASB challenger. Start it locally with:

```bash
PYTHONPATH=services/ball-worker WASB_DEVICE=cpu \
  .venv/bin/uvicorn ball_worker_service.main:app --host 127.0.0.1 --port 8092
curl --fail http://127.0.0.1:8092/health/ready
```

Or use `docker compose --profile challengers up --build ball-worker`. CPU is
correct but slow; CUDA setup is documented in
[`services/ball-worker/README.md`](services/ball-worker/README.md).

In a second terminal:

```bash
npm run dev
```

Open [http://localhost:5188](http://localhost:5188). API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Full infrastructure

```bash
docker compose up --build
```

Open [http://localhost:8080](http://localhost:8080). MinIO Console is exposed at [http://localhost:9001](http://localhost:9001).

## Scene contract

A scene stores reconstruction and metadata bindings separately:

- `canonicalPeople[]`: authoritative video identity, source tracklets,
  observation bboxes, evidence, confidence and optional roster binding;
- `canonicalPeople[].externalPlayerId`: authoritative optional roster binding;
- `tracks[].canonicalPersonId`: optional renderable 3D projection of that
  identity; a canonical person remains editable when this track is absent;
- `tracks[].keyframes`: world-space `x/z`, timestamp, and confidence;
- `ball.keyframes`: active world-space `x/y/z` trajectory;
- `ball.automaticKeyframes` / `ball.manualKeyframes`: retained detector and
  human-authored layers selected by `ball.mode`;
- `matchBinding`: versioned offline project snapshot with event, teams, full
  lineup, timeline, provider IDs and project ownership metadata;
- `tracks[].externalPlayerId`: derived render/compatibility mirror of the
  canonical binding; it is never edited directly;
- `eventBindings[]`: external event placed at a scene timestamp;
- `payload.playerActions[]`: compact persistent action intervals keyed by
  `canonicalPersonId`, with semantic keypoints and manual/automatic provenance;

This makes the replay portable and prevents an upstream API outage or ID change from corrupting the animation.

The editor first keeps the provider saved by the current project; without one,
it selects the registry's resolved `defaultProvider`. It can query either
registered provider by a pairing such as `Spain vs Belgium` or by matchday.
Selecting a match persists the provider name, its event ID and both external
team IDs immediately; player and timeline bindings remain explicit editor
choices. IDs are meaningful only inside the provider recorded in the project
snapshot.

## Player action timeline

Select a canonical player on the video or in the 3D scene, move the shared
playhead, and choose **Add action**. The selected-player timeline supports 21
semantic action types, editable start/end times, and multiple significant
phase markers: `wind-up`, `contact`, `release`, `apex`, `impact`, and
`recovery`. Clicking an interval or marker seeks the synchronized video/3D
playhead.

Every manual add, edit, and delete is persisted immediately; a separate scene
save is not required:

```http
POST   /api/scenes/{sceneId}/player-actions
DELETE /api/scenes/{sceneId}/player-actions/{actionId}
```

The records live under `payload.playerActions` and use
`canonicalPersonId` as the authoritative actor identity. At playback time the
web app derives a normalized interval phase and nearest semantic marker for
`ThreeViewport` when the selected identity has a render track. This is only a
renderer-neutral preview: automatic action
recognition/review, fingerprint invalidation, UCS assets, and Three.js
`AnimationClip` synchronization are not implemented yet. The data contract,
workflow, and remaining debt are documented in
[`docs/PLAYER_ACTIONS.md`](docs/PLAYER_ACTIONS.md).

## Video ingestion

Use **Import clip** in the top bar and select a continuous gameplay shot. The current baseline performs real media processing:

1. saves the private source clip;
2. reads duration, dimensions, and FPS with FFprobe;
3. creates an H.264/yuv420p browser proxy and poster with FFmpeg;
4. extracts up to 10 FPS JPEG frames for people/calibration and lazily caches up to 25 FPS source-resolution frames for the ball;
5. detects continuous camera shots and ranks candidates at least four seconds long;
6. automatically creates up to five child moment scenes;
7. runs a local YOLO medium detector on the primary moment, links observations into tracks, compensates broadcast camera motion, calibrates the visible pitch markings, evaluates explicit QA gates, and opens the reconstructed scene.

FFmpeg scene-change analysis finds continuous camera shots inside highlight montages. Child scenes reuse the same browser proxy with exact source offsets, so processing is idempotent and does not duplicate media. The parent scene keeps every candidate for manual review, while recommended moments are marked and ready to open.

Player reconstruction runs at 10 FPS with `yolo26m.pt`. Ball reconstruction is independent: `dedicated-ultralytics` uses the one-class Roboflow Sports YOLOv8x checkpoint with overlapping 640px tiles, while `wasb-service` uses three temporal frames. The resolver retains top-K candidates and chooses a global path with explicit `observed`, `inferred`, and `occluded` states; only short gaps supported on both sides are interpolated. Lower-spec machines can override `BALL_ANALYSIS_FRAME_RATE` and `BALL_DETECTION_INFERENCE_BATCH_SIZE`. The **Rebuild tracks** action reruns the processor in the background and the Vue editor shows dense-frame progress.

Detector provenance and per-frame diagnostics are stored under
`videoAsset.reconstruction.ballDetection`; trajectory coverage, gaps,
speed/acceleration violations and ambiguity live under
`payload.ball.diagnostics`. A rejected calibration keeps image-space evidence
but does not publish a fabricated 3D ball path. Single-view height remains
unknown and the render height is explicitly marked as a placeholder. Clean
post-NMS ball candidates are cached by dense-frame and detector fingerprints,
so an identical rebuild reruns calibration and trajectory resolution without
repeating the expensive tiled model inference.

When the detector path is unreliable, select **Match ball** and switch
**Trajectory source → Manual keypoints**. A separate ball timeline appears:
move the playhead, add a keypoint, then click anywhere on the 3D pitch (or edit
metric X/Z in the inspector). Add another keypoint later in time and playback
linearly interpolates the ball between them. Keypoint time is editable to the
millisecond and markers can be removed individually. Switching back to
**Automatic detection** restores the latest detector result without deleting
the manual path; later reconstruction refreshes only the automatic layer.

Local detections first form short tracklets. PRTReID then supplies a separate
256D soccer appearance space; the offline resolver may join non-overlapping
tracklets only with strong ReID, reliable jersey/external identity evidence, or
an explicit manual decision. Team, proximity and pitch motion are constraints,
not identity proof. The resulting `canonicalPeople[]` is independent of metric
calibration, so a detected person is not lost merely because no 3D position was
accepted. Quality-ranked independent crops replace the old first-frame sampling,
and unchanged base detections/ReID/OCR outputs are content/model/policy cached
for correction rebuilds. See
[`docs/IDENTITY_RESOLUTION.md`](docs/IDENTITY_RESOLUTION.md).

To correct recognition, move to a frame and choose **Analyze frame**, then
**Label frame**. Click any box—including an unmatched detection—or drag a new
box around a missing person. The identity action can confirm the observation,
exclude one observation or the complete identity, merge duplicates, or split a
wrongly merged identity over a half-open time range. Roster bind/unbind/clear works
from the canonical identity inspector even when that person is off-screen; the
server anchors it to a saved detector observation and never treats an OCR
candidate as confirmation. The preview updates immediately and saving queues
the rebuild atomically; missing, ambiguous, recycled, self, excluded-target and
cyclic corrections fail closed. Rebuilds preserve the last successful scene
until a complete replacement is ready.

Correction save/delete and rebuild queueing are one fingerprint-guarded database
transaction. If the API restarts after that transaction but before its background
callback runs, the continuous recovery monitor atomically claims the persisted
`queued` run and continues it with the same run ID, run revision, and input
fingerprint. A separate database lease has an owner token, heartbeat, and expiry;
another process cannot steal an active run, while an expired or legacy
`processing` run is reclaimed automatically. Terminal success/failure removes
the lease, and an old owner is fenced from publishing after takeover.
Every `SceneDocument` response also carries a full-document `revision`.
Existing-row saves and worker progress writes compare and increment it atomically;
stale clients or workers receive a conflict instead of replacing unrelated newer
data. SQLite uses `BEGIN IMMEDIATE`, so this guarantee also holds across
API/worker processes.

The coordinate stage sends sampled frames to the isolated SoccerNet PnLCalib worker. It predicts semantic field keypoints and lines, performs RANSAC plus point/line refinement, and returns an image-to-pitch homography per frame. Players use their sampled-frame solution; denser ball frames use a bounded interpolation only between two QA-accepted samples across a reliable camera-motion edge, otherwise recording an explicit nearest-frame fallback. Computed scenes store calibration coverage, backend, inliers, reprojection error and per-frame evidence in `videoAsset.reconstruction`; the independent quality verdict determines whether the result passes, needs review, or must be rejected. The bundled Roboflow Sports 32-keypoint model fills missed frames and is the default local fallback when no worker URL is configured. The former Hough fit and screen-relative 2.5D projection are retained only as explicit last-resort fallbacks and must not be silently mixed into a metric run.

The visible goal side is inferred from the calibrated penalty/goal-area position in the image. Attack direction is stored separately: use **Attack goal → Left/Right** in the editor toolbar when the automatic orientation is wrong or the clip does not contain enough motion evidence. Changing it updates only `pitchOrientation.attackingGoal` metadata; it never mirrors players, ball, calibration, or manual anchors. The explicit choice remains authoritative on later **Rebuild tracks** runs.

For replays of the same event, open the parent video scene, select two to six continuous shots under **Multi-angle passes**, and start the analysis. Every angle keeps its own reconstruction and quality report. The resulting composite scene uses the strongest calibrated pass as its canonical timeline and records how many independent passes support the pitch, players, and ball. Cross-angle identity is fused only after accepted time alignment and explicit roster or reliable jersey evidence; proximity alone is never sufficient.

After clock alignment is accepted, the composite may attach cross-angle
identity evidence supported by the same explicit roster player or reliable
shirt number. Team/role/number conflicts and ambiguous candidates abstain;
observations from another camera stay namespaced and never become fabricated
reference-camera positions. Manual alignment anchors are persisted and must be
finite, in range and monotonic.

The prototype uses Ultralytics YOLO and the Roboflow Sports ball checkpoint. Ultralytics is AGPL-3.0 and the Roboflow dataset is CC BY 4.0; this matches the current non-commercial enthusiast scope, but licensing and attribution must be revisited before a commercial release. The pinned WASB implementation is MIT; its external soccer dataset terms remain separate.

The runtime QA contract, calibration evidence format, initial gates, gold-set workflow, and the honest boundary between a ground-plane game state and full 3D are documented in [`docs/RECONSTRUCTION_QUALITY.md`](docs/RECONSTRUCTION_QUALITY.md).

Development limits are 250 MB and 60 seconds. The intended product input remains a 5–20 second continuous football clip.

## Verification

```bash
npm run typecheck
npm run test
npm run build
.venv/bin/pytest -q apps/api/tests
.venv/bin/pytest -q services/identity-worker/tests
.venv/bin/pytest -q services/ball-worker/tests
.venv/bin/pytest -q services/jersey-ocr-worker/tests
.venv/bin/pytest -q services/model-validation/tests -rs
```

Data attribution: match metadata is provided by [TheSportsDB](https://www.thesportsdb.com/).

## Research notes

- [Football data/API comparison](docs/FOOTBALL_DATA_APIS.md)
- [SoccerNet, Roboflow Sports, and Game State roadmap](docs/GAME_STATE_RECONSTRUCTION_ROADMAP.md)
- [Ball detection, temporal tracking, QA, and deployment](docs/BALL_TRACKING.md)
- [Pitch calibration architecture and deployment](docs/CALIBRATION.md)
- [Canonical identity architecture and QA](docs/IDENTITY_RESOLUTION.md)
- [Real-model identity/OCR validation harness](services/model-validation/README.md)
- [Selected player/ball Path Tracking layer](docs/PATH_TRACKING.md)
- [Player action annotations and deferred UCS animation debt](docs/PLAYER_ACTIONS.md)
- [Technical debt audit](docs/TECHNICAL_DEBT.md)
