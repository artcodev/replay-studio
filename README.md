# Replay Studio

Vue 3 + Three.js workspace for organizing football-match projects, extracting
moments from one or more videos, reconstructing them in 3D, and reviewing the
result against a provider-neutral match snapshot.

## Included

- interactive 3D pitch with orbit, broadcast, tactical, and goal-line cameras;
- timeline playback, slow motion, ball trajectory, track confidence, and event markers;
- reversible manual ball trajectories with a dedicated keypoint timeline and pitch placement;
- persistent player-action intervals and multi-phase keypoints on a dedicated
  timeline for the selected canonical player, with a renderer-neutral 3D
  playback preview;
- player selection and keyframe editing directly on the pitch;
- first-class projects containing one optional canonical match, multiple video
  assets, segments, project identities, and analysis jobs;
- provider-neutral match, lineup, player, and event data through server-side
  API-Football and TheSportsDB adapters;
- team-name and matchday search with an immutable project-level match snapshot;
- FastAPI project API with project-owned scene/media routes, SQLite for zero-config
  development and PostgreSQL in Docker;
- compact `AnalysisRun` telemetry with phase progress and terminal history;
  compact jobs and leases own atomic cancellation/retry, last-good result
  preservation, and execution by dedicated runners;
- Docker infrastructure for PostgreSQL, Redis, FastAPI, the Vue
  application, and isolated SoccerNet calibration, PRTReID identity, and jersey
  OCR workers;
- video ingestion with upload validation, FFprobe metadata, H.264 proxy, poster,
  sampled detector frames, and a synchronized video/3D workspace;
- separate player and ball reconstruction: tiled Roboflow Sports detection or a
  temporal WASB worker, camera-compensated global trajectory resolution,
  team-color clustering, and pitch-line calibration;
- multi-angle analysis that reconstructs 2–6 selected views independently,
  keeps the strongest pass as the canonical 3D trajectory, and adds
  conflict-checked cross-view identity evidence only after accepted alignment.
- frame-level human correction: select any detected person or draw a missing box, assign a player/goalkeeper/referee/other role, bind a roster identity, or ignore a phantom.

API-Football is the preferred match-data provider and requires a server-side
API key. TheSportsDB v1 is registered alongside it and can use its free key
(`123`). `MATCH_DATA_PROVIDER` selects the adapter used by project-scoped match
search. Vue receives only Replay Studio IDs and canonical match DTOs; provider
names and upstream IDs remain in integration tables and are exposed only by the
explicit diagnostics endpoint. Refresh uses the source of the saved snapshot
and never silently sends an ID to another provider. A project remains usable
from its persisted snapshot while the upstream provider is unavailable.

## Project workflow

1. Open **Projects** and create or select a project.
2. In **Match**, search by team pairing or matchday and select the canonical
   match. This is optional while the project is a draft.
3. Upload one or more videos into that project. Processing creates proxies,
   posters and candidate segments but does not reconstruct every segment.
4. Open a segment and explicitly run **Build automatic tracks** for that
   segment. Sibling segments and other projects are not queued implicitly.
5. Use **Analysis** to inspect completed, current and pending phases or request
   cancellation. Reconstruction cancellation releases its lease immediately,
   preserves the last accepted scene, and allows an explicit retry; the fenced
   worker cannot publish later.
6. Use **Identities** to review the cross-scene graph and explicitly reassign a
   scene membership when two local IDs are the same real person.
7. For another angle, upload it to the same project. Cross-video compositions
   reference project segment IDs; source media and segment-local calibration
   remain separate.

The canonical domain boundaries are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the concrete UI/API workflow is
in [`docs/PROJECT_WORKFLOW.md`](docs/PROJECT_WORKFLOW.md).
The 2026-07-18 live storage cutover reduced Shot 2 by 92.55% during artifact
publication and left all eight compact Scene rows at 81,565 bytes total. Exact
sizes and post-cutover idle network deltas are recorded in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Local development

Requirements: Node.js 22+, Python 3.11+, and FFmpeg/FFprobe.

```bash
npm install
python3 -m venv .venv
. .venv/bin/activate
pip install -e 'apps/api[test]'
cp .env.example .env
python -m alembic -c apps/api/alembic.ini upgrade head
docker compose up --build -d calibration-worker
curl --fail http://127.0.0.1:8090/health/ready
uvicorn app.main:app --app-dir apps/api
```

Add your API-Football key to `.env` before using project match search:

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

Restart the API after changing environment configuration. Existing project
snapshots retain private source provenance, and refresh resolves that provenance
server-side. The normal project and match responses never require Vue to know
which upstream adapter supplied the data.

`GET /api/health` reports the resolved match-data adapter readiness without
returning credentials. Match search itself is scoped to the active project at
`GET /api/projects/{projectId}/match/search`.

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
independent crops may help join compatible track fragments. OCR roster matches
remain suggestions and are never silently accepted as an external player
identity. Runtime, offline provisioning and the multipart batch contract are
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
complete roster snapshot is required; a truncated free-provider response
remains manual-only. Saved provider data can be refreshed, or a strict JSON
roster can be imported through
`POST /api/projects/{projectId}/match/import`. A validated example for the
current Spain–Belgium clip lives at
[`data/matches/spain-belgium-2026-qf.json`](data/matches/spain-belgium-2026-qf.json).

Match settings are project-scoped. Canonical snapshots are stored once in the
normalized match tables; Scene routes cannot mutate match state.
Reconstruction remains segment/scene-scoped: only an
explicitly requested shot or composition is queued. API startup performs schema
initialization only; it never derives projects,
matches, snapshots, roster IDs or jobs from scene payloads. Existing prototype
data must be explicitly re-imported or reset instead of being silently mutated
on every process restart. The API never executes reconstruction jobs; the
dedicated runner is the sole claimant.

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

Reconstruction and video/multi-pass orchestration require their dedicated
runners in local development. Start both with the same `DATABASE_URL` and
`MEDIA_ROOT` as the API:

```bash
PYTHONPATH=apps/api .venv/bin/python -m app.reconstruction_runner
PYTHONPATH=apps/api .venv/bin/python -m app.pipeline_runner
```

Do not intentionally run multiple local claimers to make one scene faster. The
lease prevents duplicate publication, while
`RECONSTRUCTION_RECOVERY_MAX_WORKERS=1` is the default budget for the single
configured runner.

## Full infrastructure

```bash
docker compose up --build
```

The one-shot `migrate` service upgrades PostgreSQL first; the API and both
runners start only after it exits successfully. The API only publishes compact
jobs and their telemetry. Each runner isolates work in child processes, so
cancellation terminates blocked native processing before a slot is reused by a
retry. Open
[http://localhost:8080](http://localhost:8080).

## Database migrations

Alembic is the only schema-management path. API startup also upgrades to
`head`, which keeps SQLite development zero-config. Startup applies schema
changes only; it never derives projects, ownership, matches or jobs from Scene
JSON.

```bash
python -m alembic -c apps/api/alembic.ini current
python -m alembic -c apps/api/alembic.ini upgrade head
```

Do not replace migrations with `Base.metadata.create_all()` or edit a live
schema by hand. Back up the database before upgrading user data. CI applies the
full chain to a fresh PostgreSQL service. Prototype data is re-imported or reset
only by an explicit operator decision; it is not supported by runtime fallback
branches.

## Scene analysis contract

The normalized project model owns match, media, segment, identity and job
records. A project-owned `SceneDocument` is the compact analysis/editor control
and summary model:

- `canonicalPeople[]`: authoritative compact video identity, confidence,
  observation count and optional roster binding;
- `canonicalPeople[].externalPlayerId`: authoritative optional roster binding;
- `tracks[].canonicalPersonId`: optional renderable 3D projection of that
  identity; a canonical person remains editable when this track is absent;
- `artifactManifest.identityTimeline`: immutable source tracklets, observation
  bboxes and world-space `x/z` keyframes;
- `artifactManifest.ballTrajectory`: immutable active, automatic and manual
  world-space `x/y/z` keyframe layers selected by compact `ball.mode`;
- `artifactManifest.calibrationFrames`: immutable per-frame calibration and
  ball-detection evidence;
- `tracks[].externalPlayerId`: derived render mirror of the
  canonical binding; it is never edited directly;
- `eventBindings[]`: external event placed at a scene timestamp;
- `payload.playerActions[]`: compact persistent action intervals keyed by
  `canonicalPersonId`, with semantic keypoints and manual/automatic provenance;

`ProjectPersonMembership` connects a scene-local `canonicalPersonId` to a
project identity without pretending that equal detector IDs from two scenes
refer to the same person. Provider IDs are deliberately absent from normal
project, match and identity DTOs. See
[`docs/PROJECT_IDENTITIES.md`](docs/PROJECT_IDENTITIES.md).

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
POST   /api/projects/{projectId}/scenes/{sceneId}/player-actions
DELETE /api/projects/{projectId}/scenes/{sceneId}/player-actions/{actionId}
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
7. marks those moments ready for review; automatic reconstruction starts only
   after the user explicitly selects a moment and runs **Build automatic tracks**.

FFmpeg scene-change analysis finds continuous camera shots inside highlight montages. Child scenes reuse the same browser proxy with exact source offsets, so processing is idempotent and does not duplicate media. The parent scene keeps every candidate for manual review, while recommended moments are marked and ready to open.

The explicit reconstruction job runs player detection at 10 FPS with
`yolo26m.pt`. Ball reconstruction is independent: `dedicated-ultralytics` uses
the one-class Roboflow Sports YOLOv8x checkpoint with overlapping 640px tiles,
while `wasb-service` uses three temporal frames. The resolver retains top-K
candidates and chooses a global path with explicit `observed`, `inferred`, and
`occluded` states; only short gaps supported on both sides are interpolated.
Cold pitch calibration directly solves camera anchors at most one second apart
and lets the temporal solver reconstruct the intervening frames; manual anchors
and the single-frame calibration command are not sampled away.
Lower-spec machines can override `BALL_ANALYSIS_FRAME_RATE` and
`BALL_DETECTION_INFERENCE_BATCH_SIZE`. The **Rebuild tracks** action reruns the
processor in the background and the Vue editor shows dense-frame progress.

Compact detector provenance and QA counts remain under
`videoAsset.reconstruction`; per-frame detector/calibration evidence and full
trajectory diagnostics are immutable artifacts fetched through bounded
reconstruction-series windows. A rejected calibration keeps image-space evidence
but does not publish a fabricated 3D ball path. Single-view height remains
unknown and the render height is explicitly marked as a placeholder. Clean
post-NMS ball candidates are cached by dense-frame and detector fingerprints,
so an identical rebuild reruns calibration and trajectory resolution without
repeating the expensive tiled model inference. During a cold detector run, a
clean contiguous primary-backend prefix is also checkpointed every
`BALL_DETECTION_CHECKPOINT_INTERVAL` frames (four by default). Cancelling and
rebuilding the exact same inputs resumes after that prefix instead of discarding
all completed tiled inference. Fallback or failed frames are never added to the
resumable prefix.

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

Correction save/delete and compact-job publication are one fingerprint-guarded
database transaction. The dedicated runner atomically claims the persisted
`queued` reconstruction job and continues it with the same analysis run ID,
run revision, and input fingerprint. A separate database lease has an owner
token, heartbeat, and expiry;
another process cannot steal an active job, while an expired
`processing` job is reclaimed automatically. Terminal success/failure removes
the lease, and an old owner is fenced from publishing after takeover.
Every `SceneDocument` response also carries a full-document `revision`.
Existing-row saves and worker progress writes compare and increment it atomically;
stale clients or workers receive a conflict instead of replacing unrelated newer
data. SQLite uses `BEGIN IMMEDIATE`, so this guarantee also holds across
API/worker processes.

The coordinate stage sends sampled frames to the isolated SoccerNet PnLCalib worker. It predicts semantic field keypoints and lines, performs RANSAC plus point/line refinement, and returns an image-to-pitch homography per frame. Players use their sampled-frame solution; denser ball frames use a bounded interpolation only between two QA-accepted samples across a reliable camera-motion edge, otherwise recording an explicit nearest-frame fallback. Computed scenes retain compact calibration coverage, backend, inlier and reprojection-error summaries; per-frame evidence is published to `artifactManifest.calibrationFrames`. The independent quality verdict determines whether the result passes, needs review, or must be rejected. The bundled Roboflow Sports 32-keypoint model fills missed frames and is the default local fallback when no worker URL is configured. The former Hough fit and screen-relative 2.5D projection are retained only as explicit last-resort fallbacks and must not be silently mixed into a metric run.

The visible goal side is inferred from the calibrated penalty/goal-area position in the image. Attack direction is stored separately: use **Attack goal → Left/Right** in the editor toolbar when the automatic orientation is wrong or the clip does not contain enough motion evidence. Changing it updates only `pitchOrientation.attackingGoal` metadata; it never mirrors players, ball, calibration, or manual anchors. The explicit choice remains authoritative on later **Rebuild tracks** runs.

For replays of the same event, select project segments representing the relevant
views and create a composition. Segments may belong to different video assets in
the same project. Every angle keeps its own reconstruction and quality report;
the composition references those sources instead of copying media. The result
uses the strongest calibrated pass as its canonical timeline and records how
many independent passes support the pitch, players, and ball. Cross-angle
identity is fused only after accepted time alignment and explicit roster or
reliable jersey evidence; proximity alone is never sufficient.

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

Match-data attribution and usage terms depend on the configured adapter:
[API-Football](https://www.api-football.com/) or
[TheSportsDB](https://www.thesportsdb.com/). Persisting a snapshot does not
remove the provider's attribution or licensing requirements.

## Research notes

- [Football data/API comparison](docs/FOOTBALL_DATA_APIS.md)
- [SoccerNet, Roboflow Sports, and Game State roadmap](docs/GAME_STATE_RECONSTRUCTION_ROADMAP.md)
- [Ball detection, temporal tracking, QA, and deployment](docs/BALL_TRACKING.md)
- [Pitch calibration architecture and deployment](docs/CALIBRATION.md)
- [Canonical identity architecture and QA](docs/IDENTITY_RESOLUTION.md)
- [Real-model identity/OCR validation harness](services/model-validation/README.md)
- [Selected player/ball Path Tracking layer](docs/PATH_TRACKING.md)
- [Player action annotations and deferred UCS animation debt](docs/PLAYER_ACTIONS.md)
- [Project workflow, public API and runtime model](docs/PROJECT_WORKFLOW.md)
- [Project-level identity persistence](docs/PROJECT_IDENTITIES.md)
- [Versioned reconstruction quality benchmark](docs/QUALITY_BENCHMARK.md)
- [Technical debt audit](docs/TECHNICAL_DEBT.md)
