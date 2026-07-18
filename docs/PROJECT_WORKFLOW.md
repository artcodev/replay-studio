# Project workflow and operations

Replay Studio is project-first. A project is the ownership and navigation
boundary for match data, source media, segments, scenes, identities and
analysis work. There are no global scene, video, match-catalog or cancellation
resources.

## Domain hierarchy

```text
Project
├── Canonical Match snapshot (optional)
├── VideoAssets
│   └── Segments
│       └── Scene analysis read model
├── ProjectPeople
│   └── scene-local identity memberships
└── AnalysisRuns
```

`Match` owns football facts. `VideoAsset` owns source and derived media
metadata. `Segment` owns a bounded source range. `SceneDocument` is the compact
editor/reconstruction read model for one segment or composition; immutable
artifacts own dense time series. `AnalysisRun` is compact UI telemetry, while
`reconstruction_jobs` and `pipeline_jobs` are the scheduler sources of truth.

## UI workflow

1. Open **Projects** and explicitly create or select a project. `/projects`
   never restores or guesses a prior selection.
2. In **Match**, search by teams or date and attach a result, or import a
   validated manual match JSON. This changes only the Project match snapshot.
3. Upload footage into the selected project. Upload cannot create an implicit
   project.
4. Follow durable video processing in **Analysis**. A successful generation
   publishes the asset, root timeline, source segments, compact job and
   telemetry atomically.
5. Open an asset timeline and choose one segment for reconstruction. Siblings
   and other projects are never implicit work.
6. Poll the compact project job list. Scene JSON is not fetched repeatedly; it
   is reloaded only after an observed active-to-terminal transition for the
   current run.
7. Add another source video when a moment has another angle. Build a
   composition from canonical project segment IDs.

Refreshing or importing match data never rebuilds a scene automatically. The
new immutable snapshot reference changes the reconstruction input fingerprint;
the user then explicitly rebuilds the current segment or reruns a composition.
Full roster/event data remains in the Project Match API and is never embedded
in segment Scene JSON.

## Browser routes

| View | Route |
| --- | --- |
| Projects entry | `/projects` |
| Project tab | `/projects/{projectId}/{overview|match|identities|analysis}` |
| Video timeline | `/projects/{projectId}/videos/{assetId}/timeline` |
| Segment editor | `/projects/{projectId}/segments/{segmentId}` |
| Composition editor | `/projects/{projectId}/scenes/{sceneId}` |

Vue Router owns navigation state. `ProjectAsset.timelineSceneId` resolves an
asset to its internal timeline read model; the browser never scans scenes or
selects the first result. With multiple assets, the user chooses the intended
timeline explicitly.

## Public API boundary

| Purpose | Route |
| --- | --- |
| Projects | `GET/POST /api/projects` |
| One project | `GET/PATCH /api/projects/{projectId}` |
| Archive project | `POST /api/projects/{projectId}/archive` |
| Match search | `GET /api/projects/{projectId}/match/search` |
| Current match | `GET /api/projects/{projectId}/match` |
| Select/refresh match | `PUT /api/projects/{projectId}/match`, `POST .../refresh` |
| Manual match import | `POST /api/projects/{projectId}/match/import` |
| Project assets/segments/scenes | `GET .../assets`, `GET .../segments`, `GET .../scenes` |
| Upload video | `POST /api/projects/{projectId}/videos` |
| Video metadata/media/poster | `GET /api/projects/{projectId}/videos/{assetId}[/{media|poster}]` |
| Materialize segment scene | `POST .../videos/{assetId}/segments/{sourceSegmentId}/scene` |
| Scene read/write | `GET/PUT /api/projects/{projectId}/scenes/{sceneId}` |
| Bounded dense series | `GET .../scenes/{sceneId}/reconstruction-series` |
| Reconstruct scene | `POST .../scenes/{sceneId}/reconstruct` |
| Project identities | `GET /api/projects/{projectId}/identities` |
| Assign identity membership | `POST .../identities/{projectPersonId}/memberships` |
| Cross-video composition | `POST /api/projects/{projectId}/compositions` |
| Compact jobs | `GET /api/projects/{projectId}/analysis-runs` |
| Cancel one job | `POST /api/projects/{projectId}/analysis-runs/{runId}/cancel` |

The server verifies project ownership before reading a Scene, identity review,
artifact window, video file, composition source or analysis run. A resource
owned by another project returns 404. Media URLs are project-scoped response
fields and are stripped before Scene writes; they are never stored in Scene
JSON.

Provider names and upstream IDs are absent from normal project resources.
Operators can inspect provenance at
`GET /api/projects/{projectId}/integration-diagnostics`.

## Match lifecycle

```text
configured adapter
  -> validated EventBundle
  -> provider-neutral normalization
  -> internal Replay Studio IDs
  -> immutable MatchSnapshot
  -> current snapshot pointer on Project
```

Search returns an opaque internal candidate ID. Selection resolves its private
provider mapping on the server, fetches the bundle and persists canonical
teams, roster, lineup, events and substitutions. Refresh follows stored
provenance. An upstream outage does not erase the last successful snapshot.

## Durable execution and cancellation

The API enqueues work but never executes analysis. Run both claimants locally:

```bash
PYTHONPATH=apps/api .venv/bin/python -m app.pipeline_runner
PYTHONPATH=apps/api .venv/bin/python -m app.reconstruction_runner
```

Docker Compose supplies the same processes as `pipeline-runner` and
`reconstruction-runner`. Both queues use compact rows, lease heartbeats,
fencing tokens and bounded supervised child processes. Dense Scene or frame
payloads are not scheduler messages.

Cancellation is project-scoped and monotonic. Queued work becomes `cancelled`;
running cooperative jobs pass through `cancelling`. Reconstruction cancellation
fences the matching job/lease/Scene/run in one canonical lock order, preserves
the last accepted result and prevents a late worker from publishing. A new
explicit run can then be queued without reviving the cancelled generation.

Video processing writes a unique `.pipeline-runs/{generationKey}` directory.
Only a fenced terminal transaction selects the published generation. Media
readers use that selected directory and never fall back to mutable canonical
files.

## Multiple videos and compositions

Every asset and Scene has exactly one Project owner. Project segment IDs are
stable even when two assets both contain detector-local `shot-01`. Composition
creation validates every source against the project, then atomically publishes
the composite Scene, ownership link, pipeline job and AnalysisRun.

Each source remains independently calibrated and reconstructed. Time alignment
must be accepted before cross-angle evidence is fused. Explicit roster
agreement or reliable jersey/ReID evidence may support an identity link;
proximity alone cannot.
