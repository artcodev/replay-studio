# Replay Studio architecture

This document defines the canonical domain boundaries for the local-first
editor. Project, match, media, segment, analysis-run and project-identity data
are normalized. `SceneDocument` is a project-owned reconstruction read model;
compact control-plane records and immutable artifacts own execution and dense
series respectively.

## Architecture decision rule

The project optimizes for the simplest, most efficient canonical architecture,
not for backward compatibility with earlier prototypes.

- Obsolete database layouts, global scene/video routes, embedded dense results,
  startup domain backfills, dual writes and compatibility projections are not
  supported by the current runtime.
- A replacement should migrate the current first-party consumers and delete the
  superseded path in the same cutover. Keeping an old path "just in case" is
  not a requirement.
- Breaking internal API and schema changes are acceptable. Development data may
  be reset or re-imported instead of carrying a permanent compatibility layer;
  destructive operations still require explicit user approval.
- A temporary bridge must have a concrete removal condition, must not become a
  second writable source of truth, and must stay off performance-critical paths.
- Intentional current-product recovery paths are retained only for documented
  edge cases, with visible provenance, quality constraints and tests. Obsolete
  compatibility branches are deleted. Silent degradation is forbidden.
- Reviews and tests evaluate the target architecture and current product
  behavior. Obsolete contracts do not receive maintenance work solely because
  they already exist.

## Implementation status

| Boundary | Current state |
| --- | --- |
| Project ownership | Normalized projects plus unique scene/asset ownership links |
| Match | Canonical revisioned snapshots; provider provenance kept separately |
| Media and segments | Project-scoped assets/segments; every upload requires an existing project |
| Analysis progress | Compact `AnalysisRun` telemetry/history rows; never a scheduling source |
| Reconstruction execution | Dedicated runner only, with compact DB jobs and lease/fingerprint fencing |
| Video/multi-pass execution | Dedicated pipeline runner, compact `pipeline_jobs`, expiring leases and process isolation |
| Cancellation | Lease-fenced atomic cancellation for reconstruction, video processing and multi-pass jobs |
| Multiple videos | One project can own multiple assets; compositions reference project segments across assets |
| Identities | Project people/memberships, post-publish sync, public list/reassign API and explicit membership assignment in the project tab |
| Reconstruction result | Compact Scene control/read model plus immutable SHA-256 artifacts; Vue reads bounded time/frame windows |

Global scene/video/catalog routes and scene-level match mutation routes do not
exist. Every editor resource is addressed through its owning project.

## Core hierarchy

```text
Project
├── one canonical Match (optional while the project is a draft)
├── many VideoAssets
│   └── many Segments
│       └── versioned SceneAnalysis results
├── many ProjectPeople
└── many AnalysisRuns
```

`Project` is the user's workspace. `Match` is football-domain data. A
`VideoAsset` is one uploaded recording. A `Segment` is a bounded interval in
one asset. `SceneDocument` contains camera, detection, tracking, ball, action,
and QA summaries for one segment. A multi-angle composition references
segments and never owns copies of their source media.

The same match may be referenced by more than one project. A project may also
exist without a match while footage is being ingested.

## Ownership rules

| Scope | Authoritative data |
| --- | --- |
| Project | title, active segment, processing defaults, assets, project people |
| Match | teams, roster, lineup, substitutions, events, score, competition |
| Provider integration | external ID mappings, source revision metadata, fetch status, credentials scope |
| Video asset | media metadata, processing state, detected source segments |
| Segment | source range, replay/live classification, match clock mapping |
| Scene analysis | calibration, observations, tracklets, ball, actions, QA |
| Analysis run | phase telemetry, diagnostics, terminal history, input revision |

Pitch calibration, camera orientation, the visible goal side, and tracks are
segment-local. Roster players and confirmed real-world identities are
project/match data. A segment tracklet may point to a `ProjectPerson`; an
unconfirmed detector observation must remain local until there is enough
evidence to propose a cross-segment link.

## Provider-neutral match contract

The browser and persisted project model use internal IDs only:

```text
upstream API -> provider adapter -> validation -> normalization
             -> external-reference resolution -> canonical MatchSnapshot
```

Provider names and upstream IDs live in integration tables. They do not appear
in the normal `Project` or `CanonicalMatch` response. This is not the same as
discarding provenance: the server retains canonical source revisions and
mappings so it can refresh and audit the selected source. Retention of the full
raw upstream JSON is not implemented yet.

Canonical snapshots are revisioned and immutable from a reconstruction's point
of view. Manual scene corrections remain separate. Refreshing match data
creates a new snapshot and never starts reconstruction automatically. The
immutable reconstruction input contains only
`matchSnapshotRef { id, contentHash, schemaVersion }`; the compact job pins its
fingerprint, while roster, events and provider metadata are never copied into
`SceneDocument`. A worker resolves that exact snapshot through the compact,
unique `ProjectScene` ownership link and fails closed if ownership or the hash
fence does not match. `AnalysisRun` never selects, gates or claims work.
`SceneRepository` owns only revisioned scene-document reads and CAS writes;
`ReconstructionRunRepository` owns queue, claim, lease, heartbeat, progress,
terminal publication, and recovery. Neither repository discovers ownership by
scanning scene payloads: scheduler commands resolve the compact project and
segment context through `ProjectResourceRepository` inside the same
transaction. This repository intentionally remains one transactional
composition owner: enqueue, claim/reclaim and terminal publication must fence
the compact job, lease, Scene revision and telemetry in one commit and one
consistent `job -> lease -> scene` lock order. It is not a domain god object:
`reconstruction_run_contract.py` owns pure run/fence validation,
`reconstruction_run_scene_transition.py` owns the immutable Scene transition
to `processing`, and `reconstruction_run_queries.py` only builds SQL statements
without opening sessions. No other module consumes those SQL builders, so
transaction execution and atomicity cannot drift into a second repository.
The project-scoped
HTTP boundary compares the stored fingerprint with the current compact
snapshot reference and exposes `inputState: stale` plus an explicit rebuild
action. Identity review and mutation endpoints receive the normalized snapshot
explicitly; the browser obtains match content only from the Project Match API.
A review request is assembled by `identity_review_projection.py` from the
checksum-verified identity artifact; observation and canonical-person shaping
live in their own pure projections. `identity_review_http_presenter.py` adds
project-scoped crop links, while `identity_review_crop_service.py` alone owns
frame-path resolution and image encoding. The route publishes a strict
capability-specific response contract. The identity-diagnostics reference in
the reconstruction artifact manifest is the capability boundary: its absence
is a normal query state exposed as HTTP 200 plus typed `availability`, while a
published reference that cannot pass immutable artifact validation fails with
HTTP 503. HTTP 409 is reserved for an actual conflicting command and is never
used for read-model readiness. Roster readiness comes only from the structured
`rosterQuality` fields; provider warning prose is display-only.
A generalized dependency/stale graph for compositions and future derived
artifacts remains part of the canonical artifact migration.

## Analysis job invariants

1. One explicit request queues exactly one segment or one explicitly selected
   multi-angle composition. Project siblings are never implicit work.
2. Live phase progress is stored outside the multi-megabyte scene document.
3. The terminal result is published once through the existing fenced CAS.
4. Queue ownership is protected by a database lease and input fingerprint.
5. `cancelled` is distinct from `failed`; reconstruction cancellation updates
   the compact job, lease, Scene status and telemetry in one transaction. This
   preserves the last good result, enables immediate retry, and prevents later
   publication by the cancelled worker.
6. The API publishes a compact reconstruction job and the dedicated configured
   claimant executes it with a bounded worker count. Each job is a
   supervised child process; cancellation terminates and reaps that process
   before the slot is assigned to a retry. Navigating to another project does
   not create more work or change the queued segment.

Video processing and multi-pass use the same control-plane principles through
`pipeline_jobs`. The API transaction publishes a job and its `AnalysisRun`
telemetry pair, but only the dedicated pipeline runner claims work. Claim,
heartbeat, progress, wait/yield, cancellation and terminal transitions are
lease-fenced. Multi-pass queues durable child reconstruction jobs, yields while
they are active using compact status rows, and reads dense Scene/artifact data
only after every child is terminal. Its final Scene and terminal job state are
one transaction.

`PipelineStore` owns only compact job discovery, claims, leases, heartbeat,
progress and wait/yield transitions. Resource-aware failure and cancellation
are atomic application commands in `PipelineTerminalService`; first-party
runner and HTTP consumers call that boundary directly. It fences the compact
job and publishes the corresponding `VideoAsset` or multi-pass `Scene` state in
one transaction, so the scheduler repository never imports resource models.

Detection-model comparison is also a durable `pipeline_jobs` kind. The HTTP
request only enqueues work and returns `202`; both model passes run in the
pipeline runner. The compact report is revision- and lease-fenced when it is
published to the Scene, while Vue follows the shared `AnalysisRun` telemetry
and reloads the result only after the run becomes terminal.

Derived video media is immutable. Each pipeline lease writes a distinct
`.pipeline-runs/<generation>` directory; the fenced terminal transaction selects
that generation in `VideoAsset.generation_key` while publishing the Scene graph.
It never renames or overwrites canonical files inside a database transaction.
An interrupted worker therefore leaves only an unreferenced generation suitable
for later garbage collection.

## Reconstruction artifact boundary

`SceneRow.payload` is the compact control/read model. It owns identity labels,
counts, QA summaries, last-accepted result status and one versioned
`artifactManifest`; it never owns frame-series results. The immutable
content-addressed artifacts are:

| Manifest key | Content |
| --- | --- |
| `identityDiagnostics` | Full ReID/OCR/resolver evidence used by review tooling |
| `identityTimeline` | Track keyframes/observations and canonical-person observations |
| `ballTrajectory` | Active/automatic/manual ball samples and full resolver diagnostics |
| `calibrationFrames` | Per-frame calibration hypotheses and ball-detection frame evidence |

Each reference carries kind, schema version, byte size and SHA-256. A consumer
must validate all four values and fails explicitly on a missing or corrupt
artifact. Backend code may call `hydrate_scene_reconstruction` for a temporary
in-memory working document, but must call
`publish_dense_reconstruction_artifacts` before persisting a result. There is
no embedded-field read fallback and no dual write.

Artifact behavior is split by reason to change. Identity diagnostics/timelines,
ball trajectories and calibration frames each own their publication contract;
`reconstruction_artifact_hydration` owns strict manifest loading and temporary
materialization. `reconstruction_artifact_publication` is the only bulk
coordinator: it publishes the three materialized domains first, then replaces
the manifest and compact Scene projections together. Capability-specific
commands import the relevant owner directly (for example, a manual ball edit
does not pass through the bulk coordinator). There is no aggregate artifact
service facade.

The 2026-07-18 live cutover reduced Shot 2 from 4,468,127 to 332,876 bytes
(92.55%) during artifact publication. After the obsolete fields were removed,
all eight persisted Scene payloads measured 81,565 bytes total and the largest
measured 70,472 bytes. The exact control-plane traffic sample is recorded in
[`PERFORMANCE.md`](PERFORMANCE.md).

The browser loads
`/api/projects/{projectId}/scenes/{sceneId}/reconstruction-series` in windows
of at most 30 seconds (and at most 900 frames when a frame range is supplied).
Repeated live status polling reads only
`/api/projects/{projectId}/analysis-runs`; a Scene and its series are reloaded
only after an observed active-to-terminal transition for the current run.

Prototype development data is re-imported or explicitly reset after approval;
startup never derives normalized ownership, match snapshots or jobs from Scene
JSON.

## Canonical boundary rules

Architecture changes follow these rules:

1. move the current Vue and worker consumers to normalized project, segment,
   job and artifact contracts;
2. make the new contract authoritative before accepting writes through it;
3. remove the superseded global routes, hydrated compatibility fields and
   duplicate storage rather than maintaining them indefinitely (startup domain
   backfill has already been removed);
4. reset or explicitly re-import local development data when that is safer and
   simpler than preserving prototype schemas;
5. keep only deliberate product recovery paths with explicit
   quality/provenance metadata; delete obsolete compatibility branches and
   their tests;
6. never leave two writable sources of truth after a cutover.

## Code module boundaries

The runtime is organized around composition roots and cohesive domain modules,
not framework-sized god files.

- `app.main` only constructs FastAPI, installs middleware/error handlers, and
  includes routers. It owns no endpoint or domain workflow.
- HTTP routers are split by resource/capability (`video`, scene document,
  analysis, identity, calibration, match import). They translate transport
  errors and delegate work; they do not become persistence repositories.
- Project ownership is resolved in one resource-access boundary before a scene
  or video reaches a route. A route must not reproduce ownership queries.
- Import normalization is transport- and persistence-agnostic. Canonical match
  persistence is a separate application step.
- Transport contracts are strict and capability-specific. Match, project,
  scene, calibration, ball and action DTOs live with their capability; there
  is no generic `schemas` module or browser `types`/`api` barrel. Project
  lifecycle, normalized Match/snapshot, project identity, Segment and
  AnalysisRun DTOs have separate owner modules. `project_contract_base` owns
  only the shared fail-closed serialization policy; it is not a re-export
  registry, and `project_schemas` does not exist.
- A match provider has three one-way owners: HTTP/cache transport, pure mapping
  into canonical data, and provider orchestration. Provider selection is an
  explicit request/configuration value, never task-local global state.
- Video ingest separates FFmpeg invocation, pure segment planning, generation
  materialization and atomic pipeline publication. Dense media never travels
  through Scene JSON or scheduler rows.
- Reconstruction exposes one explicit public package surface. Detection,
  calibration, tracking, identity correction, ball analysis, progress, and
  execution are internal cohesive modules and may not reach back into HTTP.
- Reconstruction has no generic domain aggregate. Fail-closed errors, the
  detector observation DTO and passive track-state DTO have independent
  contract owners. `TrackState` contains data and derived read-only
  properties only: validated observation mutation lives in
  `track_observation_accumulator`, while ReID quality scoring, bounded sample
  selection and role voting live in `track_reid_accumulator`. Tracking and
  identity capabilities import these owners directly; no facade re-exports
  them.
- Reconstruction execution is a thin phase composition root. Sampled-frame
  preparation/scan, temporal calibration, dense ball analysis, representative
  calibration selection, boundary-result projection, identity resolution and
  publication payload construction have separate owners. Dense-ball progress
  and failure behavior are immutable job inputs captured when the run is
  queued; this phase never re-reads mutable live settings.
  Single-frame review likewise composes camera context, ball inference and
  canonical-identity projection instead of duplicating the batch pipeline.
- Model backends expose a provider-neutral contract plus an explicit factory.
  Ultralytics and WASB candidate parsing, transports and inference adapters are
  separate; reconstruction depends on the contract and records effective
  backend/fallback provenance. Worker images use multipart/binary transport;
  frame bytes are never base64-encoded into JSON or scheduler state.
  The WASB worker further isolates its immutable provider DTOs, strict asset
  loader, model configuration, image geometry and thread-safe runtime; its HTTP
  root never imports the concrete provider directly.
- Calibration acquisition, temporal resolution, metric projection, evidence
  serialization, quality policy and preview persistence have separate owners.
  Evidence/quality modules are pure; only preview/application services may
  persist a calibration edit. Temporal resolution itself is split into an
  immutable result contract, camera-motion homography paths, normalized DLT
  consensus, hypothesis construction and the final selection policy; the old
  all-in-one temporal solver module is not an import surface.
- Dense-ball detection is a state-machine coordinator over four direct
  capabilities: frame-source selection, primary/fallback execution, evidence
  materialization and cache/checkpoint lifecycle. Cache identity and strict
  envelope decoding are pure modules; filesystem locking and atomic replace
  remain together so the partial-prefix/complete-publication race has one
  owner.
- Dense-ball projection has no aggregate facade. Its immutable frame context,
  bounded homography math, temporal camera/calibration resolver, candidate
  metric materializer and publication-status rule are separate one-way
  capabilities. The resolver records exact/interpolated/fallback provenance;
  the materializer consumes that decision but never chooses calibration.
- Jersey OCR separates evidence contracts, conservative temporal fusion and
  review-only roster ranking. Its API client composes independent fail-closed
  validators for readiness/model identity, individual OCR evidence
  (candidate, quality and polygon), and batch identity/diagnostics; there is no
  aggregate protocol facade. Reconstruction separately owns partition-aware
  crop sampling, worker inference and immutable-observation reassignment after
  split/merge, so neither a crop policy nor an outage can bypass identity
  invariants.
- Identity correction has no aggregate graph/association facade. Annotation
  semantics, lineage graph, validation, undo, roster decisions, raw-track
  merging, split partitioning, scene-document corrections, persistent-ID
  assignment, closed-set roster resolution and API projections are explicit
  one-way capabilities. Bind/Unbind is the sole roster-identity write path;
  generic frame annotations cannot carry an external player ID. Roster edits
  have no persistence switch: pure Set/Unbind and Clear planners mutate only a
  hydrated draft; the queue route uses the draft owner before its fenced CAS,
  while standalone commands always publish dense artifacts and persist the
  scene. Split/merge undo lineage is validated by the pure Clear planner before
  any mutation becomes durable.
- Canonical-person publication is not one broad document builder. Observation
  and evidence projection, one-person document projection, publication
  diagnostics and closed-set roster orchestration are separate one-way
  capabilities; consumers import the orchestration use case directly.
- Frame annotation editing has no persistence switch or aggregate module.
  Sampled-frame/image targeting, pure upsert planning, pure split/merge undo
  planning, the upsert/delete commands, and the shared artifact/repository
  commit boundary are separate capabilities. Commands always persist; editor
  queue composition drafts an in-memory mutation and publishes it through the
  reconstruction queue CAS, while domain tests call the pure planners.
- Reconstruction queueing has no persistence switch. The pure
  `reconstruction_queue_draft` capability receives an explicit resolved model,
  detector input, snapshot reference, frame count and run ID and returns a new
  queued Scene without storage access. The `reconstruction_queue` command alone
  hydrates and publishes immutable dense artifacts, then atomically commits the
  compact Scene, job and telemetry through `ReconstructionRunRepository`.
  Project/match transaction composition and domain tests use the draft directly
  only when they own the surrounding write; normal callers use the command.
- Player-action edits follow the same explicit command boundary: storage-free
  in-memory upsert/delete planning is independent from the always-persist
  commands. Render-track IDs from prototype scenes are not accepted as a
  fallback for missing canonical people.
- Persistence modules separate repositories (queries and atomic writes) from
  application services (cross-repository orchestration). A broad store object
  must not accumulate unrelated project, match, analysis, and job behavior.
- Project identity persistence loads and locks aggregate rows in
  `ProjectIdentityRepository`, delegates merge/preserve/orphan policy to the
  pure `project_identity_reconciliation` planner, and applies the returned
  mutation plan atomically. Provider-neutral read documents are built by
  `project_identity_projection` from one people query plus one batched
  memberships query; projection code performs no SQL and list reads are never
  N+1.
- `ProjectStore` owns only project-header lifecycle and revision CAS.
  `ProjectResourceRepository` is the sole runtime boundary for unique
  Scene/VideoAsset ownership, Segment metadata and transaction-aware
  reconstruction context; pipeline services call its in-transaction methods
  instead of writing membership tables themselves.
- Multi-angle analysis has no aggregate service-locator module. Composition
  creation, durable job transitions, temporal alignment, evidence fusion,
  scoring, progress, and terminal publication are separate capabilities;
  numerical alignment/fusion modules do not import persistence.
- Reconstruction QA crosses three explicit pure boundaries: typed evidence
  collection (`quality_measurements`), stable metric serialization
  (`quality_metric_report`), and versioned gate assessment
  (`quality_gate_report`). `quality_metrics` is only the artifact-hydrating
  report use case; it owns no measurement algorithms or thresholds.
- Reconstruction track publication has no shared track-helper aggregate.
  Detector-box overlap, image-to-pitch projection, latent off-camera presence,
  authoritative observation merge, deterministic canonical-person IDs and
  trajectory filtering each have direct capability owners. The trajectory
  materializer depends on projection and latent-presence policy in one
  direction; observation publication does not leak back into those algorithms.
- Vue's root component is only a route-aware composition shell. Feature state
  and effects live in typed composables; rendering and controls live in feature
  components. A composable must not merely hide the previous god component in
  one equally large file.
- The editor route composes five explicit UI domains: project/scene session,
  viewport/playback, analysis/calibration, scene composition, and
  identity/match. `EditorWorkspaceSurface` consumes those typed contexts and
  owns layout only. Domains receive their dependencies directly; callback
  registries, generic save-hook registries and editor-wide service locators are
  forbidden.
- The editor session keeps its writable active Scene distinct from the
  read-only asset timeline Scene. A segment route saves and reconstructs only
  its child Scene while the master timeline continues to read the asset's
  explicit `timelineSceneId`; the timeline projection is never a second write
  target.
- The project workspace is a thin coordinator over catalog, match, media,
  project-identity and analysis-job resources. Consumers address the owning
  resource directly; the coordinator does not re-export a flat writable store.
- Browser transport is split by capability over one small request boundary,
  and domain DTOs are imported from their owning type modules. Compatibility
  barrels such as `lib/api.ts` and `types.ts` are not part of the architecture.
- The editor viewport composes independent video-review, frame-detection,
  calibration and 3D panes. `ThreeViewport.vue` adapts Vue lifecycle only;
  renderer/camera/controls, pitch, players, ball, markers and pointer selection
  are framework-independent owners with explicit disposal.
- Frame-analysis commands do not own detector-selection policy. Pure canonical
  identity/track selection lives in `frameAnalysisSelection`; frame annotation
  drafts, pointer gestures and remote mutation orchestration are three separate
  capabilities. Manual-ball trajectory normalization is likewise independent
  from optimistic persistence and editor selection state.
- Calibration QA and identity review surfaces consume focused presentation
  projections. Manual roster-picker validation owns its own reactive state;
  the review component only translates user intent to explicit emitted
  commands. Opening a project or segment creates no implicit actor selection
  and performs no identity-review query. The browser loads review evidence only
  when the user explicitly selects a canonical person, the Binding inspector is
  visible, and the active Scene advertises an identity-diagnostics artifact;
  requests are deduplicated by Scene revision and artifact identity. Large
  scoped templates and styles are not split solely by line count when their
  script and public UI contract remain cohesive.
- Worker `main` modules construct HTTP applications and translate errors only.
  Request validation, cache policy, model loading and inference live in worker
  capability modules, which are testable without FastAPI. Ball, identity and
  jersey workers expose provider protocols/DTOs independently from concrete
  WASB, PRTReID, MMOCR or EasyOCR runtimes; factories are composition roots,
  not service dependencies.
  The calibration worker likewise has no aggregate engine facade: immutable
  engine DTOs/protocols, frame decoding, pinned PnLCalib runtime/model loading,
  heatmap inference, geometric projection/quality gating, LRU/TTL caching and
  lock-aware batch orchestration are separate owners. Its HTTP root selects the
  concrete engine factory; the application service depends only on the engine
  provider protocol.
- Model-validation manifests have no aggregate parser facade. Immutable
  contracts, strict JSON/path primitives, crop labels, identity pairs,
  thresholds and content fingerprinting have independent owners;
  `manifest_loader` only composes them. Unknown fields, dataset-size limits and
  paths escaping the manifest directory fail closed before worker inference.

Architectural tests protect dependency direction: route modules never import
the FastAPI composition root, and pure normalizers never import HTTP or
persistence layers. File length is a diagnostic rather than the rule: a long,
cohesive numerical algorithm can be valid, while a shorter file with several
independent reasons to change is still split.

Historical prototype-data preservation is a separate, explicit product
requirement; it is not assumed by default.

## UI information architecture

The intended complete navigation is:

```text
Projects
└── Project
    ├── Match
    ├── Media & Segments
    ├── Timeline
    ├── Editor
    ├── Identities
    ├── Analysis Jobs
    └── Settings
```

The project workspace always has an explicit `projectId`, and Vue Router is the
source of truth for project, tab and editor resource. The `/projects`
collection has no selected project and does not use remembered or match-backed
preferences; only an explicit navigation or deep link establishes project
context. A video timeline route is
addressed by normalized `assetId`; `ProjectAsset.timelineSceneId` resolves its
internal root scene without scanning or selecting the first project scene.
Selecting a concrete moment is a separate segment route. A generic scene route
is reserved for compositions. The global **Open
editor** action opens the only ready timeline; with multiple videos the user
chooses **Open timeline** on an asset card. It must never fall through to an
arbitrary segment scene. The Match tab uses the canonical snapshot and remains
usable offline after a successful sync. Integration diagnostics may show
provider provenance in an administrative view, but provider selection is not a
project-domain concern. The current project shell implements Overview, Match,
Identities and Analysis; composition authoring, project Settings and a fully
separated match-level Timeline remain pending.

## Quality gate

Performance changes to calibration, detection, tracking, identity, or ball
analysis are accepted only against a versioned labelled benchmark. Required
measurements are calibration reprojection error, detection precision/recall,
track identity metrics (IDF1/GS-HOTA where labels permit), fragmentation,
metric projection error, ball recall/error, and identity accuracy plus
abstention coverage. The evaluator and schemas exist, but the current
Spain–Belgium manifest has no reviewed gold labels, so it cannot yet enforce a
real-model product gate.
