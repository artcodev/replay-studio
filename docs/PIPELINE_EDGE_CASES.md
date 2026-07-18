# Video-to-game-state: edge cases and response policy

This document is the operational checklist for the Replay Studio pipeline. It
separates a visual continuity estimate from measured evidence: an actor may stay
visible for the whole scene while their off-camera position remains explicitly
`inferred`, low-confidence, and excluded from accuracy metrics.

## Non-negotiable invariants

1. Raw detections, calibration candidates, association edges, manual decisions,
   and the accepted reconstruction are separate versioned artifacts.
2. A manual decision has higher priority than model output and records its scope:
   one observation, a time range, or the complete identity.
3. An actor never disappears from an accepted moment because detector confidence
   dipped. Its display track covers `t=0..duration`; unknown intervals are latent
   state with uncertainty, not invented observations.
4. Inferred positions never count as detector recall, calibration support,
   continuity, or physics evidence.
5. Camera cuts are hard temporal barriers. Optical flow, homographies, and local
   track IDs do not propagate across them without an explicit cross-shot match.
6. Every destructive correction has preview and provenance. Save/delete is an
   atomic correction+queue transaction; deleting the correction is the current
   undo operation and deterministically rebuilds from the previous evidence.

## Edge-case matrix

| Priority | Edge case | Failure mode | Automated response | Editor response | Metric / acceptance check |
|---|---|---|---|---|---|
| P0 | A real person is missed on one frame | Player flashes out or never becomes a track | Keep low-score candidates near an existing track; retain latent actor state through the gap | Draw/select the person, **Confirm**, then bind role/identity | Far-side recall; correction and queued rebuild publish atomically; confirmed detection must enter the next rebuild |
| P0 | Phantom person | A spectator, board graphic, goalpost, or grass patch becomes a player | Require temporal/field support before creating an automatic actor | **Ignore** one detection or the full identity | Observation ignore removes only that evidence; identity ignore resolves a unique raw track or fails safely; zero silent/wrong deletion |
| P0 | One person gets several IDs | Duplicate players occupy nearby positions | ReID/team/trajectory hypothesis with an acyclic identity graph; ambiguous geometric remap is rejected | Select source IDs and **Merge into…** canonical actor | ID switches and duplicate-assignment frames; overlap seconds only with timestamps or explicit FPS; merge is idempotent and preserves last-good on ambiguity |
| P0 | Two people are merged | One actor teleports between nearby players | Preserve competing association hypotheses at crossings; reject impossible speed; split recomputes partition-local appearance, role, manual and roster semantics | Split at the selected frame/range and confirm each branch | Association accuracy; speed/overlap gates; no inherited evidence or roster binding on the wrong partition |
| P0 | Different confirmed roster players are merged | Two known players collapse into one canonical actor and one roster identity wins silently | Treat distinct confirmed `externalPlayerId` values as a hard conflict at save, raw-track and published-scene stages | Hide incompatible merge targets; stale submissions show a conflict and preserve last-good | Zero accepted merges with conflicting confirmed roster IDs |
| P0 | Long occlusion or person leaves the crop | Player disappears, then appears from nowhere | `continuous-latent` presence: interpolate bounded gaps; before/after observations roam within 0.65 m of the nearest known point; uncertainty grows | Inspector distinguishes observed vs inferred and can anchor a missing interval | Presence coverage is 100%; observed coverage is reported separately |
| P0 | Calibration is mirrored | Attack and every projected position are on the wrong half | Score both semantic left/right hypotheses; use independent direct anchors and event-side priors only as veto/support | Set attacking direction separately; manual pitch anchors correct geometry | Side agreement, semantic-line residual, event-zone consistency |
| P0 | Only part of the pitch is visible | Four-corner homography fails or chooses a plausible wrong rectangle | PnLCalib points+lines hypotheses, semantic line intersections, temporal evidence from later frames | Calibrate current frame and adjust numbered semantic anchors | Reprojection p50/p95, line F1, independent-anchor coverage |
| P0 | Pan/zoom between frames | A single homography drifts and players slide | Per-frame camera candidates plus forward/backward temporal graph and uncertainty | Add manual anchor frames; interpolate only within the same shot | Calibration max gap; camera-motion residual; drift at manual anchors |
| P0 | Replay/cut/transition | Tracks or calibration cross into another shot | Detect cut/fade/scoreboard transition and reset all shot-local state | Explicitly group replay passes after temporal alignment | Zero propagation paths across a cut; shot-boundary precision/recall |
| P0 | Late/stale worker result | An older run overwrites newer edits | **Implemented:** assign a unique `runId`, monotonic `runRevision`, and SHA-256 `inputFingerprint`; publish progress/final/failure through atomic compare-and-swap on the active run and unchanged inputs | A superseded run exits without writing; retain the last accepted run | Integration tests cover two runs, edit-during-run, stale queue snapshots, and stale failure; no mixed revisions |
| P0 | API/reconstruction runner crashes after a scene run was queued or claimed | Persisted correction remains forever at `queued`/`processing`, or two processes analyze the same scene | API atomically persists a compact reconstruction job and never executes it; the dedicated runner continuously claims queued/stale jobs with a process-independent owner lease; every publish is fenced by lease + run + input + revision | Compact `AnalysisRun` progress resumes after TTL; active leases are not stolen; a delayed old owner and duplicate terminal task are no-ops | Compact-polling, two-engine queued/reclaim, heartbeat, active-not-stolen, old-owner fencing and terminal cleanup tests |
| P0 | Cancel races with final reconstruction publish or retry | A user sees `cancelled`, but a late worker replaces the accepted scene, the old native inference still occupies the only slot, or an already accepted `ready` scene contradicts its job | In one DB transaction mark the reconstruction run and matching scene `cancelled`, delete its lease, restore `frames-ready`, and advance the scene revision; final publication checks the run under the same write lock; the dedicated runner terminates and reaps the fenced child process before retry; an already accepted terminal scene wins and reconciles the job | Analysis immediately releases both the logical lock and physical runner slot; `cancelling` remains visible only for cooperative job kinds | Atomic cancel-before-publish, ready-before-cancel, cancel-then-requeue, SQLite two-connection and blocking-process regressions; zero successful publication by the cancelled run |
| P0 | Partial worker failure | New tracks combine with old calibration/metadata | Stage immutable artifacts, validate, then atomically publish the run | Keep last successful scene and expose failed phase/log | Failure-injection tests at every phase |
| P1 | Goalkeeper/referee wears an unusual kit | Team clustering excludes or mislabels a valid person | Role classifier and spatial/roster priors remain separate from kit clustering | Assign goalkeeper/referee/other explicitly | Per-role precision/recall, not only player recall |
| P1 | Roster has duplicate/similar names or unknown shirt number | Wrong external player binding | Up to five quality-ranked/temporally diverse OCR crops; reliable only after repeated agreement; roster candidates are never auto-bindings; confirmed binding follows its anchor partition through split | Dedicated canonical **Bind / Unbind** works off-screen from saved detector evidence and validates team/duplicates; the next binding edit on the split owner rekeys the existing correction instead of duplicating it | OCR reliable/provisional/conflict counts; one durable binding correction after split/rekey; worker outage/rebuild survives; stale publish rejected |
| P1 | Roster binding bypasses canonical commands | A generic frame edit assigns a real player without team, duplicate-owner or split-lineage validation | Frame annotations cannot carry roster IDs; dedicated Bind/Unbind/Clear is the sole write boundary and incompatible owners abort | Bind/unbind only from canonical inspector; generic editor changes role/label/identity graph | Zero generic roster writes; zero duplicate roster owners; split-local binding tests |
| P1 | Roster player is unbound | The negative edit is counted as a positive manual confirmation and falsely resolves the actor | Persist an observation-anchored tombstone for deterministic remap, but exclude it from positive evidence and manual metrics | Show the player as unbound while retaining their provisional identity state | No manual evidence/resolved status/confidence boost from tombstone; external ID is null |
| P1 | Substitute enters or a player truly leaves | Full-match continuity invents a player before/after participation | Apply 0–100% presence only to a short accepted moment; for long clips use lineup/substitution lifecycle gates | Mark active interval or use match event | Presence outside participation interval is zero; event-time tolerance |
| P1 | Person is outside the field (warm-up, medic, coach) | Clamp turns sideline staff into an on-pitch player | Keep unclamped image observation; require field/role evidence before metric projection | Mark `other` or ignore | Boundary/clamp ratio; outside-field false positives |
| P1 | Player is airborne/sliding | Bounding-box foot point is not the ground contact | Temporal ground-contact filter and pose/segmentation cue; keep uncertainty | Optional manual ground point | Pitch error during jumps/slides vs ordinary running |
| P1 | Ball is confused with a head, boot, logo, or line | Ball teleports or follows a player | Ball-specific detector, motion prior, size/shape gating, possession/event hypotheses | Confirm/reject ball candidate on key frames | Ball recall, acceleration/velocity violations, possession consistency |
| P1 | Motion blur, compression, interlace, duplicated frames | Missed/duplicate observations and bad optical flow | Decode by PTS, deinterlace when detected, frame-quality score, skip duplicates | Show unusable-frame warning | PTS monotonicity, duplicate ratio, blur distribution |
| P1 | Letterbox/crop/resolution changes | Annotation and calibration coordinates shift | Store source dimensions and normalized coordinates; one explicit image transform chain | Zoom/pan changes view only, never annotation coordinates | Round-trip screen↔image coordinate tests at every zoom |
| P1 | Manual confirm/ignore collides with an active rebuild | Decision is lost or partially applied | Active mutation returns 409; accepted save/delete and queued rebuild publish in one fingerprint-guarded CAS | Disable conflicting action while processing, then poll the returned run | Concurrency and atomic correction+queue integration tests |
| P1 | Identity/manual edit reruns unchanged person inference | Corrections are slow and transient detector output may drift | Persist the complete pre-annotation person + sampled generic COCO-ball artifact per exact JPEG/model/NMS/filter/schema contract; corrupt/tampered data is a miss and only primary-complete output is atomically published | Show cache hit/miss/write/error/provider-call diagnostics | Identical rebuild has zero base provider calls; one changed JPEG has one miss; model/policy/corruption tests |
| P1 | Merge cycles or self-merge | Identity graph cannot resolve deterministically | Canonical union-find target, reject self/cycle, flatten aliases on write | Explain conflict and keep previous state | Property tests: acyclic, idempotent, order-independent |
| P1 | Identity remap is missing or ambiguous after detector/tracker changes | Exclude/merge silently affects no one or the wrong nearby player | Exact annotation first; strict metadata/trajectory gates; fail closed with ranked candidate diagnostics | Show correction ID, reason and candidate residuals; keep last-good tracks | Missing/ambiguous/threshold tests; zero silent correction |
| P1 | Multi-angle clocks do not align | Same event/person is fused at wrong time | Validated manual anchors or accepted automatic alignment; identity requires same external player or reliable jersey, never proximity alone | Adjust temporal anchors per pass | Alignment residual/overlap; conflicts and ambiguity abstain; foreign observations remain namespaced |
| P1 | API/runner crashes during video ingest or multi-pass | Compact job remains running even though no process owns it | API only enqueues; the dedicated pipeline runner claims with expiring leases, heartbeats, recovery and fenced terminal publication. Multi-pass children are durable reconstruction jobs and are excluded from duplicate recovery while owned | Show queue/lease/retry state and preserve last accepted artifacts | Lease/recovery/cancellation/duplicate-owner tests; pending multi-host backpressure and SLO drills |
| P1 | Similar kits / lighting changes | Team label flips within a track | Track-level color embedding, white balance normalization, roster/formation prior | Lock team for canonical identity | Team-switch count per identity |
| P2 | Scoreboard/advert overlay covers the pitch | Overlay edges become pitch lines or people | Detect static graphics mask and exclude it from geometry/detection | Edit mask for a source video | False line/keypoint support inside mask |
| P2 | Camera shake or rolling shutter | Global homography is locally inconsistent | Robust local features and short uncertainty spike; consider mesh warp only if benchmark proves value | Add an anchor after the shake | Spatial residual map, not only one mean error |
| P2 | Celebration/crowd enters field | More than 22 people breaks team capacity heuristics | Phase-aware capacity; preserve `other/unknown` candidates instead of silent deletion | Classify or hide non-participants | Recall does not drop solely because capacity is reached |
| P2 | Pitch dimensions differ | Positions are systematically scaled | Use competition/venue metadata or explicit dimensions; never silently assume if known | Edit pitch dimensions | Known-line distance residual |

## Continuous actor presence contract

Each player keyframe contains two independent notions:

```json
{
  "t": 1.6,
  "x": 8.4,
  "z": -3.1,
  "confidence": 0.18,
  "observed": false,
  "presenceState": "inferred-gap",
  "projectionSource": "presence-inferred",
  "positionUncertaintyMetres": 4.7
}
```

Allowed states are `observed`, `inferred-before-first`, `inferred-gap`, and
`inferred-after-last`. The first and last display keyframes are always `0` and
`scene.duration`. Inferred samples are bounded to the pitch and deterministic so
the same inputs produce the same replay.

This policy solves visual popping, but it does **not** prove where an off-camera
player was. The 3D inspector should eventually visualize uncertainty (for
example, a subtle ground halo), while labels and QA use observed evidence only.

For a full match, continuous presence must be constrained by a participation
interval derived from lineup/substitution events or a manual lifecycle. A player
who has not entered the match is not an off-camera actor.

## Manual identity operation semantics

- **Confirm**: convert a selected model candidate into persisted positive
  evidence. It may create an actor even when automatic support is below the
  normal acceptance threshold.
- **Ignore**: create negative evidence. Scope must be explicit; one-frame ignore
  must not silently erase an otherwise valid full track.
- **Merge**: map one or more aliases to a canonical actor. Observations are
  combined, timestamp collisions prefer manual evidence and then higher model
  confidence, and impossible-speed conflicts remain visible for review.
  Different confirmed external roster IDs are a hard conflict: the editor hides
  the target and every server-side application path fails closed.
- **Split**: break a canonical identity over a half-open `[start, end)` range.
  The selected immutable observation is stored with frame/time/bbox evidence;
  detector reorder is remapped only through a unique strict geometry match.
  Missing, ambiguous or recycled-ID targets fail closed, and a persisted
  cannot-link barrier prevents the global identity resolver from reconnecting
  the two partitions. Appearance, ReID role votes, manual semantics and confirmed
  roster binding are recomputed from each partition's observations. The binding
  follows its anchor and its correction is rekeyed if the new partition becomes
  the canonical owner. The editor previews affected and remaining observations.
- **Unbind roster player**: keep a durable observation-anchored tombstone for
  rebuild/remap, but do not count it as positive/manual identity evidence or use
  it to resolve the person. It clears the external player ID without creating a
  contradictory second correction.
- Implemented operations store authoring time, affected IDs, action and scope.
  Deleting the persisted correction is the current reversible undo; a future
  multi-step edit history may additionally store previous-value snapshots.

## Metrics that must be shown per run

| Layer | Minimum metrics |
|---|---|
| Detection | Precision/recall on labeled frames, small/far-side recall, false positives per minute |
| Calibration | Direct and temporal coverage, p50/p95 reprojection, semantic-line F1, max gap, side agreement |
| Tracking | IDF1/HOTA or GS-HOTA association component, ID switches, fragments, duplicate-assignment frame count; duplicate-overlap seconds plus explicit timebase only when timestamps/FPS make duration valid |
| Projection | Pitch-position p50/p95 error, clamp ratio, impossible-speed ratio, uncertainty calibration |
| Presence | Display coverage, observed span, inferred ratio, maximum inferred interval |
| Identity | Canonical/source-tracklet counts, resolved/provisional/excluded groups, accepted/review/rejected edges, ambiguous links, ReID usable/selected-independent crops, jersey reliable/provisional/conflict evidence, manual/conflict counts; with ground truth: IDF1/precision/recall, ID switches, fragments and duplicate assignments; HOTA/GS-HOTA only from the official evaluator |
| Runtime | Cold/warm duration per phase, queue time, model load time, cache hit, peak RAM/VRAM |

No single combined score should hide a failed required gate. Processing can be
complete while quality is `review` or `reject`.

## Recommended implementation order

1. Keep regression coverage for the implemented confirm/exclude/merge/split flow and
   revision-safe CAS rebuild publishing.
2. Add a multi-step edit history and optional assisted range propagation across
   long occlusions; persisted correction deletion is the current deterministic undo.
3. Keep base sampled-frame person/generic-ball artifacts limited to pre-annotation
   detections. Persist optional tracker state only if its complete association
   and configuration contract can be versioned without making annotations,
   calibration or identity decisions sticky.
4. Build a 100–300-frame validation set from the current video, including the
   far-side player, crossings, partial-pitch views, goalkeeper, cuts, and ball.
5. Add labelled same/different-player crops and readable/unreadable jersey views;
   run the versioned model-validation harness before changing ReID/OCR thresholds.
6. Benchmark the current short-horizon tracker + SoccerNet PRTReID resolver
   against StrongSORT/TrackLab using the same labelled frames and layer metrics.
7. Add participation lifecycles before applying continuous presence to clips
   longer than a highlight moment.
