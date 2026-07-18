# Canonical player identity

## What this layer fixes

Detector labels such as `person-8`, online tracker aliases and 3D actors are
different things. Earlier they were represented by one `tracks[]` object. This
caused two systematic errors:

1. a valid video person became “not linked” when calibration rejected their
   pitch coordinate;
2. two fragments of the same player could become unrelated 3D actors, while a
   fresh frame analysis generated another unrelated `person-N` label.

The authoritative chain is now:

```text
detection observation → local tracklet → canonical person → optional 3D track
```

`canonicalPeople[]` owns identity. `tracks[]` is only the renderable metric
projection and references its owner through `canonicalPersonId`.

## Scene contract

```json
{
  "canonicalPersonId": "canonical-61fc55b63529",
  "displayName": "Away person",
  "identityStatus": "provisional",
  "identityConfidence": 0.84,
  "identitySource": "reid+trajectory",
  "teamId": "away",
  "role": "player",
  "jerseyNumber": null,
  "externalPlayerId": null,
  "memberTrackletIds": ["tracklet-0003", "tracklet-0011"],
  "observations": [],
  "renderTrackId": "auto-away-02",
  "evidence": [],
  "rosterCandidates": [],
  "conflicts": []
}
```

Every observation keeps a stable ID, source frame/time, original bbox,
tracklet and canonical IDs. Metric projection status is independent:
`accepted`, `rejected` or `unprojected`. Therefore exact video selection works
even if `renderTrackId` is null.

## ReID worker

The primary provider is the SoccerNet PRTReID/BPBreID baseline:

- Python 3.9;
- PyTorch 1.13.1 / torchvision 0.14.1;
- normalized 256D global embeddings;
- role scores for player, goalkeeper, referee and other;
- pinned source commits and MD5-verified official checkpoints.

It is isolated at port `8091`. `/health/live` checks only HTTP liveness;
`/health/ready` verifies assets and constructs the real model. The main API
does not become unhealthy when ReID is unavailable. Instead, it records the
failure, disables automatic cross-gap stitching and publishes local identities
as `provisional`. HSV features remain confined to short online association and
are never compared with the PRT vector space.

Small, out-of-frame or blurry crops return `usable:false`. They are retained in
coverage diagnostics and are not treated as negative identity evidence.

ReID sampling is tracklet-level. The reconstruction keeps quality-ranked,
temporally separated crops instead of the first frames it happens to see. A
strong automatic edge requires independent support in both directions; one
nearly identical crop is review evidence, not permission to merge two people.
Every result also carries a `pixel-evidence-v1` fingerprint computed from the
decoded crop pixels. The same image is therefore one support even if it returns
through another observation ID, HTTP batch, merge or split. The worker uses a
bounded content/model/policy-keyed cache with in-request deduplication and
concurrent single-flight, while the API reports usable versus selected
independent samples and unique versus duplicate pixel evidence. Readiness
requires the fixed backend, dimension, normalization and fingerprint protocol
plus a non-empty model version. The first inference batch then fixes the exact
accepted `modelContract`; any backend/model/version/dimension drift in later
batches fails the whole call instead of mixing embedding spaces. Malformed
item, role, confidence, quality or embedding shapes are likewise rejected
instead of being partially accepted.

## Offline resolver

The resolver receives one immutable summary per local tracklet. It first
applies hard constraints:

- temporal overlap;
- different confirmed manual identities;
- conflicting external player ID or confirmed manual team/role;
- conflicting reliable jersey numbers;
- physically unreachable pitch displacement, including calibration uncertainty.

Automatically inferred team/role disagreement is review evidence rather than a
hard veto: kit clustering can be wrong for keepers, referees and changing
illumination. Confirmed manual semantics remain authoritative.

Pitch distance and team colour can reject a candidate but cannot prove
identity. An automatic edge is accepted only with at least one identity signal:

- strong normalized ReID agreement;
- a reliable matching jersey number;
- the same accepted external roster ID.

Near-equal alternatives are downgraded to review. A Hungarian assignment then
enforces at most one predecessor and one successor. Manual must-link decisions
have the highest priority. Every input tracklet appears in exactly one output
group; unresolved fragments are never silently discarded.

Pairwise compatibility alone is not sufficient: an unlabeled middle fragment
could otherwise connect two incompatible endpoints. After assignment, each
automatic edge is added to the prospective connected component in descending
confidence order. The complete union is rechecked for conflicting manual
identity, external player, reliable jersey, manual team/role and ReID vector
dimension. A bridge that introduces any such conflict is retained as a review
edge, never an accepted identity link.

## Jersey-number OCR boundary

Jersey OCR runs in a separate Python 3.9 worker on port `8093`. Its
`jersey-ocr.v1` HTTP contract is provider-neutral: MMOCR DBNet+SAR is the
SoccerNet-compatible default and EasyOCR is an explicit alternative. Both
produce crop evidence with immutable observation/tracklet identity, raw text,
numeric candidates, confidence, crop QA and fail-closed ambiguity. Missing
models leave the service live but not ready and never invent a shirt number.

Reconstruction chooses up to five quality-ranked, temporally separated views
per tracklet in the normal case. If persisted split ranges exist, it keeps the
same bounded reservoir independently for each prospective partition, avoiding
both unbounded all-frame OCR and loss of the only readable crop in a short
range. The pre-resolver still consumes at most five views; after split/merge,
the retained raw worker results are reassigned through immutable observation
IDs and sampled/fused again per final canonical owner. A number becomes reliable only
with at least two independent agreeing samples and sufficient confidence;
single readings remain provisional, while competing readings fail closed as a
conflict. Reliable tracklet evidence may join otherwise compatible fragments,
but never accepts a roster identity. Evidence from one split partition cannot
leak into another canonical person.

The worker cache is keyed by exact crop bytes, model/checkpoint and OCR policy.
Like ReID, OCR returns a decoded-pixel `pixel-evidence-v1` fingerprint. Fusion
keeps one deterministic best row for identical pixels, so duplicated crop IDs,
batches and merge history cannot manufacture independent jersey votes. Cache
hits, provider inference, request/pixel deduplication and conflicts are exposed
in diagnostics. The client validates the complete status/candidate/reason/
confidence/quality contract and fails closed on an incompatible worker. A
first successful OCR batch fixes the accepted provider/model/capabilities
contract; any drift in later batches fails the whole call, so rolling deploys
cannot silently mix provenance inside one fusion result. A
PARSeq-style joint tracklet provider remains a benchmark candidate; the
provider-neutral contract does not assume MMOCR or EasyOCR is the final
accuracy winner.

## Persisted roster and closed-set player resolver

Real names are resolved only against the project's current canonical match
snapshot. It contains the event, teams, complete player list, lineup roles,
timeline, substitutions and an explicit `rosterQuality` decision. Provider
provenance is stored separately. Scene routes cannot write match data. Loading
an existing scene never silently refreshes data from the network, so one
reconstruction run cannot mix two roster revisions.

TheSportsDB free responses that contain five or fewer than eleven players per
team are marked `partial`. They remain available in the manual Bind selector,
but are not a valid closed set and cannot generate automatic player-name
hypotheses. Refresh the current Project match through:

```http
POST /api/projects/{projectId}/match/refresh
```

If the provider cannot return a complete roster, a strict user-owned JSON
snapshot can be imported through:

```http
POST /api/projects/{projectId}/match/import
Content-Type: application/json
```

Both operations persist a new canonical Project snapshot and do not queue
reconstruction. The changed snapshot participates in the reconstruction input
fingerprint; the user explicitly rebuilds the current single-pass scene or
reruns a composition.
The project includes a validated full Spain–Belgium 2026 fixture at
`data/matches/spain-belgium-2026-qf.json` as an importable example.

For every canonical person, the closed-set resolver combines repeated jersey
OCR, team/role constraints, confirmed manual bindings and (when match-clock
alignment exists) participation windows and player-specific events. One global
Hungarian assignment enforces one player per simultaneous identity and has an
explicit unknown/abstain alternative. A suggestion needs a minimum identity
signal and a margin over the next global solution. Team or role alone can never
produce a name. Confirmed bindings remain authoritative and contradictions are
reported rather than overwritten.

Resolver output is review-only: it always publishes
`autoBindings: []`. `canonicalPeople[].externalPlayerId` changes only through
the dedicated manual Bind endpoint. Rejecting a roster hypothesis is also a
durable reconstruction input:

```http
POST /api/projects/{projectId}/scenes/{sceneId}/canonical-people/{canonicalPersonId}/roster-rejections
{ "external_player_id": "..." }
```

Rejected rows are excluded from later global assignments and the decision is
included in the reconstruction fingerprint, so an older worker cannot restore
the hypothesis.

The read-only review workbench is available at
`GET /api/projects/{projectId}/scenes/{sceneId}/identity-review`. It returns the priority queue,
best observation crops, ReID/OCR readiness and rejection reasons, roster
hypotheses, conflicts and the completeness status of the saved roster. Crop
URLs resolve exact persisted observation bboxes; clients cannot request an
arbitrary file path or crop. A valid `sourceFrameIndex: 0` is preserved rather
than treated as a missing value. Roster completeness is derived from structured
`rosterQuality` flags; human-readable provider warnings never change identity
eligibility.

## Manual editing

The editor sends `canonical_person_id` separately from the optional
`source_track_id`. Consequently all actions also work for a video-only person:

- **Confirm** adds positive identity, role and label evidence; roster identity
  is accepted only through the dedicated canonical Bind endpoint;
- **Exclude** can affect one observation or the complete canonical identity;
- **Merge** targets either a canonical person or a scene-local annotation;
- **Split identity here / range** moves observations in a half-open
  `[start, end)` interval to a new canonical identity and persists a cannot-link
  barrier so a later merge pass cannot immediately reconnect both partitions;
- **Bind roster player / Unbind / Clear Unbind** uses a dedicated canonical
  endpoint. The
  server anchors the decision to an already-saved detector observation, checks
  team and duplicate bindings, stores a stable identity-scope correction and
  queues the rebuild in the same fingerprint-guarded CAS write. `Unbind`
  persists an explicit negative decision; `Clear Unbind` is the only operation
  that removes it and allows evidence to suggest a roster player again;
- deleting a generic Confirm/Exclude/Merge/Split correction remains its undo
  operation. Generic annotation deletion rejects dedicated roster corrections.

Split corrections snapshot the selected immutable observation ID together with
its frame, time and bbox. A detector reorder is accepted only when exactly one
same-frame candidate passes the strict geometry gate; a recycled detector ID,
missing target or ambiguous overlap fails closed instead of selecting a nearby
person. The editor previews the affected/remaining observation counts before
save. Save/delete and the queued rebuild use the same fingerprint-guarded CAS,
so deleting the correction is deterministic undo and stale work cannot publish.

Split state is partition-local rather than copied from the original identity.
Appearance features and ReID role votes are rebuilt from retained observations;
manual kind/label and confirmed `externalPlayerId` are derived only from the
positive annotations whose immutable anchor belongs to that partition. The new
split branch starts without inherited resolver evidence/conflicts, then records
the explicit manual-split evidence and is resolved again from its own data. A
confirmed roster binding therefore follows its anchor observation to exactly
one partition and never remains positive on both sides.

Exact annotation and bbox overlap are used before trajectory remapping for the
other correction actions. Missing or ambiguous remaps fail closed and retain the
last successful scene.

The roster UI never mutates `externalPlayerId` directly in scene JSON. OCR
roster matches are suggestions only; accepting one calls the durable binding
endpoint. This also works for an off-screen canonical person with no current 3D
projection and survives a detector/ReID/OCR outage. Unbind replaces the same
stable correction rather than creating contradictory history. If a split has
moved the binding to a new canonical person, the next bind/unbind locates the
existing correction by the immutable observation owner, rekeys its deterministic
ID and all published references, and leaves only one roster correction.
Clearing an Unbind uses a separate DELETE on that canonical person's roster
endpoint. It follows only the active merge lineage and the selected partition's
transitive split ancestors, validates hidden undo snapshots against their
source/target ownership, and removes matching snapshots. Unrelated or sibling
branches are preserved; related malformed metadata fails atomically. Therefore
ordered Undo Merge/Split cannot silently resurrect a cleared decision.

The generic frame-confirm editor deliberately has no roster selector, and its
API rejects a non-null external player ID. Team, uniqueness and ownership checks
therefore cannot be bypassed outside canonical Bind/Unbind/Clear. Older scenes may
still contain generic roster confirms; a dedicated bound or unbound correction
supersedes those values independently of frame/list order while retaining all
annotation IDs for audit and split localization. Conflicting dedicated decisions
or ambiguous owners fail closed.

An unbind is persisted as an observation-anchored tombstone so later rebuilds
can still remap the decision deterministically. It is not positive identity
evidence: it is excluded from positive annotation IDs, manual evidence and
manual-decision counts, and cannot by itself mark a person `resolved` or set
manual confidence/source. The pre-binding provisional identity semantics are
retained after the external player ID is cleared.
When a split changes the correction owner, the optimistic unbind baseline
(display name, status, confidence and source) is recomputed from positive
non-roster annotations in that partition rather than copied from the old owner.

Two identities carrying different confirmed `externalPlayerId` values cannot
be merged. The editor omits such targets, and save-time validation, raw-track
correction and published-scene correction independently fail closed if a stale
client or older scene still submits the operation.

Nested split corrections are topologically ordered by the identities they
produce and consume; orphan and cyclic lineage is rejected. A child split must
resolve inside its actual parent partition rather than falling back to an
unrelated raw track. Merge semantics are equally explicit: the terminal target
is the survivor, compatible unbound roster tombstones are stored in undo
metadata, and deleting the merge restores them only after validating that no
live dependent correction or bound roster decision would be orphaned.
An Unbind that existed before a split is consolidated back onto the recombined
source identity when the split is undone. A roster decision created or changed
after a merge records that merge as a dependency and blocks merge undo until
the user explicitly Unbinds and Clears it, because assigning that decision to
either pre-merge identity would be ambiguous.

When a manually bound roster number disagrees with reliable OCR, the binding
remains authoritative but the canonical person receives an explicit
`manual-roster-jersey-conflict`; roster suggestions are suppressed until
review. A bound external player missing or duplicated in the persisted match
snapshot is reported as `manual-roster-player-missing`. Changing the bound
match validates both published canonical people and durable bound corrections,
then queues a fingerprint-guarded rebuild. It never silently carries a player
into an incompatible roster.

## Rebuild and multi-angle behaviour

Base person detections are persisted before annotations and identity work. The
artifact key includes the exact sampled JPEG, detector checkpoint/provenance,
runtime, device, class/NMS/filter/feature policy and schema. A correction rebuild
therefore reapplies annotations, calibration, tracking, ReID and OCR without
rerunning unchanged base inference. Corrupt, tampered, fallback or partial
artifacts are safe misses and are never published as primary cache entries.

All identity/roster mutations are blocked while a reconstruction is queued or
processing. Accepted correction saves/deletes, roster Bind/Unbind/Clear and match
binding changes publish their updated inputs together with the next queued run
under compare-and-swap. The generic whole-scene PUT cannot alter reconstruction
inputs or runtime run fields and rejects stale clients. This prevents a late
save or worker from resurrecting an older correction graph.

Queued and interrupted processing jobs use a process-independent database lease.
The scene run ID, run revision, input fingerprint and full-document revision are
still the publication guards; the lease adds an expiring owner token and a
heartbeat that does not mutate the scene revision. Active leases cannot be
stolen, while processing work without an active lease is reclaimed; an old owner cannot
publish after takeover. Terminal jobs clear the lease, while an invalid recorded
input fingerprint fails closed and can be explicitly queued again from current
inputs instead of remaining stuck forever.

Aligned replay passes may add cross-view identity evidence only when the
alignment is independently accepted and identity is supported by the same
explicit external player or a reliable jersey number. Team, role and jersey
conflicts block fusion; ambiguous candidates abstain. Foreign observations stay
namespaced in `multiAngleEvidence` and are never injected into the reference
camera's observation/trajectory stream. Manual temporal anchors override DTW
only after finite, in-range, monotonic validation.

## UI selection rule

Video → 3D and 3D → video selection first compare `canonicalPersonId`.
`matchedTrackId` is only a migration fallback for old scenes. A canonical person
without a render track remains selectable and opens the same identity inspector
with `Not projected in 3D` instead of the misleading `not linked` state.

## Runtime metrics

`videoAsset.reconstruction.diagnostics.identity` reports:

- local tracklet and canonical person counts;
- resolved/provisional/excluded counts;
- accepted, manual, review, rejected and ambiguous edge counts;
- rejection reasons;
- preserved observation coverage;
- ReID provider/model readiness;
- requested, usable, selected-independent and rejected crop counts, crop
  qualities, cache/provider counters, unique/duplicate pixel fingerprints and
  usable crop ratio;
- reliable/provisional/conflicting jersey counts and OCR cache/provider
  counters, including unique/duplicate pixel fingerprints;
- association score p10/p50 and accepted/review distributions;
- manual decision and conflict counts;
- manual roster/OCR mismatch and missing-player conflict counts;
- applied manual split ranges and their source/new canonical identities.

For an unlabelled clip, IDF1, true ID-switch and duplicate-assignment counts are
explicitly unavailable rather than inferred from trajectory smoothness. When
labelled observations are supplied, the quality layer computes IDF1,
identity precision/recall, switches and fragments and enables an identity gate.
Frame-only labels still support `duplicateAssignmentFrameCount`, ordering,
switch and fragment counts, but `duplicateOverlapSeconds` remains null unless
rows carry `sceneTime`/`time` or the validation payload declares an explicit
`identityAssignmentFrameRate`; a frame index is never reported as a duration.
With explicit FPS, duration is exactly duplicate-frame count divided by FPS;
label sampling gaps are not treated as continuous overlap.
`duplicateOverlapTimebase` explicitly reports `seconds`,
`frame-index+explicit-fps`, `frame-index-without-fps` or `mixed`, and the
normalized `identityAssignmentFrameRate` is included in the result.
HOTA/GS-HOTA remains null until the official evaluator is run; it is never
approximated by a home-grown score.

## Operational rollout and known next steps

The recommended real-player identification rollout is evidence fusion rather
than a single name-recognition model:

1. **Make real inference observable:** run the pinned ReID and jersey OCR
   workers, require both `/health/ready` checks, rebuild a representative scene,
   and retain provider/checkpoint provenance plus crop rejection reasons.
2. **Constrain the search space:** bind the exact match roster and, where the
   provider supplies it, lineup, goalkeeper, substitution and event-time data.
   Team, role and active-time constraints remove impossible roster candidates;
   they do not prove a name.
   The current free TheSportsDB response may contain only five lineup entries;
   such a snapshot is explicitly incomplete and must not be used for automatic
   naming. The editor now accepts a reviewed strict JSON roster, persists its
   provenance, and marks incomplete imports manual-only. Legacy bindings that
   contain only an event ID can be refreshed or replaced through the same UI;
   both the UI and reconstruction read the resulting offline v2 snapshot.
3. **Recover anonymous continuity first:** use short tracklets, PRTReID,
   trajectory physics and manual split/merge barriers to keep one
   `canonicalPersonId` through camera motion, exits and re-entry. ReID is
   appearance evidence for sameness, not a global database of player names.
4. **Attach names conservatively:** repeated independent jersey reads plus the
   bound team roster create a ranked candidate. Only an explicit user Bind
   promotes a candidate to `externalPlayerId`; the full persisted roster also
   remains manually selectable when the resolver correctly abstains. A sufficiently large close-up
   face/head embedding can later become another independent review signal, but
   must abstain on broadcast-scale or occluded faces and requires a separately
   licensed roster gallery.
5. **Exploit aligned replays:** propagate only independently supported jersey,
   roster or reviewed identity evidence between accepted multi-angle passes.
   Cross-view pitch proximity alone is never sufficient.
6. **Build and version a labelled product crop/tracklet set:** benchmark
   MMOCR/EasyOCR against a PARSeq-style tracklet provider, and benchmark PRT
   thresholds plus the online tracker on crossings, far-side players,
   goalkeeper/referee kits, blur and long exits. Compare StrongSORT/TrackLab
   under the same ground truth.
7. **Gate releases on identity metrics:** IDF1, identity precision/recall,
   switches, fragments, duplicate assignments, jersey exact-read/abstention,
   roster top-1/top-k and confidence calibration. Run official HOTA/GS-HOTA for
   reportable benchmark results; runtime smoothness is not ground truth.
8. **Scale only after accuracy is measured:** add GPU production images,
   cross-job batching and a shared persistent cache. For multi-host scale,
   evolve the compact DB job table into a brokered multi-host queue while
   retaining input/revision CAS and owner fencing.

## Verification

```bash
docker compose up --build -d identity-worker
curl --fail http://127.0.0.1:8091/health/live
curl --fail http://127.0.0.1:8091/health/ready
.venv/bin/pytest -q apps/api/tests/test_identity_resolver.py
.venv/bin/pytest -q apps/api/tests/test_canonical_identity_integration.py
.venv/bin/pytest -q apps/api/tests/test_canonical_roster_binding.py
.venv/bin/pytest -q services/identity-worker/tests
.venv/bin/pytest -q services/jersey-ocr-worker/tests
.venv/bin/pytest -q services/model-validation/tests
npm run test
npm run typecheck
npm run build
```
