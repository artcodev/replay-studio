# Player actions: annotations now, 3D animation later

This document deliberately separates *describing what happened* from
*animating a 3D body*. Action annotations belong to the reconstruction editor;
character rigs and clips remain a downstream renderer concern.

Status (17 July 2026): the manual annotation slice of `TD-ACT-01` is
implemented. The editor persists player-centric action intervals and multiple
semantic keypoints, exposes them on the selected canonical player's timeline,
and, when that identity has a selected render track, derives a renderer-neutral
playback preview for the 3D viewport.

The following work is **not** implemented:

- automatic action recognition and suggestion review (`Accept` / `Reject`);
- dependency fingerprints and invalidation after ball, track, or identity
  changes;
- pose artifacts or a labelled action-model validation pipeline;
- UCS model/clip loading, `AnimationClip` selection, retargeting, blending,
  root-motion handling, IK, or pose synchronization.

`TD-ANIM-01` therefore remains separate and must not shape or block the action
annotation contract.

## Implemented scene contract

Compact action records are stored persistently in
`SceneDocument.payload.playerActions`. The authoritative actor key is
`canonicalPersonId`, never a detector label such as `person-8` or an optional
render-track ID.

```json
{
  "id": "action-f67671fd...",
  "canonicalPersonId": "canonical-61fc55b63529",
  "type": "pass",
  "startTime": 12.38,
  "endTime": 13.1,
  "keypoints": [
    { "kind": "wind-up", "time": 12.46 },
    { "kind": "contact", "time": 12.74 },
    { "kind": "recovery", "time": 13.02 }
  ],
  "confidence": 1,
  "status": "confirmed",
  "source": "manual",
  "createdAt": "2026-07-17T12:00:00+00:00",
  "updatedAt": "2026-07-17T12:01:00+00:00"
}
```

Manual writes are server-owned for provenance: the API returns
`confidence: 1`, `status: "confirmed"`, and `source: "manual"`. Times are
finite scene-relative seconds, normalized to milliseconds, with
`startTime < endTime`. Every keypoint must lie inside its interval. Exact
duplicate `(time, kind)` markers collapse deterministically; one action may
contain at most 24 compact keypoints.

The supported semantic keypoints are `wind-up`, `contact`, `release`, `apex`,
`impact`, and `recovery`. These markers describe meaningful phases; they do
not contain skeleton poses or imply a particular animation clip.

Dense pose sequences, when added later, must remain immutable analysis
artifacts addressed by URI/content hash. They must not be embedded in scene
JSON.

## Manual editor workflow

1. Select a canonical player on the video or in the 3D scene. The player
   action timeline appears above the shared transport.
2. Move the shared playhead to the relevant moment and choose **Add action**.
   A confirmed manual `pass` interval is created with an action-appropriate
   default duration and phase marker.
3. Select the interval to seek to its start, then choose any supported action
   type and edit its start/end times.
4. Add, change, move, or remove as many semantic phase markers as needed. A
   marker click seeks the video and 3D scene to that exact time.
5. Each add, edit, or delete is persisted immediately through the scene action
   API; no separate **Save scene** step is required. While reconstruction is
   queued or processing, action mutations are locked.

The timeline is scoped to the selected `canonicalPersonId`. Overlapping
intervals use separate visual lanes. Manual intervals are editable; an
imported `source: "automatic"` interval is currently displayed as review-only
because its accept/reject API is still technical debt.

## Action taxonomy

The implemented taxonomy contains 21 types:

- locomotion/body state: `idle`, `walk`, `run`, `sprint`, `turn`, `jump`,
  `fall`, `get-up`;
- ball actions: `first-touch`, `drive`, `pass`, `cross`, `shot`, `header`,
  `throw-in`, `clearance`;
- defensive actions: `tackle`, `slide-tackle`, `block`, `interception`;
- skill/deception: `feint`.

Each type has editor metadata for label, category color, default interval
duration, and default significant phase. This metadata is an authoring aid,
not a recognition or animation-quality claim.

## API contract

Create or update one manual action:

```http
POST /api/scenes/{sceneId}/player-actions
Content-Type: application/json

{
  "id": "action-client-generated-id",
  "canonicalPersonId": "canonical-61fc55b63529",
  "type": "shot",
  "startTime": 8.125,
  "endTime": 8.9,
  "keypoints": [
    { "kind": "wind-up", "time": 8.2 },
    { "kind": "contact", "time": 8.54 }
  ]
}
```

Delete one manual action:

```http
DELETE /api/scenes/{sceneId}/player-actions/{actionId}
```

Both endpoints return the updated `SceneDocument`. They validate the canonical
person, scene bounds, taxonomy, keypoint kinds, and interval containment. They
return `409` while reconstruction is busy, `404` for a missing scene/action or
canonical person, and `422` for an invalid mutation. The manual endpoints
preserve unrelated records and refuse to overwrite or delete automatic
suggestions.

## Renderer-neutral 3D preparation

At the shared playhead the frontend deterministically selects one active,
non-rejected interval for the selected canonical player. A manual confirmed
interval takes priority over automatic hypotheses. When the selection also has
a render track, the derived playback state passed to `ThreeViewport` contains:

- the semantic action record;
- normalized interval phase `0..1`;
- elapsed time and interval duration;
- the nearest significant keypoint, its normalized phase, and temporal offset.

`ThreeViewport` displays this data as an action/phase/keypoint preview only.
It neither loads a character asset nor selects or advances an animation clip.
This renderer-neutral seam is intended to let the future animation layer map
an accepted action to an `AnimationClip` and scrub an `AnimationAction` from
the absolute shared playhead without changing stored editor data.

## Remaining TD-ACT-01 work: recognition and review

An eventual recognizer should combine player crops/pose with track velocity,
neighbours, pitch orientation, and ball motion. Pose alone cannot reliably
separate a pass from a shot or a tackle from a fall. Its output should create
`source: "automatic"`, `status: "suggested"` records with calibrated
confidence and compact evidence. Unknown/ambiguous must remain a valid result.

The missing review layer needs explicit `Accept`, `Reject`, `Change type`, and
`Adjust as manual` operations. Automatic suggestions must never overwrite
manual records. A versioned input fingerprint should cover only the evidence
used by an action hypothesis; changing the manual ball trajectory or relevant
identity/track evidence should invalidate dependent automatic suggestions
without rerunning calibration, detection, OCR, or identity resolution.

Recognition acceptance requires a labelled product set and per-class temporal
metrics, actor-attribution accuracy, contact-frame error, abstention rate, and
confidence calibration. None of those accuracy claims are implied by the
manual editor.

### Planned implementation order

1. **Ball-centric primitives:** infer touch, possession, possession transfer,
   pass, shot, cross, clearance and interception from the reviewed player and
   ball trajectories. These hypotheses must preserve the contributing ball,
   actor and calibration evidence instead of publishing only a class label.
2. **Body-centric primitives:** add pose-derived run/sprint, turn, jump, tackle,
   block, fall, get-up and wind-up hypotheses. Pose is supporting evidence; it
   cannot independently decide ball semantics such as pass versus shot.
3. **Review workflow:** show automatic records as `suggested` and provide
   explicit Accept, Reject, Change type and Convert to manual operations. A
   suggestion may never overwrite a confirmed manual interval or keypoint.
4. **Selective invalidation:** fingerprint the exact player identity, track,
   pose, ball and calibration evidence used by each suggestion. Recompute only
   dependent hypotheses after a correction.
5. **Measured acceptance:** report per-class precision/recall/F1, temporal IoU,
   actor-attribution accuracy, contact/onset error, abstention and confidence
   calibration on a versioned labelled product set before enabling suggestions
   by default.

## TD-ANIM-01 — deferred UCS animation synchronization

The licensed models and animation catalogue in `/Users/art/code/art-lab/ucs`
are the intended future renderer source. The future layer should:

1. load and clone the UCS skinned player rig;
2. map a confirmed action interval to a compatible clip and mirrored side;
3. add reviewed sidecar metadata for wind-up/contact/apex/recovery frames,
   active foot/head, ball height, direction, speed range, and loopability;
4. keep calibrated actor X/Z and heading authoritative; UCS root motion must
   not independently move a player across the pitch;
5. scrub poses from the shared absolute playhead instead of accumulating
   renderer delta time;
6. add transition blending, foot locking, joint limits, and optional IK for
   confirmed ball contacts;
7. measure foot sliding, ball-contact residual, transition jerk, and fallback
   clip coverage.

Animation failure or absence must never change an accepted action, canonical
identity, ball path, or reconstruction geometry.

## Completion boundaries

The shipped manual slice is complete when manual intervals and semantic
keypoints can be created, edited, deleted, persisted, selected, and previewed
against the shared playhead. That boundary is now implemented.

The full action-annotation debt is not complete until automatic hypotheses can
be generated, reviewed, and selectively invalidated with measured quality.
The UCS debt is complete only when confirmed annotations produce deterministic,
seek-safe 3D poses with validated contact and blending metrics; playing an
arbitrary clip is not sufficient.
