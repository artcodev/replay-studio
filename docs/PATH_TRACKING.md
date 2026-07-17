# Path Tracking layer

`Path Tracking` is a read-only presentation layer for the currently selected
render track or match ball. It does not alter calibration, tracking, identity,
ball keypoints, or the future player-action annotations.

## Current contract

- The layer is enabled independently in **3D view → Path tracking** and remains
  enabled while the user selects another player or the ball.
- Selection first resolves through the canonical identity system, then uses its
  current render track. Detector labels and current-frame box IDs are never
  treated as durable actor keys. An identity without a render track gets an
  explicit empty state.
- The whole highlight path is drawn. A current-time marker keeps the complete
  route readable while the shared playhead moves.
- Solid segments are backed by observed tracking samples. Dashed, lower-opacity
  segments are reconstructed/inferred presence and must never be presented as
  measured positions.
- A timestamped invalid sample is a path barrier. A player edge above 14 m/s or
  a ball edge above 60 m/s is treated as a calibration/identity teleport and is
  not joined. An observed-to-observed gap longer than one second is displayed
  as inferred; position uncertainty above three metres is also downgraded.
- The current-time marker is hidden inside a barrier instead of interpolating a
  false position. Selecting another actor rebuilds both surfaces immediately,
  so the previous actor's route cannot remain stale.

## Video geometry

Historical screen-space bounding boxes from different frames must not be joined:
a broadcast pan or zoom would turn camera motion into false player motion.
Instead, the pitch-space route is projected through the current frame's
QA-accepted image-to-pitch calibration. Between adjacent accepted calibration
samples, consistently scaled image-to-pitch homographies are blended and
revalidated before a single inversion. This mirrors the server's bounded dense
projection algorithm; interpolating already-projected pixels is forbidden.
A rejected, missing, cut, unreliable camera-motion edge, oversized bracket, or
unsafe matrix is a hard barrier. A bounded nearest sample (at most 250 ms) and
an interpolated camera are labelled in the video legend together with available
uncertainty.

If the entire route is outside the current field of view, the legend says so
instead of promising a hidden SVG. Multi-pass video projection is currently
limited to the reference camera because each secondary angle needs its own
accepted calibration. The ball overlay is explicitly a ground-plane projection:
its reconstructed height remains visible only in 3D until full camera extrinsics
are available.

Zoom and pan transform the video and its SVG path overlay together. The overlay
is non-interactive, so frame selection and manual annotation keep their existing
pointer and keyboard behaviour.

## Future action markers

The layer is the intended visualization surface for reviewed `TD-ACT-01`
intervals and keypoints such as pass contact, shot, tackle, block, or jump. Those
records remain keyed by `canonicalPersonId` and scene time as specified in
[`PLAYER_ACTIONS.md`](PLAYER_ACTIONS.md). Adding action markers must not couple
this layer to the deferred UCS animation synchronization (`TD-ANIM-01`).

Suggested and confirmed actions must be visually distinct, low-confidence
hypotheses must be allowed to abstain, and an action marker may be shown only
when its actor/time maps to the selected path. Animation playback is outside
this layer's contract.
