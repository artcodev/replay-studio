# Replay Studio UI information architecture

## Problem

The former stage toolbar mixed camera navigation, view visibility, CV model
selection, expensive reconstruction commands, calibration QA, and player
editing in one horizontally scrolling row. At a 1280 px viewport several
controls extended underneath the Inspector, while duplicated commands such as
`Edit position` existed both in the toolbar and next to the selected player's
coordinates.

The redesign follows one rule: controls are grouped by the user's intention,
not by the internal subsystem that implements them.

## Considered variants

### A. Compact contextual toolbar — implemented

```text
[Camera: Broadcast] [Layout: Video + 3D]   [Analyze frame] [Reconstruction ▾] [Eye ▾]
```

- Camera and workspace layout always expose their current value.
- `Analyze frame` remains a direct action because it starts the main manual
  correction workflow.
- Model selection, reconstruct, model comparison, frame calibration and QA are
  grouped under `Reconstruction`.
- The eye disclosure contains independent 3D layers and render quality.
- At constrained desktop widths, disclosure text collapses to accessible icons
  while native camera/layout values remain visible.

This has the lowest interaction cost for the present editor and does not add a
second toolbar over a split video/3D viewport.

### B. Two-level video/3D toolbar

One row controls workspace and analysis; a second local row inside the 3D pane
controls camera, visibility, and quality. This scales well when the renderer
gains environment, animation, formation, measurement, or presentation tools,
but it currently consumes valuable vertical space and duplicates context.

Use this variant only after the 3D-specific command count grows beyond the eye
menu and one camera selector.

### C. Command palette / professional workspace

All commands are searchable and panels are dockable. This is appropriate for a
larger desktop production tool, but it is excessive for the current enthusiast
MVP and hides common commands behind keyboard or palette knowledge.

## Implemented ownership

| Surface | Responsibility |
|---|---|
| Camera selector | Broadcast, orbit, tactical, and goal-line presets |
| Layout selector | Video only, split video + 3D, or 3D only |
| Analyze frame | Detect/select/draw a person on the current video frame |
| Reconstruction menu | Detector model, full rebuild, model comparison, calibration and QA |
| View menu | Players, labels, ball, ball path, analysis markers, render quality |
| Inspector position card | The only entry point for manual pitch-position editing |
| Tracked objects search | Find by display name, shirt number, track ID, team or external binding |

## Render profiles

`Basic` is the safe default: DPR is capped at 1.25, dynamic shadows are off and
lighting is intentionally simple. Both profiles use four non-shadow-casting
stadium floodlights; `Enhanced` increases their output, raises DPR to 2, enables
a 2048 px `PCFSoftShadowMap`, shadow bias/normal bias and a restrained fill light. The
renderer switches profiles without replacing the WebGL canvas. Returning to
Basic releases shadow GPU targets.

Visibility layers are independent. In particular, player labels can remain
visible when player meshes are hidden; hidden meshes do not intercept picking.
Preferences persist locally between scenes and page reloads.

## Removed rudiments

- ambiguous `Video / 3D` boolean;
- top-level `Trajectory`, `Labels`, model, compare, calibration and QA buttons;
- duplicate top-level `Edit position`;
- always-on `Draft` badge;
- decorative scene-ID watermark and pseudo-live green dot;
- duplicate project import button in the sidebar.

## Accessibility and responsive contract

- Disclosure triggers expose `aria-expanded`, `aria-controls`, an accessible
  name and a hover title when their visible text collapses.
- View settings use native checkboxes/radios; Escape closes the panel and
  restores focus, click outside closes it, and options have 44 px targets.
- Camera/layout use native selects and never hide an option by CSS position.
- The toolbar does not horizontally scroll or overlap the Inspector.
- Below 1080 px, split view becomes a vertical video/3D stack instead of silently
  deleting the video pane.

## Sensible next iteration

If the editor grows further, split the Inspector into task-oriented `Object`,
`Frame`, `Quality`, and `Events` tabs, and group tracked objects by team/role.
The next visual-quality step should be real player assets, pitch texture and
contact detail; additional renderer toggles alone will not create the desired
wow effect.
