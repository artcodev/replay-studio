from __future__ import annotations

"""Application commands for the small user-authored scene document fields.

These are the last editor writes that used to travel as a whole scene
document through the generic ``PUT /scenes/{id}``. That path made every UI
edit lose a race against the reconstruction runner's own writes, because a
single document revision fenced fields with entirely independent writers.

Each command instead applies one domain onto the *currently stored* scene,
mirroring the ball-trajectory and player-action commands, so the revision
fence stays a genuine guard for programmatic writes instead of fighting the
editor.
"""

from typing import Mapping, Sequence

from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


def _editable_scene(scene: dict) -> dict:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before editing the scene"
        )
    return scene


def set_scene_title(scene: dict, title: str) -> dict:
    """Rename one scene without republishing its whole document."""

    cleaned = str(title).strip()
    if not cleaned:
        raise ReconstructionError("Scene title cannot be empty")
    _editable_scene(scene)["title"] = cleaned
    return scenes.put(scene)


def set_scene_event_bindings(
    scene: dict,
    bindings: Sequence[Mapping[str, object]],
) -> dict:
    """Replace the match event markers pinned to this scene's timeline."""

    editable = _editable_scene(scene)
    duration = float(editable.get("duration") or 0.0)
    normalized: list[dict] = []
    for binding in bindings:
        scene_time = float(binding["sceneTime"])
        if not 0.0 <= scene_time <= max(duration, scene_time):
            raise ReconstructionError(
                f"Event marker at {scene_time}s is outside the scene"
            )
        normalized.append(
            {
                "sceneTime": round(scene_time, 3),
                "externalEventId": str(binding["externalEventId"]),
                "label": str(binding["label"]),
                "type": str(binding["type"]),
            }
        )
    editable.setdefault("payload", {})["eventBindings"] = normalized
    return scenes.put(scene)


def set_track_metadata(
    scene: dict,
    track_id: str,
    *,
    label: str | None = None,
    number: int | None = None,
) -> dict:
    """Rename or renumber one published track."""

    editable = _editable_scene(scene)
    tracks = editable.get("payload", {}).get("tracks")
    if not isinstance(tracks, list):
        raise ReconstructionError("Scene has no tracks")
    track = next(
        (item for item in tracks if str(item.get("id")) == str(track_id)),
        None,
    )
    if track is None:
        # A rebuilt reconstruction may have retired this track: fail closed
        # instead of silently dropping the rename.
        raise ReconstructionError(f"Unknown track: {track_id}")
    if label is not None:
        cleaned = str(label).strip()
        if not cleaned:
            raise ReconstructionError("Track label cannot be empty")
        track["label"] = cleaned
    if number is not None:
        track["number"] = int(number)
    return scenes.put(scene)


__all__ = (
    "set_scene_event_bindings",
    "set_scene_title",
    "set_track_metadata",
)
