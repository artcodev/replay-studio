from __future__ import annotations

"""Durable manual corrections of a player's pitch trajectory (TD-TRACK-01).

Dragging a player used to mutate the in-memory scene only: the client
stripped dense track keyframes before writing, so the edit was silently
discarded while the UI reported "Saved". This module makes the correction a
first-class, durable one — mirroring the manual ball trajectory.

Corrections are anchored to ``canonicalPersonId``, the identity that
``assign_persistent_canonical_person_ids`` keeps stable across rebuilds, not
to the per-run render track id. A track without a canonical identity cannot
carry a durable correction and is refused instead of accepting an edit that
the next rebuild would silently drop.
"""

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


CORRECTIONS_KEY = "trackTrajectoryCorrections"


def _canonical_person_for_track(scene: Mapping[str, Any], track_id: str) -> str:
    tracks = scene.get("payload", {}).get("tracks")
    if not isinstance(tracks, list):
        raise ReconstructionError("Scene has no tracks")
    track = next(
        (
            item
            for item in tracks
            if isinstance(item, Mapping) and str(item.get("id")) == str(track_id)
        ),
        None,
    )
    if track is None:
        raise ReconstructionError(f"Unknown track: {track_id}")
    canonical_person_id = str(track.get("canonicalPersonId") or "").strip()
    if not canonical_person_id:
        raise ReconstructionError(
            f"Track {track_id} has no canonical identity yet, so a trajectory "
            "correction could not survive a rebuild. Confirm the player's "
            "identity first."
        )
    return canonical_person_id


def _normalized_keyframes(
    keyframes: Sequence[Mapping[str, Any]],
    duration: float,
) -> list[dict]:
    normalized: list[dict] = []
    for keyframe in keyframes:
        try:
            time = float(keyframe["t"])
            x = float(keyframe["x"])
            z = float(keyframe["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionError(
                "Every trajectory keyframe requires finite t, x and z"
            ) from exc
        if not all(isfinite(value) for value in (time, x, z)):
            raise ReconstructionError(
                "Every trajectory keyframe requires finite t, x and z"
            )
        if time < 0.0 or (duration > 0.0 and time > duration):
            raise ReconstructionError(
                f"Trajectory keyframe at {time}s is outside the scene"
            )
        normalized.append(
            {"t": round(time, 3), "x": round(x, 2), "z": round(z, 2)}
        )
    normalized.sort(key=lambda item: item["t"])
    deduplicated: list[dict] = []
    for keyframe in normalized:
        if deduplicated and deduplicated[-1]["t"] == keyframe["t"]:
            deduplicated[-1] = keyframe
            continue
        deduplicated.append(keyframe)
    return deduplicated


def track_trajectory_corrections(scene: Mapping[str, Any]) -> dict[str, list[dict]]:
    """Manual pitch keyframes by canonical person, safe against bad payloads."""

    stored = scene.get("payload", {}).get(CORRECTIONS_KEY)
    if not isinstance(stored, list):
        return {}
    corrections: dict[str, list[dict]] = {}
    for entry in stored:
        if not isinstance(entry, Mapping):
            continue
        canonical_person_id = str(entry.get("canonicalPersonId") or "").strip()
        keyframes = entry.get("keyframes")
        if not canonical_person_id or not isinstance(keyframes, list):
            continue
        corrections[canonical_person_id] = [
            deepcopy(item) for item in keyframes if isinstance(item, Mapping)
        ]
    return corrections


def apply_track_trajectory_correction(
    keyframes: Iterable[Mapping[str, Any]],
    manual: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Overlay authoritative manual points onto one track's keyframes.

    A manual point replaces any automatic keyframe at the same instant and is
    published as observed, manual-sourced evidence, so review never mistakes
    a user-authored position for a model observation.
    """

    merged = {
        round(float(item["t"]), 3): deepcopy(item)
        for item in keyframes
        if isinstance(item, Mapping) and item.get("t") is not None
    }
    for point in manual:
        time = round(float(point["t"]), 3)
        merged[time] = {
            **merged.get(time, {}),
            "t": time,
            "x": float(point["x"]),
            "z": float(point["z"]),
            "confidence": 1.0,
            "observed": True,
            "presenceState": "observed",
            "positionSource": "manual",
            "projectionSource": "manual",
        }
    return [merged[key] for key in sorted(merged)]


def set_track_trajectory(
    scene: dict,
    track_id: str,
    keyframes: Sequence[Mapping[str, Any]],
) -> dict:
    """Publish a durable manual trajectory correction for one player."""

    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before editing a trajectory"
        )
    canonical_person_id = _canonical_person_for_track(scene, track_id)
    normalized = _normalized_keyframes(
        keyframes, float(scene.get("duration") or 0.0)
    )
    payload = scene.setdefault("payload", {})
    stored = [
        entry
        for entry in (payload.get(CORRECTIONS_KEY) or [])
        if isinstance(entry, Mapping)
        and str(entry.get("canonicalPersonId") or "") != canonical_person_id
    ]
    if normalized:
        stored.append(
            {
                "canonicalPersonId": canonical_person_id,
                "keyframes": normalized,
                "updatedAt": datetime.now(UTC).isoformat(),
            }
        )
    payload[CORRECTIONS_KEY] = stored
    return scenes.put(scene)


__all__ = (
    "CORRECTIONS_KEY",
    "apply_track_trajectory_correction",
    "set_track_trajectory",
    "track_trajectory_corrections",
)
