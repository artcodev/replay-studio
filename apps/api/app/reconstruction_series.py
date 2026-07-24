"""Bounded read model for dense reconstruction artifacts."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping

from .track_trajectory_corrections import (
    apply_track_trajectory_correction,
    track_trajectory_corrections,
)
from .artifact_store import ArtifactStore
from .reconstruction_artifact_hydration import load_dense_reconstruction_artifacts


MAX_SERIES_WINDOW_SECONDS = 30.0
MAX_SERIES_FRAME_WINDOW = 900


class ReconstructionSeriesWindowError(ValueError):
    pass


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _frame_index(value: Mapping[str, Any]) -> int | None:
    raw = value.get("frameIndex", value.get("sourceFrameIndex"))
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _in_window(
    value: Mapping[str, Any],
    *,
    start: float,
    end: float,
    frame_start: int | None,
    frame_end: int | None,
) -> bool:
    time = _number(value.get("sceneTime", value.get("t")))
    if time is not None and not (start <= time <= end):
        return False
    frame = _frame_index(value)
    if frame_start is not None and frame is not None and frame < frame_start:
        return False
    if frame_end is not None and frame is not None and frame > frame_end:
        return False
    return True


def reconstruction_series_window(
    scene: Mapping[str, Any],
    *,
    start: float,
    end: float,
    frame_start: int | None = None,
    frame_end: int | None = None,
    track_id: str | None = None,
    canonical_person_id: str | None = None,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    duration = float(scene.get("duration") or 0.0)
    if not isfinite(start) or not isfinite(end) or start < 0.0 or end < start:
        raise ReconstructionSeriesWindowError("The time window is invalid")
    if end - start > MAX_SERIES_WINDOW_SECONDS + 1e-9:
        raise ReconstructionSeriesWindowError(
            f"The time window may not exceed {MAX_SERIES_WINDOW_SECONDS:g} seconds"
        )
    if start > duration or end > duration + 1e-6:
        raise ReconstructionSeriesWindowError("The time window exceeds the scene duration")
    if (frame_start is None) != (frame_end is None):
        raise ReconstructionSeriesWindowError(
            "frame_start and frame_end must be supplied together"
        )
    if frame_start is not None and frame_end is not None:
        if frame_start < 0 or frame_end < frame_start:
            raise ReconstructionSeriesWindowError("The frame window is invalid")
        if frame_end - frame_start + 1 > MAX_SERIES_FRAME_WINDOW:
            raise ReconstructionSeriesWindowError(
                f"The frame window may not exceed {MAX_SERIES_FRAME_WINDOW} frames"
            )

    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    loaded = load_dense_reconstruction_artifacts(reconstruction, store=store)
    identity = loaded.get("identityTimeline") or {}
    # Durable manual trajectory corrections are authoritative over model
    # keyframes and are anchored to the canonical identity, so they survive a
    # rebuild that renumbers render tracks.
    corrections = track_trajectory_corrections(scene)
    canonical_by_track = {
        str(item.get("id")): str(item.get("canonicalPersonId") or "")
        for item in scene.get("payload", {}).get("tracks") or []
        if isinstance(item, Mapping)
    }
    tracks = []
    for track in identity.get("tracks") or []:
        if not isinstance(track, Mapping):
            continue
        identifier = str(track.get("id") or "")
        if track_id is not None and identifier != track_id:
            continue
        manual = corrections.get(canonical_by_track.get(identifier, ""), [])
        published_keyframes = (
            apply_track_trajectory_correction(track.get("keyframes") or [], manual)
            if manual
            else track.get("keyframes") or []
        )
        tracks.append(
            {
                "id": identifier,
                "keyframes": [
                    deepcopy(item)
                    for item in published_keyframes
                    if isinstance(item, Mapping)
                    and _in_window(
                        item,
                        start=start,
                        end=end,
                        frame_start=frame_start,
                        frame_end=frame_end,
                    )
                ],
                "observations": [
                    deepcopy(item)
                    for item in track.get("observations") or []
                    if isinstance(item, Mapping)
                    and _in_window(
                        item,
                        start=start,
                        end=end,
                        frame_start=frame_start,
                        frame_end=frame_end,
                    )
                ],
            }
        )

    people = []
    for person in identity.get("canonicalPeople") or []:
        if not isinstance(person, Mapping):
            continue
        identifier = str(person.get("canonicalPersonId") or "")
        if canonical_person_id is not None and identifier != canonical_person_id:
            continue
        people.append(
            {
                "canonicalPersonId": identifier,
                "observations": [
                    deepcopy(item)
                    for item in person.get("observations") or []
                    if isinstance(item, Mapping)
                    and _in_window(
                        item,
                        start=start,
                        end=end,
                        frame_start=frame_start,
                        frame_end=frame_end,
                    )
                ],
            }
        )

    ball_source = loaded.get("ballTrajectory") or {}
    ball = {
        key: [
            deepcopy(item)
            for item in ball_source.get(key) or []
            if isinstance(item, Mapping)
            and _in_window(
                item,
                start=start,
                end=end,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        ]
        for key in ("keyframes", "automaticKeyframes", "manualKeyframes")
    }
    calibration_source = loaded.get("calibrationFrames") or {}
    calibration = {
        "frameEvidence": [
            deepcopy(item)
            for item in calibration_source.get("frameEvidence") or []
            if isinstance(item, Mapping)
            and _in_window(
                item,
                start=start,
                end=end,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        ]
    }
    ball_detection = {
        "frames": [
            deepcopy(item)
            for item in (calibration_source.get("ballDetection") or {}).get("frames") or []
            if isinstance(item, Mapping)
            and _in_window(
                item,
                start=start,
                end=end,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        ]
    }
    return {
        "schemaVersion": 1,
        "sceneId": str(scene.get("id") or ""),
        "window": {
            "start": start,
            "end": end,
            "frameStart": frame_start,
            "frameEnd": frame_end,
        },
        "tracks": tracks,
        "canonicalPeople": people,
        "ball": ball,
        "calibration": calibration,
        "ballDetection": ball_detection,
    }


__all__ = [
    "MAX_SERIES_FRAME_WINDOW",
    "MAX_SERIES_WINDOW_SECONDS",
    "ReconstructionSeriesWindowError",
    "reconstruction_series_window",
]
