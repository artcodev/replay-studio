from __future__ import annotations

"""Normalize reconstruction QA evidence without applying metrics or policy."""

from math import isfinite
from statistics import median
from typing import Any, Iterable, Sequence


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def bounded_ratio(value: Any) -> float | None:
    result = finite_number(value)
    if result is None:
        return None
    return min(1.0, max(0.0, result))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(value for value in values if isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, quantile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def reconstruction_document(scene: dict[str, Any]) -> dict[str, Any]:
    return (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        or {}
    )


def frame_evidence(
    scene: dict[str, Any],
    supplied: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if supplied is not None:
        return [item for item in supplied if isinstance(item, dict)]
    calibration = reconstruction_document(scene).get("calibration") or {}
    return [
        item
        for item in calibration.get("frameEvidence") or []
        if isinstance(item, dict)
    ]


def frame_time(item: dict[str, Any]) -> float | None:
    for key in ("sceneTime", "time", "t"):
        value = finite_number(item.get(key))
        if value is not None:
            return value
    return None


def is_accepted_frame(item: dict[str, Any]) -> bool:
    return str(item.get("status") or "").lower() in {"accepted", "ready", "valid"}


def projection_source(item: dict[str, Any]) -> str | None:
    projection = item.get("projection") or {}
    source = projection.get("source") if isinstance(projection, dict) else None
    source = source or item.get("projectionSource")
    if source is None:
        return None
    return str(source).strip().lower()


def is_fallback_projection(source: str | None) -> bool:
    return bool(
        source
        and source
        in {
            "none",
            "fallback",
            "screen",
            "screen-relative",
            "screen-approximate",
            "screen-projected",
            "approximate",
            "representative-approximate",
        }
    )


def sample_cadence(times: Sequence[float]) -> float | None:
    ordered = sorted(set(times))
    deltas = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    return median(deltas) if deltas else None


def scene_keyframes(
    scene: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    payload = scene.get("payload") or {}
    result: list[tuple[str, str, dict[str, Any]]] = []
    for track in payload.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("id") or "unknown")
        for keyframe in track.get("keyframes") or []:
            if isinstance(keyframe, dict):
                result.append(("person", track_id, keyframe))
    for keyframe in (payload.get("ball") or {}).get("keyframes") or []:
        if isinstance(keyframe, dict):
            result.append(("ball", "ball", keyframe))
    return result
