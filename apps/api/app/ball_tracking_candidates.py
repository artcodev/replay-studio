"""Validation and normalization of per-frame ball detector candidates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

from .ball_tracking_contract import (
    BallFrameInput,
    BallTrackingConfig,
    MotionCoordinateSelector,
)


@dataclass(frozen=True)
class BallCandidate:
    frame_index: int
    time: float
    rank: int
    candidate_id: str
    image_x: float
    image_y: float
    confidence: float
    pitch_x: float | None
    pitch_z: float | None
    stabilised_x: float | None
    stabilised_y: float | None
    stabilised_source: str | None
    provenance: dict[str, Any]
    source: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class NormalizedBallFrame:
    time: float
    candidates: tuple[BallCandidate, ...]


@dataclass(frozen=True)
class NormalizedBallFrames:
    frames: tuple[NormalizedBallFrame, ...]
    invalid_candidate_count: int
    dropped_by_top_k_count: int


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _first_number(source: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_number(source.get(key))
        if value is not None:
            return value
    return None


def _nested_point(source: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = source.get(key)
    if not isinstance(value, Mapping):
        return None
    x = _first_number(value, "x", "X")
    y = _first_number(value, "y", "Y")
    return (x, y) if x is not None and y is not None else None


def _pitch_point(source: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _first_number(source, "pitchX", "pitch_x")
    z = _first_number(source, "pitchZ", "pitch_z")
    nested = source.get("pitch")
    if (x is None or z is None) and isinstance(nested, Mapping):
        x = _first_number(nested, "x", "X")
        z = _first_number(nested, "z", "Z", "y", "Y")
    return (x, z) if x is not None and z is not None else None


def _automatic_stabilised_point(
    source: Mapping[str, Any],
) -> tuple[float, float, str] | None:
    key_pairs = (
        ("stabilizedX", "stabilizedY", "camera-stabilized"),
        ("stabilisedX", "stabilisedY", "camera-stabilised"),
        ("stableX", "stableY", "camera-stabilized"),
        ("cameraCompensatedX", "cameraCompensatedY", "camera-compensated"),
    )
    for x_key, y_key, label in key_pairs:
        x = finite_number(source.get(x_key))
        y = finite_number(source.get(y_key))
        if x is not None and y is not None:
            return x, y, label
    for key, label in (
        ("stabilized", "camera-stabilized"),
        ("stabilised", "camera-stabilised"),
        ("cameraCompensated", "camera-compensated"),
        ("motionCoordinates", "camera-motion-hook"),
    ):
        point = _nested_point(source, key)
        if point is not None:
            return point[0], point[1], label
    return None


def _candidate_provenance(
    source: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    supplied = source.get("provenance")
    if isinstance(supplied, Mapping):
        provenance: dict[str, Any] = deepcopy(dict(supplied))
    elif supplied is not None:
        provenance = {"value": deepcopy(supplied)}
    else:
        provenance = {}
    for key in (
        "source",
        "backend",
        "detector",
        "model",
        "modelVersion",
        "inferenceId",
        "passId",
        "tile",
        "metadata",
    ):
        if key in source and key not in provenance:
            provenance[key] = deepcopy(source[key])
    provenance.setdefault("candidateId", candidate_id)
    return provenance


def _frame_contents(
    frame: BallFrameInput, fallback_index: int
) -> tuple[Sequence[Any], float]:
    if isinstance(frame, Mapping):
        time = _first_number(frame, "t", "time", "timestamp", "sceneTime")
        candidates = frame.get("candidates")
        if candidates is None:
            candidates = frame.get("detections")
        if candidates is None:
            candidates = frame.get("balls")
    elif isinstance(frame, tuple) and len(frame) == 2:
        candidates, time = frame
        time = finite_number(time)
    else:
        raise ValueError(
            f"Ball frame {fallback_index} must be a mapping or (candidates, time) tuple"
        )
    if time is None:
        raise ValueError(f"Ball frame {fallback_index} has no finite timestamp")
    if candidates is None:
        candidates = []
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError(f"Ball frame {fallback_index} candidates must be a sequence")
    return candidates, time


def normalize_ball_frames(
    frames: Sequence[BallFrameInput],
    config: BallTrackingConfig,
    coordinate_selector: MotionCoordinateSelector | None,
) -> NormalizedBallFrames:
    """Validate timestamps/candidates and retain the top-K hypotheses per frame."""

    result: list[NormalizedBallFrame] = []
    invalid_candidates = 0
    dropped_by_top_k = 0
    previous_time: float | None = None
    for frame_index, frame in enumerate(frames):
        raw_candidates, time = _frame_contents(frame, frame_index)
        if previous_time is not None and time <= previous_time:
            raise ValueError("Ball-frame timestamps must be strictly increasing")
        previous_time = time
        parsed: list[tuple[float, int, Mapping[str, Any]]] = []
        for source_index, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                invalid_candidates += 1
                continue
            x = finite_number(raw.get("x"))
            y = finite_number(raw.get("y"))
            confidence = _first_number(raw, "confidence", "score", "probability")
            if x is None or y is None or confidence is None:
                invalid_candidates += 1
                continue
            parsed.append((min(1.0, max(0.0, confidence)), source_index, raw))
        parsed.sort(key=lambda item: (-item[0], item[1]))
        dropped_by_top_k += max(0, len(parsed) - config.top_k_per_frame)
        normalized: list[BallCandidate] = []
        for rank, (confidence, source_index, raw) in enumerate(
            parsed[: config.top_k_per_frame], start=1
        ):
            identifier = raw.get("candidateId", raw.get("id", raw.get("detectionId")))
            candidate_id = (
                str(identifier).strip()
                if identifier is not None and str(identifier).strip()
                else f"ball-f{frame_index}-c{source_index}"
            )
            pitch = _pitch_point(raw)
            stabilised: tuple[float, float, str] | None = None
            if coordinate_selector is not None:
                selected = coordinate_selector(raw, frame_index)
                if selected is not None:
                    if len(selected) != 3:
                        raise ValueError("coordinate_selector must return (x, y, source)")
                    stable_x = finite_number(selected[0])
                    stable_y = finite_number(selected[1])
                    if (
                        stable_x is None
                        or stable_y is None
                        or not str(selected[2]).strip()
                    ):
                        raise ValueError(
                            "coordinate_selector returned invalid coordinates"
                        )
                    stabilised = stable_x, stable_y, str(selected[2]).strip()
            if stabilised is None:
                stabilised = _automatic_stabilised_point(raw)
            normalized.append(
                BallCandidate(
                    frame_index=frame_index,
                    time=time,
                    rank=rank,
                    candidate_id=candidate_id,
                    image_x=float(raw["x"]),
                    image_y=float(raw["y"]),
                    confidence=confidence,
                    pitch_x=pitch[0] if pitch is not None else None,
                    pitch_z=pitch[1] if pitch is not None else None,
                    stabilised_x=stabilised[0] if stabilised is not None else None,
                    stabilised_y=stabilised[1] if stabilised is not None else None,
                    stabilised_source=stabilised[2] if stabilised is not None else None,
                    provenance=_candidate_provenance(raw, candidate_id),
                    source=deepcopy(dict(raw)),
                )
            )
        result.append(NormalizedBallFrame(time=time, candidates=tuple(normalized)))
    return NormalizedBallFrames(
        frames=tuple(result),
        invalid_candidate_count=invalid_candidates,
        dropped_by_top_k_count=dropped_by_top_k,
    )


__all__ = [
    "BallCandidate",
    "NormalizedBallFrame",
    "NormalizedBallFrames",
    "finite_number",
    "normalize_ball_frames",
]
