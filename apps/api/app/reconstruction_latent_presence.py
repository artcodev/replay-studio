from __future__ import annotations

"""Materialize explicitly uncertain off-camera presence for a scene track."""

from math import cos, sin

import numpy as np


def _presence_keyframe(
    keyframe: dict,
    time: float,
    x: float,
    z: float,
    state: str,
    uncertainty: float,
) -> dict:
    return {
        **keyframe,
        "t": round(time, 3),
        "x": round(x, 2),
        "z": round(z, 2),
        # Presence confidence is deliberately separate from detector evidence.
        # The renderer keeps actors alive for the whole scene, while QA excludes
        # these inferred samples via ``observed=False``.
        "confidence": 0.18,
        "observed": False,
        "presenceState": state,
        "projectionSource": "presence-inferred",
        "calibrationFrameIndex": None,
        "positionUncertaintyMetres": round(uncertainty, 2),
        "projection": {
            "source": "presence-inferred",
            "calibrationFrameIndex": None,
            "uncertaintyMetres": round(uncertainty, 2),
        },
    }


def _bounded_pitch_position(x: float, z: float, pitch: dict) -> tuple[float, float]:
    margin = 0.25
    half_length = max(margin, float(pitch["length"]) / 2 - margin)
    half_width = max(margin, float(pitch["width"]) / 2 - margin)
    return (
        max(-half_length, min(half_length, x)),
        max(-half_width, min(half_width, z)),
    )


def _roaming_presence_position(
    anchor: dict,
    elapsed: float,
    seed: int,
    pitch: dict,
    *,
    reverse: bool = False,
) -> tuple[float, float]:
    """Return a deterministic, deliberately small latent-position movement."""

    direction = -1.0 if reverse else 1.0
    phase = (seed % 17) * 0.37
    amplitude = min(0.65, max(0.0, elapsed) * 0.10)
    x = float(anchor["x"]) + direction * amplitude * sin(elapsed * 0.83 + phase)
    z = float(anchor["z"]) + amplitude * 0.72 * cos(elapsed * 0.67 + phase)
    return _bounded_pitch_position(x, z, pitch)


def materialize_continuous_presence(
    keyframes: list[dict],
    duration: float,
    pitch: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    """Extend latent actor presence from 0% through 100% of the scene.

    Detector-backed points remain the only observed evidence. Long internal
    gaps are low-confidence interpolations; the tails roam within a small,
    deterministic neighbourhood of the nearest observed position.
    """

    if not keyframes:
        return [], {
            "policy": "continuous-latent",
            "coverage": 0.0,
            "observationCount": 0,
            "inferredKeyframeCount": 0,
        }

    duration = max(0.0, float(duration))
    observed = sorted(
        (
            {
                **keyframe,
                "observed": True,
                "presenceState": "observed",
            }
            for keyframe in keyframes
        ),
        key=lambda item: float(item["t"]),
    )
    positive_deltas = [
        float(right["t"]) - float(left["t"])
        for left, right in zip(observed, observed[1:])
        if 1e-6 < float(right["t"]) - float(left["t"]) <= 1.0
    ]
    cadence = float(np.median(positive_deltas)) if positive_deltas else 0.2
    fill_step = max(0.25, min(0.75, cadence * 2.0))
    gap_threshold = max(0.6, cadence * 2.5)
    inferred: list[dict] = []

    first = observed[0]
    first_time = max(0.0, float(first["t"]))
    if first_time > 1e-6:
        times = [0.0]
        cursor = fill_step
        while cursor < first_time - 1e-6:
            times.append(cursor)
            cursor += fill_step
        base_uncertainty = float(first.get("positionUncertaintyMetres") or 1.0)
        for time in times:
            elapsed = first_time - time
            x, z = _roaming_presence_position(
                first,
                elapsed,
                seed,
                pitch,
                reverse=True,
            )
            inferred.append(
                _presence_keyframe(
                    first,
                    time,
                    x,
                    z,
                    "inferred-before-first",
                    min(18.0, base_uncertainty + 1.5 + elapsed * 1.8),
                )
            )

    for left, right in zip(observed, observed[1:]):
        left_time = float(left["t"])
        right_time = float(right["t"])
        gap = right_time - left_time
        if gap <= gap_threshold:
            continue
        cursor = left_time + fill_step
        base_uncertainty = max(
            float(left.get("positionUncertaintyMetres") or 1.0),
            float(right.get("positionUncertaintyMetres") or 1.0),
        )
        while cursor < right_time - 1e-6:
            mix = (cursor - left_time) / gap
            x = float(left["x"]) + (float(right["x"]) - float(left["x"])) * mix
            z = float(left["z"]) + (float(right["z"]) - float(left["z"])) * mix
            x, z = _bounded_pitch_position(x, z, pitch)
            inferred.append(
                _presence_keyframe(
                    left,
                    cursor,
                    x,
                    z,
                    "inferred-gap",
                    min(18.0, base_uncertainty + 1.0 + gap * 0.8),
                )
            )
            cursor += fill_step

    last = observed[-1]
    last_time = min(duration, float(last["t"]))
    if last_time < duration - 1e-6:
        times: list[float] = []
        cursor = last_time + fill_step
        while cursor < duration - 1e-6:
            times.append(cursor)
            cursor += fill_step
        times.append(duration)
        base_uncertainty = float(last.get("positionUncertaintyMetres") or 1.0)
        for time in times:
            elapsed = time - last_time
            x, z = _roaming_presence_position(last, elapsed, seed, pitch)
            inferred.append(
                _presence_keyframe(
                    last,
                    time,
                    x,
                    z,
                    "inferred-after-last",
                    min(18.0, base_uncertainty + 1.5 + elapsed * 1.8),
                )
            )

    combined = sorted([*observed, *inferred], key=lambda item: float(item["t"]))
    # Prefer observed evidence when floating-point rounding produces the same
    # timestamp as an inferred sample.
    deduplicated: list[dict] = []
    for keyframe in combined:
        if (
            deduplicated
            and abs(float(deduplicated[-1]["t"]) - float(keyframe["t"])) < 1e-6
        ):
            if keyframe.get("observed"):
                deduplicated[-1] = keyframe
            continue
        deduplicated.append(keyframe)

    observed_start = float(observed[0]["t"])
    observed_end = float(observed[-1]["t"])
    inferred_count = sum(item.get("observed") is False for item in deduplicated)
    coverage = (
        1.0
        if deduplicated
        and float(deduplicated[0]["t"]) <= 1e-6
        and float(deduplicated[-1]["t"]) >= duration - 1e-6
        else 0.0
    )
    return deduplicated, {
        "policy": "continuous-latent",
        "coverage": coverage,
        "observationCount": len(observed),
        "inferredKeyframeCount": inferred_count,
        "observedStart": round(observed_start, 3),
        "observedEnd": round(observed_end, 3),
        "observedSpanRatio": (
            round(max(0.0, observed_end - observed_start) / duration, 3)
            if duration > 1e-6
            else 1.0
        ),
        "sampleCadenceSeconds": round(cadence, 3),
    }


__all__ = ["materialize_continuous_presence"]
