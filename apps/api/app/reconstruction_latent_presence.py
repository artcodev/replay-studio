from __future__ import annotations

"""Materialize uncertain positions only inside a track's observed lifetime."""

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
        # QA and rendering can distinguish this bridge from observed samples via
        # ``observed=False``.
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


def materialize_continuous_presence(
    keyframes: list[dict],
    duration: float,
    pitch: dict,
    _seed: int,
) -> tuple[list[dict], dict]:
    """Keep observed evidence and bridge only long internal detection gaps.

    There is no evidence that an actor exists before its first or after its
    last observation. Those tails must remain absent instead of being rendered
    as a positional guess for the whole scene.
    """

    if not keyframes:
        return [], {
            "policy": "observed-window-with-latent-gaps",
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
    observed_span_ratio = (
        max(0.0, observed_end - observed_start) / duration
        if duration > 1e-6
        else 1.0
    )
    return deduplicated, {
        "policy": "observed-window-with-latent-gaps",
        "coverage": round(observed_span_ratio, 3),
        "observationCount": len(observed),
        "inferredKeyframeCount": inferred_count,
        "observedStart": round(observed_start, 3),
        "observedEnd": round(observed_end, 3),
        "observedSpanRatio": round(observed_span_ratio, 3),
        "sampleCadenceSeconds": round(cadence, 3),
    }


__all__ = ["materialize_continuous_presence"]
