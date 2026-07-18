from __future__ import annotations

"""Projection provenance and pitch-boundary quality measurements."""

from typing import Any

from .quality_evidence import (
    finite_number,
    is_fallback_projection,
    projection_source,
    reconstruction_document,
    scene_keyframes,
)
from .quality_measurement_domain import ProjectionMeasurements


def _fallback_measurement(
    scene: dict[str, Any],
) -> tuple[float | None, int, int, str]:
    reconstruction = reconstruction_document(scene)
    diagnostics = reconstruction.get("diagnostics") or {}
    fallback_count = finite_number(diagnostics.get("projectionFallbackCount"))
    observation_count = finite_number(diagnostics.get("projectedObservationCount"))
    if (
        fallback_count is not None
        and observation_count is not None
        and observation_count > 0
    ):
        return (
            fallback_count / observation_count,
            int(fallback_count),
            int(observation_count),
            "diagnostics",
        )

    explicit = [
        source
        for kind, _, keyframe in scene_keyframes(scene)
        if kind != "person" or keyframe.get("observed") is not False
        if (source := projection_source(keyframe)) is not None
    ]
    if explicit:
        count = sum(is_fallback_projection(source) for source in explicit)
        return count / len(explicit), count, len(explicit), "keyframe-provenance"

    all_keyframes = scene_keyframes(scene)
    coordinate_space = str(reconstruction.get("coordinateSpace") or "").lower()
    if all_keyframes and coordinate_space.startswith("screen-"):
        return 1.0, len(all_keyframes), len(all_keyframes), "coordinate-space-inference"
    return None, 0, len(all_keyframes), "missing"


def _boundary_measurement(
    scene: dict[str, Any],
) -> tuple[float | None, int, int, str]:
    pitch = (scene.get("payload") or {}).get("pitch") or {}
    length = finite_number(pitch.get("length")) or 105.0
    width = finite_number(pitch.get("width")) or 68.0
    half_length, half_width = length / 2.0, width / 2.0
    observed = clamped = explicit_flags = 0
    for kind, _, keyframe in scene_keyframes(scene):
        if kind == "person" and keyframe.get("observed") is False:
            continue
        x = finite_number(keyframe.get("x"))
        z = finite_number(keyframe.get("z"))
        if x is None or z is None:
            continue
        observed += 1
        projection = keyframe.get("projection") or {}
        explicit = keyframe.get("wasClamped")
        if explicit is None and isinstance(projection, dict):
            explicit = projection.get("clamped")
        if isinstance(explicit, bool):
            explicit_flags += 1
            clamped += int(explicit)
        elif abs(x) >= half_length - 0.01 or abs(z) >= half_width - 0.01:
            clamped += 1
    if not observed:
        return None, 0, 0, "missing"
    source = (
        "explicit-keyframe-flags"
        if explicit_flags == observed
        else "boundary-contact-inference"
    )
    return clamped / observed, clamped, observed, source


def collect_projection_measurements(
    scene: dict[str, Any],
) -> ProjectionMeasurements:
    fallback_ratio, fallback_count, projected_count, fallback_source = (
        _fallback_measurement(scene)
    )
    clamp_ratio, clamp_count, position_count, clamp_source = _boundary_measurement(
        scene
    )
    return ProjectionMeasurements(
        fallback_ratio=fallback_ratio,
        fallback_count=fallback_count,
        projected_count=projected_count,
        fallback_source=fallback_source,
        clamp_ratio=clamp_ratio,
        clamp_count=clamp_count,
        position_count=position_count,
        clamp_source=clamp_source,
    )
