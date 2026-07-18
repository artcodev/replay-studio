from __future__ import annotations

"""Trajectory speed, continuity, and fragmentation measurements."""

from dataclasses import replace
from math import hypot, isfinite
from statistics import median
from typing import Any, Iterable, Sequence

from .quality_evidence import finite_number, frame_time, percentile, sample_cadence
from .quality_measurement_domain import (
    ContinuityMeasurements,
    MotionMeasurements,
    SpeedMeasurements,
)
from .quality_policy import QualityThresholds


def _valid_track_points(track: dict[str, Any]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for keyframe in track.get("keyframes") or []:
        if not isinstance(keyframe, dict):
            continue
        t, x, z = (
            finite_number(keyframe.get(key)) for key in ("t", "x", "z")
        )
        confidence = finite_number(keyframe.get("confidence"))
        # Inferred presence is rendering state, not observed physics evidence.
        if (
            t is None
            or x is None
            or z is None
            or confidence == 0.0
            or keyframe.get("observed") is False
        ):
            continue
        points.append({"t": t, "x": x, "z": z})
    return sorted(points, key=lambda item: item["t"])


def _speed_measurement(
    series: Iterable[tuple[str, Sequence[dict[str, float]]]],
    limit_mps: float,
    *,
    source: str,
) -> SpeedMeasurements:
    speeds: list[float] = []
    violating_tracks: set[str] = set()
    for track_id, points in series:
        for left, right in zip(points, points[1:]):
            elapsed = right["t"] - left["t"]
            if elapsed <= 1e-6:
                continue
            speed = hypot(
                right["x"] - left["x"], right["z"] - left["z"]
            ) / elapsed
            if isfinite(speed):
                speeds.append(speed)
                if speed > limit_mps:
                    violating_tracks.add(track_id)
    violations = sum(speed > limit_mps for speed in speeds)
    return SpeedMeasurements(
        ratio=violations / len(speeds) if speeds else None,
        violations=violations,
        segment_count=len(speeds),
        p95_metres_per_second=percentile(speeds, 0.95),
        maximum_metres_per_second=max(speeds) if speeds else None,
        violating_track_count=len(violating_tracks),
        source=source,
    )

def _continuity_measurement(
    tracks: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
) -> ContinuityMeasurements:
    series = [
        (str(track.get("id") or "unknown"), _valid_track_points(track))
        for track in tracks
    ]
    evidence_times = [
        time for item in evidence if (time := frame_time(item)) is not None
    ]
    cadence = sample_cadence(evidence_times)
    if cadence is None:
        short_deltas = sorted(
            right["t"] - left["t"]
            for _, points in series
            for left, right in zip(points, points[1:])
            if 1e-6 < right["t"] - left["t"] <= 1.0
        )
        cadence = percentile(short_deltas, 0.25)
    if cadence is None or cadence <= 0:
        return ContinuityMeasurements(None, None, 0, len(tracks), None, None)

    gap_threshold = max(0.6, cadence * 2.5)
    completeness: list[float] = []
    fragmented_tracks = fragments = 0
    for _, points in series:
        if not points:
            continue
        expected = max(
            1, round((points[-1]["t"] - points[0]["t"]) / cadence) + 1
        )
        completeness.append(min(1.0, len(points) / expected))
        gaps = sum(
            right["t"] - left["t"] > gap_threshold
            for left, right in zip(points, points[1:])
        )
        if gaps:
            fragmented_tracks += 1
            fragments += gaps
    valid_tracks = len(completeness)
    return ContinuityMeasurements(
        median(completeness) if completeness else None,
        fragmented_tracks / valid_tracks if valid_tracks else None,
        fragments,
        valid_tracks,
        cadence,
        gap_threshold,
    )


def collect_motion_measurements(
    scene: dict[str, Any],
    diagnostics: dict[str, Any],
    evidence: list[dict[str, Any]],
    thresholds: QualityThresholds,
) -> MotionMeasurements:
    tracks = [
        track
        for track in (scene.get("payload") or {}).get("tracks") or []
        if isinstance(track, dict)
    ]
    player_speed = _speed_measurement(
        (
            (str(track.get("id") or "unknown"), _valid_track_points(track))
            for track in tracks
        ),
        thresholds.player_speed_limit_mps,
        source="trajectory",
    )
    prefilter_samples = int(
        finite_number(diagnostics.get("preFilterSpeedSampleCount")) or 0
    )
    prefilter_violations = int(
        finite_number(diagnostics.get("preFilterSpeedViolationCount")) or 0
    )
    if prefilter_samples > 0:
        published_ratio = player_speed.ratio
        published_segment_count = player_speed.segment_count
        player_speed = replace(
            player_speed,
            published_ratio=published_ratio,
            published_segment_count=published_segment_count,
        )
        prefilter_ratio = prefilter_violations / prefilter_samples
        if published_ratio is None or prefilter_ratio > published_ratio:
            maximum = finite_number(
                diagnostics.get("preFilterMaximumSpeedMetresPerSecond")
            )
            player_speed = replace(
                player_speed,
                ratio=prefilter_ratio,
                violations=prefilter_violations,
                segment_count=prefilter_samples,
                maximum_metres_per_second=(
                    max(
                        maximum,
                        float(player_speed.maximum_metres_per_second or 0.0),
                    )
                    if maximum is not None
                    else player_speed.maximum_metres_per_second
                ),
                source="trajectory-pre-filter",
            )

    ball_frames = [
        frame
        for frame in ((scene.get("payload") or {}).get("ball") or {}).get(
            "keyframes"
        )
        or []
        if isinstance(frame, dict)
    ]
    ball_speed = _speed_measurement(
        [("ball", _valid_track_points({"keyframes": ball_frames}))],
        thresholds.ball_speed_limit_mps,
        source="trajectory",
    )
    return MotionMeasurements(
        player_speed=player_speed,
        ball_speed=ball_speed,
        continuity=_continuity_measurement(tracks, evidence),
        player_track_count=len(tracks),
    )
