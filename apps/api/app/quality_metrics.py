"""Evidence-based quality metrics for a reconstructed football scene.

The reconstruction pipeline deliberately keeps this module independent from CV
models and persistence.  It accepts the public scene document plus optional
per-frame calibration evidence and returns a JSON-serialisable QA report.  A
missing measurement is reported as ``unknown``; it is never converted into a
successful gate.

The metrics in this module are engineering guardrails, not replacements for
dataset metrics such as JaC@5, HOTA, or GS-HOTA.  Their purpose is to prevent a
completed computation from being presented as a trustworthy reconstruction
when its calibration or trajectories contain obvious failures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import hypot, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal, Sequence

from .identity_metrics import evaluate_identity_assignments


GateStatus = Literal["pass", "review", "reject", "unknown"]


@dataclass(frozen=True)
class QualityThresholds:
    """Initial engineering gates, to be tuned on the project's gold set."""

    calibration_coverage_pass: float = 0.90
    calibration_coverage_review: float = 0.75
    calibration_gap_pass_seconds: float = 0.60
    calibration_gap_review_seconds: float = 1.20
    temporal_uncertainty_p95_pass_metres: float = 2.50
    temporal_uncertainty_p95_review_metres: float = 5.00
    reprojection_p50_pass_px: float = 4.0
    reprojection_p50_review_px: float = 8.0
    reprojection_p95_pass_px: float = 8.0
    reprojection_p95_review_px: float = 20.0
    visible_side_agreement_pass: float = 0.90
    visible_side_agreement_review: float = 0.80
    semantic_alignment_f1_p10_pass: float = 0.15
    semantic_alignment_f1_p10_review: float = 0.08
    inlier_ratio_p10_pass: float = 0.70
    inlier_ratio_p10_review: float = 0.50
    projection_fallback_pass: float = 0.0
    projection_fallback_review: float = 0.05
    boundary_clamp_pass: float = 0.005
    boundary_clamp_review: float = 0.02
    player_speed_limit_mps: float = 14.0
    player_speed_violation_pass: float = 0.01
    player_speed_violation_review: float = 0.05
    ball_speed_limit_mps: float = 50.0
    ball_speed_violation_pass: float = 0.01
    ball_speed_violation_review: float = 0.05
    ball_observed_coverage_pass: float = 0.65
    ball_observed_coverage_review: float = 0.35
    ball_published_coverage_pass: float = 0.85
    ball_published_coverage_review: float = 0.60
    track_continuity_pass: float = 0.90
    track_continuity_review: float = 0.75
    track_fragmentation_pass: float = 0.10
    track_fragmentation_review: float = 0.30
    identity_idf1_pass: float = 0.75
    identity_idf1_review: float = 0.55


DEFAULT_THRESHOLDS = QualityThresholds()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _ratio(value: Any) -> float | None:
    result = _number(value)
    if result is None:
        return None
    return min(1.0, max(0.0, result))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
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


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _reconstruction(scene: dict[str, Any]) -> dict[str, Any]:
    return (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        or {}
    )


def _frame_evidence(
    scene: dict[str, Any],
    supplied: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if supplied is not None:
        return [item for item in supplied if isinstance(item, dict)]
    reconstruction = _reconstruction(scene)
    calibration = reconstruction.get("calibration") or {}
    candidates = (
        calibration.get("frameEvidence")
        or reconstruction.get("calibrationFrames")
        or reconstruction.get("frameEvidence")
        or []
    )
    return [item for item in candidates if isinstance(item, dict)]


def _frame_time(item: dict[str, Any]) -> float | None:
    for key in ("sceneTime", "time", "t"):
        value = _number(item.get(key))
        if value is not None:
            return value
    return None


def _accepted_frame(item: dict[str, Any]) -> bool:
    return str(item.get("status") or "").lower() in {"accepted", "ready", "valid"}


def _projection_source(item: dict[str, Any]) -> str | None:
    projection = item.get("projection") or {}
    source = projection.get("source") if isinstance(projection, dict) else None
    source = source or item.get("projectionSource")
    if source is None:
        return None
    return str(source).strip().lower()


def _is_fallback_source(source: str | None) -> bool:
    if not source:
        return False
    return source in {
        "none",
        "fallback",
        "screen",
        "screen-relative",
        "screen-approximate",
        "screen-projected",
        "approximate",
        "representative-approximate",
    }


def _sample_cadence(times: Sequence[float]) -> float | None:
    ordered = sorted(set(times))
    deltas = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    return median(deltas) if deltas else None


def _maximum_invalid_gap(
    evidence: Sequence[dict[str, Any]],
    duration: float | None,
) -> float | None:
    timed = sorted(
        (
            (time, _accepted_frame(item) and not _is_fallback_source(_projection_source(item)))
            for item in evidence
            if (time := _frame_time(item)) is not None
        ),
        key=lambda item: item[0],
    )
    if not timed:
        return None
    cadence = _sample_cadence([item[0] for item in timed]) or 0.0
    longest = 0.0
    run_start: float | None = None
    run_end: float | None = None
    for timestamp, accepted in timed:
        if not accepted:
            run_start = timestamp if run_start is None else run_start
            run_end = timestamp
            continue
        if run_start is not None and run_end is not None:
            longest = max(longest, run_end - run_start + cadence)
        run_start = run_end = None
    if run_start is not None and run_end is not None:
        longest = max(longest, run_end - run_start + cadence)
    if all(not accepted for _, accepted in timed) and duration is not None:
        longest = max(longest, duration)
    return max(0.0, longest)


def _metric(
    value: float | None,
    unit: str,
    *,
    source: str,
    sample_count: int | None = None,
    **details: Any,
) -> dict[str, Any]:
    return {
        "value": _round(value),
        "unit": unit,
        "source": source,
        **({"sampleCount": sample_count} if sample_count is not None else {}),
        **details,
    }


def _lower_gate(
    gate_id: str,
    label: str,
    value: float | None,
    unit: str,
    pass_at: float,
    review_at: float,
    *,
    required: bool = True,
    evidence: str = "measured",
    note: str | None = None,
) -> dict[str, Any]:
    if value is None:
        status: GateStatus = "unknown"
    elif value <= pass_at:
        status = "pass"
    elif value <= review_at:
        status = "review"
    else:
        status = "reject"
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "required": required,
        "value": _round(value),
        "unit": unit,
        "evidence": evidence if value is not None else "missing",
        "thresholds": {"passAtMost": pass_at, "reviewAtMost": review_at},
        **({"note": note} if note else {}),
    }


def _higher_gate(
    gate_id: str,
    label: str,
    value: float | None,
    unit: str,
    pass_at: float,
    review_at: float,
    *,
    required: bool = True,
    evidence: str = "measured",
    note: str | None = None,
) -> dict[str, Any]:
    if value is None:
        status: GateStatus = "unknown"
    elif value >= pass_at:
        status = "pass"
    elif value >= review_at:
        status = "review"
    else:
        status = "reject"
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "required": required,
        "value": _round(value),
        "unit": unit,
        "evidence": evidence if value is not None else "missing",
        "thresholds": {"passAtLeast": pass_at, "reviewAtLeast": review_at},
        **({"note": note} if note else {}),
    }


def _scene_keyframes(scene: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    payload = scene.get("payload") or {}
    result: list[tuple[str, str, dict[str, Any]]] = []
    for track in payload.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        track_id = str(track.get("id") or "unknown")
        for keyframe in track.get("keyframes") or []:
            if isinstance(keyframe, dict):
                result.append(("person", track_id, keyframe))
    ball = payload.get("ball") or {}
    for keyframe in ball.get("keyframes") or []:
        if isinstance(keyframe, dict):
            result.append(("ball", "ball", keyframe))
    return result


def _projection_fallback_metric(
    scene: dict[str, Any],
) -> tuple[float | None, int, int, str]:
    reconstruction = _reconstruction(scene)
    diagnostics = reconstruction.get("diagnostics") or {}
    fallback_count = _number(diagnostics.get("projectionFallbackCount"))
    observation_count = _number(diagnostics.get("projectedObservationCount"))
    if fallback_count is not None and observation_count is not None and observation_count > 0:
        return fallback_count / observation_count, int(fallback_count), int(observation_count), "diagnostics"

    explicit = [
        source
        for kind, _, keyframe in _scene_keyframes(scene)
        if kind != "person" or keyframe.get("observed") is not False
        if (source := _projection_source(keyframe)) is not None
    ]
    if explicit:
        count = sum(_is_fallback_source(source) for source in explicit)
        return count / len(explicit), count, len(explicit), "keyframe-provenance"

    all_keyframes = _scene_keyframes(scene)
    coordinate_space = str(reconstruction.get("coordinateSpace") or "").lower()
    if all_keyframes and coordinate_space.startswith("screen-"):
        return 1.0, len(all_keyframes), len(all_keyframes), "coordinate-space-inference"
    return None, 0, len(all_keyframes), "missing"


def _boundary_metric(
    scene: dict[str, Any],
) -> tuple[float | None, int, int, str]:
    pitch = (scene.get("payload") or {}).get("pitch") or {}
    length = _number(pitch.get("length")) or 105.0
    width = _number(pitch.get("width")) or 68.0
    half_length, half_width = length / 2.0, width / 2.0
    points = _scene_keyframes(scene)
    observed = 0
    clamped = 0
    explicit_flags = 0
    for kind, _, keyframe in points:
        if kind == "person" and keyframe.get("observed") is False:
            continue
        x, z = _number(keyframe.get("x")), _number(keyframe.get("z"))
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
    evidence = "explicit-keyframe-flags" if explicit_flags == observed else "boundary-contact-inference"
    return clamped / observed, clamped, observed, evidence


def _valid_track_points(track: dict[str, Any]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for keyframe in track.get("keyframes") or []:
        if not isinstance(keyframe, dict):
            continue
        t, x, z = (_number(keyframe.get(key)) for key in ("t", "x", "z"))
        confidence = _number(keyframe.get("confidence"))
        # Inferred presence keeps an actor alive outside detector observations,
        # but it must never improve physics or continuity QA artificially.
        # Legacy zero-confidence endpoints remain excluded as well.
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


def _speed_metric(
    series: Iterable[tuple[str, Sequence[dict[str, float]]]],
    limit_mps: float,
) -> dict[str, Any]:
    speeds: list[float] = []
    violating_tracks: set[str] = set()
    for track_id, points in series:
        for left, right in zip(points, points[1:]):
            elapsed = right["t"] - left["t"]
            if elapsed <= 1e-6:
                continue
            speed = hypot(right["x"] - left["x"], right["z"] - left["z"]) / elapsed
            if isfinite(speed):
                speeds.append(speed)
                if speed > limit_mps:
                    violating_tracks.add(track_id)
    violations = sum(speed > limit_mps for speed in speeds)
    return {
        "ratio": violations / len(speeds) if speeds else None,
        "violations": violations,
        "segments": len(speeds),
        "p95": _percentile(speeds, 0.95),
        "maximum": max(speeds) if speeds else None,
        "violatingTrackCount": len(violating_tracks),
    }


def _track_continuity(
    scene: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tracks = [
        track
        for track in (scene.get("payload") or {}).get("tracks") or []
        if isinstance(track, dict)
    ]
    series = [(str(track.get("id") or "unknown"), _valid_track_points(track)) for track in tracks]
    evidence_times = [time for item in evidence if (time := _frame_time(item)) is not None]
    cadence = _sample_cadence(evidence_times)
    if cadence is None:
        short_deltas = sorted(
            right["t"] - left["t"]
            for _, points in series
            for left, right in zip(points, points[1:])
            if 1e-6 < right["t"] - left["t"] <= 1.0
        )
        cadence = _percentile(short_deltas, 0.25)
    if cadence is None or cadence <= 0:
        return {
            "medianCompleteness": None,
            "fragmentedTrackRatio": None,
            "fragmentCount": 0,
            "trackCount": len(tracks),
            "sampleCadenceSeconds": None,
            "gapThresholdSeconds": None,
        }

    gap_threshold = max(0.6, cadence * 2.5)
    completeness: list[float] = []
    fragmented_tracks = 0
    fragments = 0
    for _, points in series:
        if not points:
            continue
        expected = max(1, round((points[-1]["t"] - points[0]["t"]) / cadence) + 1)
        completeness.append(min(1.0, len(points) / expected))
        gaps = sum(
            right["t"] - left["t"] > gap_threshold
            for left, right in zip(points, points[1:])
        )
        if gaps:
            fragmented_tracks += 1
            fragments += gaps
    valid_tracks = len(completeness)
    return {
        "medianCompleteness": median(completeness) if completeness else None,
        "fragmentedTrackRatio": fragmented_tracks / valid_tracks if valid_tracks else None,
        "fragmentCount": fragments,
        "trackCount": valid_tracks,
        "sampleCadenceSeconds": cadence,
        "gapThresholdSeconds": gap_threshold,
    }


def evaluate_reconstruction_quality(
    scene: dict[str, Any],
    frame_evidence: Iterable[dict[str, Any]] | None = None,
    *,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Return a serialisable QA report without mutating ``scene``.

    ``frame_evidence`` follows the reconstruction contract used by the API:
    each entry contains ``sceneTime``, ``status``, ``projectionSource``, and
    optional ``reprojectionError`` / ``inlierRatio``.  When omitted, evidence
    is read from ``reconstruction.calibration.frameEvidence`` with compatibility
    fallbacks to ``reconstruction.calibrationFrames``.
    """

    reconstruction = _reconstruction(scene)
    diagnostics = reconstruction.get("diagnostics") or {}
    evidence = _frame_evidence(scene, frame_evidence)
    duration = _number(scene.get("duration"))
    manual_calibration = str((reconstruction.get("pitchCalibration") or {}).get("method") or "").startswith("manual")

    if evidence:
        accepted = [item for item in evidence if _accepted_frame(item)]
        coverage = len(accepted) / len(evidence)
        direct = [
            item
            for item in accepted
            if _projection_source(item) in {"direct", "manual-direct"}
        ]
        temporal = [
            item
            for item in accepted
            if str(_projection_source(item) or "").startswith("temporal-")
        ]
        direct_coverage = len(direct) / len(evidence)
        temporal_coverage = len(temporal) / len(evidence)
        temporal_ambiguity_count = sum(
            item.get("solutionStatus") == "ambiguous" for item in evidence
        )
        coverage_source = "frame-evidence"
    else:
        accepted = []
        direct = []
        temporal = []
        frame_count = _number(reconstruction.get("frameCount"))
        calibrated_count = _number(diagnostics.get("calibratedFrameCount"))
        coverage = _ratio(diagnostics.get("calibrationFrameCoverage"))
        if coverage is None and frame_count and calibrated_count is not None:
            coverage = calibrated_count / frame_count
        direct_coverage = None
        temporal_coverage = None
        temporal_ambiguity_count = 0
        coverage_source = "diagnostics" if coverage is not None else "missing"

    temporal_uncertainties = [
        value
        for item in temporal
        if (
            value := _number((item.get("uncertainty") or {}).get("p95Metres"))
            or _number(item.get("positionUncertaintyMetres"))
        )
        is not None
    ]
    temporal_uncertainty_p95 = _percentile(temporal_uncertainties, 0.95)

    residuals_p50 = [
        value
        for item in accepted
        if (value := _number(item.get("reprojectionError"))) is not None
    ]
    residuals_p95 = [
        value
        for item in accepted
        if (
            value := _number(item.get("reprojectionP95"))
            or _number(item.get("reprojectionError"))
        )
        is not None
    ]
    inlier_ratios = []
    for item in accepted:
        ratio = _ratio(item.get("inlierRatio"))
        if ratio is None:
            keypoints = _number(item.get("keypointCount"))
            inliers = _number(item.get("inlierCount"))
            if keypoints and inliers is not None:
                ratio = _ratio(inliers / keypoints)
        if ratio is not None:
            inlier_ratios.append(ratio)
    residual_p50 = _percentile(residuals_p50, 0.50)
    # Each frame may expose a distribution-aware p95. The shot-level value is
    # the p95 of those per-frame tails, not the p95 of per-frame medians.
    residual_p95 = _percentile(residuals_p95, 0.95)
    inlier_p10 = _percentile(inlier_ratios, 0.10)
    alignment_f1_values = [
        value
        for item in accepted
        if (
            value := _ratio((item.get("alignmentMetrics") or {}).get("f1"))
        )
        is not None
    ]
    alignment_f1_p10 = _percentile(alignment_f1_values, 0.10)
    orientation_observations = direct
    if not orientation_observations:
        seen_orientation_anchors: set[tuple[Any, ...]] = set()
        orientation_observations = []
        for item in accepted:
            anchor_key = tuple(
                (item.get("temporal") or {}).get("anchorFrameIndices") or []
            ) or (str(_projection_source(item) or "unknown"),)
            if anchor_key in seen_orientation_anchors:
                continue
            seen_orientation_anchors.add(anchor_key)
            orientation_observations.append(item)
    known_sides = [
        str(item.get("visiblePitchSide"))
        for item in orientation_observations
        if item.get("visiblePitchSide") in {"left", "right"}
    ]
    side_counts = {side: known_sides.count(side) for side in ("left", "right")}
    visible_side = max(side_counts, key=side_counts.get) if known_sides else None
    side_agreement = (
        side_counts[visible_side] / len(known_sides)
        if visible_side is not None
        else None
    )
    ground_p50_values = [
        value
        for item in accepted
        if (value := _number(item.get("groundErrorP50Metres"))) is not None
    ]
    ground_p95_values = [
        value
        for item in accepted
        if (
            value := _number(item.get("groundErrorP95Metres"))
            or _number(item.get("groundErrorP50Metres"))
        )
        is not None
    ]
    pitch_calibration = reconstruction.get("pitchCalibration") or {}
    representative_error = next(
        (
            value
            for value in (
                _number(pitch_calibration.get("alignmentError")),
                _number(pitch_calibration.get("reprojectionError")),
                _number(diagnostics.get("calibrationReprojectionError")),
            )
            if value is not None
        ),
        None,
    )
    calibration_gap = _maximum_invalid_gap(evidence, duration) if evidence else None

    fallback_ratio, fallback_count, projected_count, fallback_source = _projection_fallback_metric(scene)
    clamp_ratio, clamp_count, position_count, clamp_source = _boundary_metric(scene)

    tracks = [
        track
        for track in (scene.get("payload") or {}).get("tracks") or []
        if isinstance(track, dict)
    ]
    player_speed = _speed_metric(
        ((str(track.get("id") or "unknown"), _valid_track_points(track)) for track in tracks),
        thresholds.player_speed_limit_mps,
    )
    player_speed_source = "trajectory"
    prefilter_speed_samples = int(
        _number(diagnostics.get("preFilterSpeedSampleCount")) or 0
    )
    prefilter_speed_violations = int(
        _number(diagnostics.get("preFilterSpeedViolationCount")) or 0
    )
    if prefilter_speed_samples > 0:
        prefilter_ratio = prefilter_speed_violations / prefilter_speed_samples
        published_ratio = player_speed["ratio"]
        player_speed["publishedRatio"] = published_ratio
        player_speed["publishedSegments"] = player_speed["segments"]
        # Reconstruction may split an implausible path and keep its longest
        # fragment for rendering. QA must still see the discarded jump.
        if published_ratio is None or prefilter_ratio > published_ratio:
            player_speed["ratio"] = prefilter_ratio
            player_speed["violations"] = prefilter_speed_violations
            player_speed["segments"] = prefilter_speed_samples
            maximum = _number(
                diagnostics.get("preFilterMaximumSpeedMetresPerSecond")
            )
            if maximum is not None:
                player_speed["maximum"] = max(
                    maximum,
                    float(player_speed["maximum"] or 0.0),
                )
            player_speed_source = "trajectory-pre-filter"
    ball_frames = [
        frame
        for frame in ((scene.get("payload") or {}).get("ball") or {}).get("keyframes") or []
        if isinstance(frame, dict)
    ]
    ball_series = _valid_track_points({"keyframes": ball_frames})
    ball_speed = _speed_metric([("ball", ball_series)], thresholds.ball_speed_limit_mps)
    ball_tracking = (
        ((scene.get("payload") or {}).get("ball") or {}).get("diagnostics")
        or (reconstruction.get("ballDetection") or {}).get("tracking")
        or diagnostics.get("ballTracking")
        or {}
    )
    ball_observed_coverage = _ratio(ball_tracking.get("observedCoverage"))
    ball_published_coverage = _ratio(ball_tracking.get("publishedCoverage"))
    ball_tracking_frame_count = int(_number(ball_tracking.get("frameCount")) or 0)
    ball_longest_gap = _number((ball_tracking.get("gaps") or {}).get("longestGapSeconds"))
    ball_path_margin = _number(ball_tracking.get("pathCostMargin"))
    continuity = _track_continuity(scene, evidence)
    payload = scene.get("payload") or {}
    validation_ground_truth = (
        payload.get("validationGroundTruth")
        or payload.get("groundTruth")
        or {}
    )
    identity_assignment_frame_rate = (
        _number(
            validation_ground_truth.get("identityAssignmentFrameRate")
            or validation_ground_truth.get("identityAssignmentsFps")
        )
        if isinstance(validation_ground_truth, dict)
        else None
    )
    identity_validation = evaluate_identity_assignments(
        validation_ground_truth.get("identityAssignments")
        if isinstance(validation_ground_truth, dict)
        else None,
        frame_rate=identity_assignment_frame_rate,
    )

    metrics = {
        "calibrationCoverage": _metric(
            coverage,
            "ratio",
            source=coverage_source,
            sample_count=len(evidence) or int(_number(reconstruction.get("frameCount")) or 0),
            acceptedFrames=len(accepted) if evidence else int(_number(diagnostics.get("calibratedFrameCount")) or 0),
        ),
        "directCalibrationCoverage": _metric(
            direct_coverage,
            "ratio",
            source="frame-evidence" if evidence else "missing",
            sample_count=len(evidence) if evidence else None,
        ),
        "temporalCalibrationCoverage": _metric(
            temporal_coverage,
            "ratio",
            source="frame-evidence" if evidence else "missing",
            sample_count=len(temporal),
            recoveredFrames=len(temporal),
        ),
        "temporalCalibrationUncertaintyP95": _metric(
            temporal_uncertainty_p95,
            "metres",
            source="temporal-hypothesis-graph" if temporal_uncertainties else "missing",
            sample_count=len(temporal_uncertainties),
        ),
        "temporalCalibrationAmbiguityRatio": _metric(
            temporal_ambiguity_count / len(evidence) if evidence else None,
            "ratio",
            source="frame-evidence" if evidence else "missing",
            sample_count=len(evidence),
            ambiguousFrames=temporal_ambiguity_count,
        ),
        "maxCalibrationGap": _metric(
            calibration_gap,
            "seconds",
            source="frame-evidence" if evidence else "missing",
            sample_count=len(evidence) if evidence else None,
        ),
        "calibrationResidualP50": _metric(
            residual_p50,
            "pixels",
            source="frame-evidence" if residuals_p50 else "missing",
            sample_count=len(residuals_p50),
        ),
        "calibrationResidualP95": _metric(
            residual_p95,
            "pixels",
            source="frame-evidence" if residuals_p95 else "missing",
            sample_count=len(residuals_p95),
        ),
        "representativeReprojectionError": _metric(
            representative_error,
            "pixels",
            source="representative-calibration" if representative_error is not None else "missing",
            sample_count=1 if representative_error is not None else 0,
        ),
        "calibrationInlierRatioP10": _metric(
            inlier_p10,
            "ratio",
            source="frame-evidence" if inlier_ratios else "missing",
            sample_count=len(inlier_ratios),
        ),
        "semanticAlignmentF1P10": _metric(
            alignment_f1_p10,
            "ratio",
            source="frame-evidence" if alignment_f1_values else "missing",
            sample_count=len(alignment_f1_values),
        ),
        "visiblePitchSideAgreement": _metric(
            side_agreement,
            "ratio",
            source="frame-evidence" if known_sides else "missing",
            sample_count=len(known_sides),
            visiblePitchSide=visible_side,
            sideVotes=side_counts,
        ),
        "groundErrorP50": _metric(
            _percentile(ground_p50_values, 0.50),
            "metres",
            source="frame-evidence" if ground_p50_values else "missing",
            sample_count=len(ground_p50_values),
        ),
        "groundErrorP95": _metric(
            _percentile(ground_p95_values, 0.95),
            "metres",
            source="frame-evidence" if ground_p95_values else "missing",
            sample_count=len(ground_p95_values),
        ),
        "projectionFallbackRatio": _metric(
            fallback_ratio,
            "ratio",
            source=fallback_source,
            sample_count=projected_count,
            fallbackObservations=fallback_count,
        ),
        "boundaryClampRatio": _metric(
            clamp_ratio,
            "ratio",
            source=clamp_source,
            sample_count=position_count,
            clampedOrBoundaryObservations=clamp_count,
            note=(
                "Exact boundary contacts are a conservative proxy because legacy keyframes lack an explicit clamp flag."
                if clamp_source == "boundary-contact-inference"
                else None
            ),
        ),
        "playerSpeedViolationRatio": _metric(
            player_speed["ratio"],
            "ratio",
            source=player_speed_source,
            sample_count=player_speed["segments"],
            limitMetresPerSecond=thresholds.player_speed_limit_mps,
            violations=player_speed["violations"],
            p95MetresPerSecond=_round(player_speed["p95"]),
            maxMetresPerSecond=_round(player_speed["maximum"]),
            violatingTrackCount=player_speed["violatingTrackCount"],
            publishedRatio=_round(player_speed.get("publishedRatio")),
            publishedSampleCount=player_speed.get("publishedSegments"),
        ),
        "ballSpeedViolationRatio": _metric(
            ball_speed["ratio"],
            "ratio",
            source="trajectory" if ball_speed["segments"] else "missing",
            sample_count=ball_speed["segments"],
            limitMetresPerSecond=thresholds.ball_speed_limit_mps,
            violations=ball_speed["violations"],
            p95MetresPerSecond=_round(ball_speed["p95"]),
            maxMetresPerSecond=_round(ball_speed["maximum"]),
        ),
        "ballObservedCoverage": _metric(
            ball_observed_coverage,
            "ratio",
            source="ball-temporal-resolver" if ball_tracking else "missing",
            sample_count=ball_tracking_frame_count,
            observedFrames=int(_number(ball_tracking.get("observedFrameCount")) or 0),
            occludedFrames=int(_number(ball_tracking.get("occludedFrameCount")) or 0),
        ),
        "ballPublishedCoverage": _metric(
            ball_published_coverage,
            "ratio",
            source="ball-temporal-resolver" if ball_tracking else "missing",
            sample_count=ball_tracking_frame_count,
            inferredFrames=int(_number(ball_tracking.get("inferredFrameCount")) or 0),
        ),
        "ballLongestUnresolvedGap": _metric(
            ball_longest_gap,
            "seconds",
            source="ball-temporal-resolver" if ball_tracking else "missing",
            sample_count=int(_number((ball_tracking.get("gaps") or {}).get("gapCount")) or 0),
        ),
        "ballPathCostMargin": _metric(
            ball_path_margin,
            "cost",
            source="ball-temporal-resolver" if ball_tracking else "missing",
            sample_count=1 if ball_path_margin is not None else 0,
            note="A larger best-vs-runner-up margin indicates a less ambiguous global trajectory.",
        ),
        "trackContinuity": _metric(
            continuity["medianCompleteness"],
            "ratio",
            source="trajectory" if continuity["medianCompleteness"] is not None else "missing",
            sample_count=continuity["trackCount"],
            sampleCadenceSeconds=_round(continuity["sampleCadenceSeconds"]),
            gapThresholdSeconds=_round(continuity["gapThresholdSeconds"]),
        ),
        "trackFragmentationRatio": _metric(
            continuity["fragmentedTrackRatio"],
            "ratio",
            source="trajectory" if continuity["fragmentedTrackRatio"] is not None else "missing",
            sample_count=continuity["trackCount"],
            fragmentCount=continuity["fragmentCount"],
        ),
        "identityIdf1": _metric(
            _number(identity_validation.get("idf1")),
            "ratio",
            source=(
                "labelled-identity-assignments"
                if identity_validation.get("groundTruthAvailable")
                else "ground-truth-unavailable"
            ),
            sample_count=int(identity_validation.get("sampleCount") or 0),
            idPrecision=identity_validation.get("idPrecision"),
            idRecall=identity_validation.get("idRecall"),
            idSwitchCount=identity_validation.get("idSwitchCount"),
            duplicateOverlapSeconds=identity_validation.get("duplicateOverlapSeconds"),
            note=(
                None
                if identity_validation.get("groundTruthAvailable")
                else "Identity accuracy cannot be inferred from observation coverage."
            ),
        ),
    }

    coverage_for_gate = None if manual_calibration and not evidence else coverage
    coverage_note = (
        "Manual calibration requires per-frame validation; one anchor frame does not prove shot-wide coverage."
        if manual_calibration and not evidence
        else None
    )
    reprojection_evidence = (
        "frame-evidence"
        if residuals_p50 or residuals_p95
        else "representative"
    )
    p50_for_gate = residual_p50 if residual_p50 is not None else representative_error
    p95_for_gate = residual_p95 if residual_p95 is not None else representative_error
    gates = [
        _higher_gate(
            "calibration-coverage",
            "Accepted calibration coverage",
            coverage_for_gate,
            "ratio",
            thresholds.calibration_coverage_pass,
            thresholds.calibration_coverage_review,
            note=coverage_note,
        ),
        _lower_gate(
            "calibration-gap",
            "Longest gap without accepted calibration",
            calibration_gap,
            "seconds",
            thresholds.calibration_gap_pass_seconds,
            thresholds.calibration_gap_review_seconds,
            required=True,
            evidence="frame-evidence",
            note="Unknown for legacy runs without per-frame calibration evidence.",
        ),
        _lower_gate(
            "temporal-uncertainty",
            "Recovered calibration uncertainty p95",
            temporal_uncertainty_p95,
            "metres",
            thresholds.temporal_uncertainty_p95_pass_metres,
            thresholds.temporal_uncertainty_p95_review_metres,
            required=bool(temporal),
            evidence=(
                "temporal-hypothesis-graph" if temporal_uncertainties else "missing"
            ),
            note="Required only when metric frames are recovered from camera-motion hypotheses.",
        ),
        _lower_gate(
            "reprojection-p50",
            "Calibration reprojection error p50",
            p50_for_gate,
            "pixels",
            thresholds.reprojection_p50_pass_px,
            thresholds.reprojection_p50_review_px,
            evidence=reprojection_evidence,
            note=("Uses one representative frame; per-frame residuals are required for a distribution." if not residuals_p50 else None),
        ),
        _lower_gate(
            "reprojection-p95",
            "Calibration reprojection error p95",
            p95_for_gate,
            "pixels",
            thresholds.reprojection_p95_pass_px,
            thresholds.reprojection_p95_review_px,
            evidence=reprojection_evidence,
            note=("Uses one representative frame; per-frame residuals are required for a true p95." if not residuals_p95 else None),
        ),
        _higher_gate(
            "inlier-ratio-p10",
            "Calibration inlier ratio p10",
            inlier_p10,
            "ratio",
            thresholds.inlier_ratio_p10_pass,
            thresholds.inlier_ratio_p10_review,
            required=False,
        ),
        _higher_gate(
            "semantic-line-alignment",
            "Bidirectional semantic-line F1 p10",
            alignment_f1_p10,
            "ratio",
            thresholds.semantic_alignment_f1_p10_pass,
            thresholds.semantic_alignment_f1_p10_review,
            required=bool(evidence),
            note="The score must compare both projected model lines to image markings and image markings back to the model.",
        ),
        _higher_gate(
            "orientation-stability",
            "Visible pitch-side agreement",
            side_agreement,
            "ratio",
            thresholds.visible_side_agreement_pass,
            thresholds.visible_side_agreement_review,
            required=side_agreement is not None,
            note="Visible pitch side is camera evidence and remains independent from team attack direction.",
        ),
        _lower_gate(
            "projection-fallback",
            "Screen-space projection fallback",
            fallback_ratio,
            "ratio",
            thresholds.projection_fallback_pass,
            thresholds.projection_fallback_review,
            evidence=fallback_source,
            note="A metric run must not silently mix pitch and screen-relative coordinates.",
        ),
        _lower_gate(
            "boundary-clamp",
            "Pitch-boundary clamp/contact ratio",
            clamp_ratio,
            "ratio",
            thresholds.boundary_clamp_pass,
            thresholds.boundary_clamp_review,
            evidence=clamp_source,
        ),
        _lower_gate(
            "player-speed",
            f"Player segments above {thresholds.player_speed_limit_mps:g} m/s",
            player_speed["ratio"],
            "ratio",
            thresholds.player_speed_violation_pass,
            thresholds.player_speed_violation_review,
            required=bool(tracks),
        ),
        _higher_gate(
            "identity-idf1",
            "Labelled identity IDF1",
            _number(identity_validation.get("idf1")),
            "ratio",
            thresholds.identity_idf1_pass,
            thresholds.identity_idf1_review,
            required=bool(identity_validation.get("groundTruthAvailable")),
            evidence=(
                "labelled-identity-assignments"
                if identity_validation.get("groundTruthAvailable")
                else "missing"
            ),
            note=(
                None
                if identity_validation.get("groundTruthAvailable")
                else "Ground truth is unavailable; runtime evidence coverage is not an accuracy metric."
            ),
        ),
        _lower_gate(
            "ball-speed",
            f"Ball segments above {thresholds.ball_speed_limit_mps:g} m/s",
            ball_speed["ratio"],
            "ratio",
            thresholds.ball_speed_violation_pass,
            thresholds.ball_speed_violation_review,
            required=False,
        ),
        _higher_gate(
            "ball-observed-coverage",
            "Ball frames supported by detector observations",
            ball_observed_coverage,
            "ratio",
            thresholds.ball_observed_coverage_pass,
            thresholds.ball_observed_coverage_review,
            required=False,
            evidence="ball-temporal-resolver" if ball_tracking else "missing",
            note="Diagnostic only: a genuinely occluded or out-of-frame ball can lower this value.",
        ),
        _higher_gate(
            "ball-published-coverage",
            "Ball frames observed or bounded-interpolated",
            ball_published_coverage,
            "ratio",
            thresholds.ball_published_coverage_pass,
            thresholds.ball_published_coverage_review,
            required=False,
            evidence="ball-temporal-resolver" if ball_tracking else "missing",
        ),
        _higher_gate(
            "track-continuity",
            "Median within-track observation completeness",
            continuity["medianCompleteness"],
            "ratio",
            thresholds.track_continuity_pass,
            thresholds.track_continuity_review,
            required=bool(tracks),
        ),
        _lower_gate(
            "track-fragmentation",
            "Tracks containing long observation gaps",
            continuity["fragmentedTrackRatio"],
            "ratio",
            thresholds.track_fragmentation_pass,
            thresholds.track_fragmentation_review,
            required=bool(tracks),
        ),
    ]

    required_gates = [gate for gate in gates if gate["required"]]
    if any(gate["status"] == "reject" for gate in required_gates):
        verdict = "reject"
    elif any(gate["status"] in {"review", "unknown"} for gate in required_gates):
        verdict = "review"
    else:
        verdict = "pass"
    counts = {
        status: sum(gate["status"] == status for gate in gates)
        for status in ("pass", "review", "reject", "unknown")
    }
    failed = [gate["id"] for gate in required_gates if gate["status"] == "reject"]
    unknown = [gate["id"] for gate in required_gates if gate["status"] == "unknown"]

    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "processingStatus": reconstruction.get("processingStatus") or reconstruction.get("status") or "unknown",
        "verdict": verdict,
        "summary": {
            **counts,
            "failedRequiredGates": failed,
            "unknownRequiredGates": unknown,
        },
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "identityValidation": identity_validation,
        "gates": gates,
        "limitations": [
            {
                "code": "ground-plane-game-state",
                "message": "Player positions are 2D foot points on a 105 x 68 metre ground plane, not 3D body reconstruction.",
            },
            {
                "code": "single-view-visibility",
                "message": "A broadcast view cannot observe off-screen players; missing players must remain unknown unless another synchronized source supplies them.",
            },
            {
                "code": "temporal-camera-hypothesis",
                "message": "Recovered calibration is inferred from direct anchor frames and QA-gated camera motion; its anchor, alternatives, and uncertainty remain attached to every recovered frame.",
            },
            {
                "code": "ball-height-unknown",
                "message": "Single-view ground homography does not recover airborne ball height; a fixed render height is not a measurement.",
            },
            {
                "code": "runtime-gates-not-benchmark",
                "message": "These gates detect engineering failures but do not replace JaC@5, HOTA, or GS-HOTA on a held-out labelled set.",
            },
        ],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction QA gates for a scene JSON document.")
    parser.add_argument("scene", type=Path, help="Path to a scene JSON document")
    parser.add_argument("--evidence", type=Path, help="Optional JSON array of per-frame calibration evidence")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    parser.add_argument(
        "--fail-on",
        choices=("never", "reject", "review"),
        default="never",
        help="Return a non-zero exit status for CI",
    )
    args = parser.parse_args(argv)
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else None
    report = evaluate_reconstruction_quality(scene, evidence)
    print(json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2))
    if args.fail_on == "review" and report["verdict"] in {"review", "reject"}:
        return 2
    if args.fail_on == "reject" and report["verdict"] == "reject":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(_main())
