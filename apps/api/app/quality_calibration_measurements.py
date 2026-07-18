from __future__ import annotations

"""Calibration coverage, residual, uncertainty, and orientation measurements."""

from typing import Any, Sequence

from .quality_evidence import (
    bounded_ratio,
    finite_number,
    frame_time,
    is_accepted_frame,
    is_fallback_projection,
    percentile,
    projection_source,
    sample_cadence,
)
from .quality_measurement_domain import CalibrationMeasurements


def _maximum_invalid_gap(
    evidence: Sequence[dict[str, Any]],
    duration: float | None,
) -> float | None:
    timed = sorted(
        (
            (
                time,
                is_accepted_frame(item)
                and not is_fallback_projection(projection_source(item)),
            )
            for item in evidence
            if (time := frame_time(item)) is not None
        ),
        key=lambda item: item[0],
    )
    if not timed:
        return None
    cadence = sample_cadence([item[0] for item in timed]) or 0.0
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


def collect_calibration_measurements(
    scene: dict[str, Any],
    reconstruction: dict[str, Any],
    diagnostics: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> CalibrationMeasurements:
    duration = finite_number(scene.get("duration"))
    manual_calibration = str(
        (reconstruction.get("pitchCalibration") or {}).get("method") or ""
    ).startswith("manual")
    reconstruction_frame_count = int(
        finite_number(reconstruction.get("frameCount")) or 0
    )
    calibrated_frame_value = finite_number(diagnostics.get("calibratedFrameCount"))
    calibrated_frame_count = int(calibrated_frame_value or 0)

    if evidence:
        accepted = [item for item in evidence if is_accepted_frame(item)]
        direct = [
            item
            for item in accepted
            if projection_source(item) in {"direct", "manual-direct"}
        ]
        temporal = [
            item
            for item in accepted
            if str(projection_source(item) or "").startswith("temporal-")
        ]
        coverage = len(accepted) / len(evidence)
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
        coverage = bounded_ratio(diagnostics.get("calibrationFrameCoverage"))
        if (
            coverage is None
            and reconstruction_frame_count
            and calibrated_frame_value is not None
        ):
            coverage = calibrated_frame_count / reconstruction_frame_count
        direct_coverage = temporal_coverage = None
        temporal_ambiguity_count = 0
        coverage_source = "diagnostics" if coverage is not None else "missing"

    temporal_uncertainties = [
        value
        for item in temporal
        if (
            value := finite_number((item.get("uncertainty") or {}).get("p95Metres"))
            or finite_number(item.get("positionUncertaintyMetres"))
        )
        is not None
    ]
    residuals_p50 = [
        value
        for item in accepted
        if (value := finite_number(item.get("reprojectionError"))) is not None
    ]
    residuals_p95 = [
        value
        for item in accepted
        if (
            value := finite_number(item.get("reprojectionP95"))
            or finite_number(item.get("reprojectionError"))
        )
        is not None
    ]
    inlier_ratios: list[float] = []
    for item in accepted:
        ratio = bounded_ratio(item.get("inlierRatio"))
        if ratio is None:
            keypoints = finite_number(item.get("keypointCount"))
            inliers = finite_number(item.get("inlierCount"))
            if keypoints and inliers is not None:
                ratio = bounded_ratio(inliers / keypoints)
        if ratio is not None:
            inlier_ratios.append(ratio)
    alignment_values = [
        value
        for item in accepted
        if (
            value := bounded_ratio((item.get("alignmentMetrics") or {}).get("f1"))
        )
        is not None
    ]

    orientation_observations = direct
    if not orientation_observations:
        seen_anchors: set[tuple[Any, ...]] = set()
        orientation_observations = []
        for item in accepted:
            anchor_key = tuple(
                (item.get("temporal") or {}).get("anchorFrameIndices") or []
            ) or (str(projection_source(item) or "unknown"),)
            if anchor_key in seen_anchors:
                continue
            seen_anchors.add(anchor_key)
            orientation_observations.append(item)
    known_sides = [
        str(item.get("visiblePitchSide"))
        for item in orientation_observations
        if item.get("visiblePitchSide") in {"left", "right"}
    ]
    side_votes = {side: known_sides.count(side) for side in ("left", "right")}
    visible_side = max(side_votes, key=side_votes.get) if known_sides else None
    side_agreement = (
        side_votes[visible_side] / len(known_sides) if visible_side is not None else None
    )
    ground_p50_values = [
        value
        for item in accepted
        if (value := finite_number(item.get("groundErrorP50Metres"))) is not None
    ]
    ground_p95_values = [
        value
        for item in accepted
        if (
            value := finite_number(item.get("groundErrorP95Metres"))
            or finite_number(item.get("groundErrorP50Metres"))
        )
        is not None
    ]
    pitch_calibration = reconstruction.get("pitchCalibration") or {}
    representative_error = next(
        (
            value
            for value in (
                finite_number(pitch_calibration.get("alignmentError")),
                finite_number(pitch_calibration.get("reprojectionError")),
                finite_number(diagnostics.get("calibrationReprojectionError")),
            )
            if value is not None
        ),
        None,
    )
    return CalibrationMeasurements(
        evidence_count=len(evidence),
        accepted_count=len(accepted),
        direct_count=len(direct),
        temporal_count=len(temporal),
        reconstruction_frame_count=reconstruction_frame_count,
        calibrated_frame_count=calibrated_frame_count,
        coverage=coverage,
        direct_coverage=direct_coverage,
        temporal_coverage=temporal_coverage,
        coverage_source=coverage_source,
        temporal_uncertainty_p95=percentile(temporal_uncertainties, 0.95),
        temporal_uncertainty_sample_count=len(temporal_uncertainties),
        temporal_ambiguity_count=temporal_ambiguity_count,
        max_gap_seconds=(
            _maximum_invalid_gap(evidence, duration) if evidence else None
        ),
        residual_p50=percentile(residuals_p50, 0.50),
        residual_p50_sample_count=len(residuals_p50),
        residual_p95=percentile(residuals_p95, 0.95),
        residual_p95_sample_count=len(residuals_p95),
        representative_error=representative_error,
        inlier_ratio_p10=percentile(inlier_ratios, 0.10),
        inlier_ratio_sample_count=len(inlier_ratios),
        alignment_f1_p10=percentile(alignment_values, 0.10),
        alignment_f1_sample_count=len(alignment_values),
        visible_side_agreement=side_agreement,
        visible_side=visible_side,
        side_votes=side_votes,
        side_vote_count=len(known_sides),
        ground_error_p50=percentile(ground_p50_values, 0.50),
        ground_error_p50_sample_count=len(ground_p50_values),
        ground_error_p95=percentile(ground_p95_values, 0.95),
        ground_error_p95_sample_count=len(ground_p95_values),
        manual_calibration=manual_calibration,
    )
