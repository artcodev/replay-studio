from __future__ import annotations

"""Apply versioned gate policy to typed reconstruction measurements."""

from math import isfinite
from typing import Any

from .quality_measurement_domain import (
    QualityGateAssessment,
    ReconstructionQualityMeasurements,
)
from .quality_policy import QualityThresholds, higher_gate, lower_gate


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def assess_quality_gates(
    measurements: ReconstructionQualityMeasurements,
    thresholds: QualityThresholds,
) -> QualityGateAssessment:
    calibration = measurements.calibration
    projection = measurements.projection
    motion = measurements.motion
    player_speed = motion.player_speed
    ball_speed = motion.ball_speed
    continuity = motion.continuity
    ball = measurements.ball_tracking
    identity = measurements.identity.validation

    coverage = (
        None
        if calibration.manual_calibration and not calibration.has_evidence
        else calibration.coverage
    )
    coverage_note = (
        "Manual calibration requires per-frame validation; one anchor frame does not prove shot-wide coverage."
        if calibration.manual_calibration and not calibration.has_evidence
        else None
    )
    reprojection_evidence = (
        "frame-evidence"
        if calibration.residual_p50_sample_count or calibration.residual_p95_sample_count
        else "representative"
    )
    residual_p50 = (
        calibration.residual_p50
        if calibration.residual_p50 is not None
        else calibration.representative_error
    )
    residual_p95 = (
        calibration.residual_p95
        if calibration.residual_p95 is not None
        else calibration.representative_error
    )
    gates = (
        higher_gate(
            "calibration-coverage",
            "Accepted calibration coverage",
            coverage,
            "ratio",
            thresholds.calibration_coverage_pass,
            thresholds.calibration_coverage_review,
            note=coverage_note,
        ),
        lower_gate(
            "calibration-gap",
            "Longest gap without accepted calibration",
            calibration.max_gap_seconds,
            "seconds",
            thresholds.calibration_gap_pass_seconds,
            thresholds.calibration_gap_review_seconds,
            required=True,
            evidence="frame-evidence",
            note="Unknown when per-frame calibration evidence is unavailable.",
        ),
        lower_gate(
            "temporal-uncertainty",
            "Recovered calibration uncertainty p95",
            calibration.temporal_uncertainty_p95,
            "metres",
            thresholds.temporal_uncertainty_p95_pass_metres,
            thresholds.temporal_uncertainty_p95_review_metres,
            required=calibration.temporal_count > 0,
            evidence=(
                "temporal-hypothesis-graph"
                if calibration.temporal_uncertainty_sample_count
                else "missing"
            ),
            note="Required only when metric frames are recovered from camera-motion hypotheses.",
        ),
        lower_gate(
            "reprojection-p50",
            "Calibration reprojection error p50",
            residual_p50,
            "pixels",
            thresholds.reprojection_p50_pass_px,
            thresholds.reprojection_p50_review_px,
            evidence=reprojection_evidence,
            note=(
                "Uses one representative frame; per-frame residuals are required for a distribution."
                if not calibration.residual_p50_sample_count
                else None
            ),
        ),
        lower_gate(
            "reprojection-p95",
            "Calibration reprojection error p95",
            residual_p95,
            "pixels",
            thresholds.reprojection_p95_pass_px,
            thresholds.reprojection_p95_review_px,
            evidence=reprojection_evidence,
            note=(
                "Uses one representative frame; per-frame residuals are required for a true p95."
                if not calibration.residual_p95_sample_count
                else None
            ),
        ),
        higher_gate(
            "inlier-ratio-p10",
            "Calibration inlier ratio p10",
            calibration.inlier_ratio_p10,
            "ratio",
            thresholds.inlier_ratio_p10_pass,
            thresholds.inlier_ratio_p10_review,
            required=False,
        ),
        higher_gate(
            "semantic-line-alignment",
            "Bidirectional semantic-line F1 p10",
            calibration.alignment_f1_p10,
            "ratio",
            thresholds.semantic_alignment_f1_p10_pass,
            thresholds.semantic_alignment_f1_p10_review,
            required=calibration.has_evidence,
            note="The score must compare both projected model lines to image markings and image markings back to the model.",
        ),
        higher_gate(
            "orientation-stability",
            "Visible pitch-side agreement",
            calibration.visible_side_agreement,
            "ratio",
            thresholds.visible_side_agreement_pass,
            thresholds.visible_side_agreement_review,
            required=calibration.visible_side_agreement is not None,
            note="Visible pitch side is camera evidence and remains independent from team attack direction.",
        ),
        lower_gate(
            "projection-fallback",
            "Screen-space projection fallback",
            projection.fallback_ratio,
            "ratio",
            thresholds.projection_fallback_pass,
            thresholds.projection_fallback_review,
            evidence=projection.fallback_source,
            note="A metric run must not silently mix pitch and screen-relative coordinates.",
        ),
        lower_gate(
            "boundary-clamp",
            "Pitch-boundary clamp/contact ratio",
            projection.clamp_ratio,
            "ratio",
            thresholds.boundary_clamp_pass,
            thresholds.boundary_clamp_review,
            evidence=projection.clamp_source,
        ),
        lower_gate(
            "player-speed",
            f"Player segments above {thresholds.player_speed_limit_mps:g} m/s",
            player_speed.ratio,
            "ratio",
            thresholds.player_speed_violation_pass,
            thresholds.player_speed_violation_review,
            required=motion.player_track_count > 0,
        ),
        higher_gate(
            "identity-idf1",
            "Labelled identity IDF1",
            _number(identity.get("idf1")),
            "ratio",
            thresholds.identity_idf1_pass,
            thresholds.identity_idf1_review,
            required=bool(identity.get("groundTruthAvailable")),
            evidence=(
                "labelled-identity-assignments"
                if identity.get("groundTruthAvailable")
                else "missing"
            ),
            note=(
                None
                if identity.get("groundTruthAvailable")
                else "Ground truth is unavailable; runtime evidence coverage is not an accuracy metric."
            ),
        ),
        lower_gate(
            "ball-speed",
            f"Ball segments above {thresholds.ball_speed_limit_mps:g} m/s",
            ball_speed.ratio,
            "ratio",
            thresholds.ball_speed_violation_pass,
            thresholds.ball_speed_violation_review,
            required=False,
        ),
        higher_gate(
            "ball-observed-coverage",
            "Ball frames supported by detector observations",
            ball.observed_coverage,
            "ratio",
            thresholds.ball_observed_coverage_pass,
            thresholds.ball_observed_coverage_review,
            required=False,
            evidence="ball-temporal-resolver" if ball.available else "missing",
            note="Diagnostic only: a genuinely occluded or out-of-frame ball can lower this value.",
        ),
        higher_gate(
            "ball-published-coverage",
            "Ball frames observed or bounded-interpolated",
            ball.published_coverage,
            "ratio",
            thresholds.ball_published_coverage_pass,
            thresholds.ball_published_coverage_review,
            required=False,
            evidence="ball-temporal-resolver" if ball.available else "missing",
        ),
        higher_gate(
            "track-continuity",
            "Median within-track observation completeness",
            continuity.median_completeness,
            "ratio",
            thresholds.track_continuity_pass,
            thresholds.track_continuity_review,
            required=motion.player_track_count > 0,
        ),
        lower_gate(
            "track-fragmentation",
            "Tracks containing long observation gaps",
            continuity.fragmented_track_ratio,
            "ratio",
            thresholds.track_fragmentation_pass,
            thresholds.track_fragmentation_review,
            required=motion.player_track_count > 0,
        ),
    )

    required = [gate for gate in gates if gate["required"]]
    if any(gate["status"] == "reject" for gate in required):
        verdict = "reject"
    elif any(gate["status"] in {"review", "unknown"} for gate in required):
        verdict = "review"
    else:
        verdict = "pass"
    counts = {
        status: sum(gate["status"] == status for gate in gates)
        for status in ("pass", "review", "reject", "unknown")
    }
    summary = {
        **counts,
        "failedRequiredGates": [
            gate["id"] for gate in required if gate["status"] == "reject"
        ],
        "unknownRequiredGates": [
            gate["id"] for gate in required if gate["status"] == "unknown"
        ],
    }
    return QualityGateAssessment(verdict=verdict, summary=summary, gates=gates)


__all__ = ["assess_quality_gates"]
