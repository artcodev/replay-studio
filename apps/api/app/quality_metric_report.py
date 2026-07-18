from __future__ import annotations

"""Serialize typed reconstruction measurements into the public metric contract."""

from math import isfinite
from typing import Any

from .quality_measurement_domain import ReconstructionQualityMeasurements
from .quality_policy import QualityThresholds


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


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


def build_quality_metrics(
    measurements: ReconstructionQualityMeasurements,
    thresholds: QualityThresholds,
) -> dict[str, dict[str, Any]]:
    calibration = measurements.calibration
    projection = measurements.projection
    motion = measurements.motion
    player_speed = motion.player_speed
    ball_speed = motion.ball_speed
    continuity = motion.continuity
    ball = measurements.ball_tracking
    identity = measurements.identity.validation
    evidence_source = "frame-evidence" if calibration.has_evidence else "missing"
    ball_source = "ball-temporal-resolver" if ball.available else "missing"

    return {
        "calibrationCoverage": _metric(
            calibration.coverage,
            "ratio",
            source=calibration.coverage_source,
            sample_count=(
                calibration.evidence_count or calibration.reconstruction_frame_count
            ),
            acceptedFrames=(
                calibration.accepted_count
                if calibration.has_evidence
                else calibration.calibrated_frame_count
            ),
        ),
        "directCalibrationCoverage": _metric(
            calibration.direct_coverage,
            "ratio",
            source=evidence_source,
            sample_count=(calibration.evidence_count if calibration.has_evidence else None),
        ),
        "temporalCalibrationCoverage": _metric(
            calibration.temporal_coverage,
            "ratio",
            source=evidence_source,
            sample_count=calibration.temporal_count,
            recoveredFrames=calibration.temporal_count,
        ),
        "temporalCalibrationUncertaintyP95": _metric(
            calibration.temporal_uncertainty_p95,
            "metres",
            source=(
                "temporal-hypothesis-graph"
                if calibration.temporal_uncertainty_sample_count
                else "missing"
            ),
            sample_count=calibration.temporal_uncertainty_sample_count,
        ),
        "temporalCalibrationAmbiguityRatio": _metric(
            (
                calibration.temporal_ambiguity_count / calibration.evidence_count
                if calibration.has_evidence
                else None
            ),
            "ratio",
            source=evidence_source,
            sample_count=calibration.evidence_count,
            ambiguousFrames=calibration.temporal_ambiguity_count,
        ),
        "maxCalibrationGap": _metric(
            calibration.max_gap_seconds,
            "seconds",
            source=evidence_source,
            sample_count=(calibration.evidence_count if calibration.has_evidence else None),
        ),
        "calibrationResidualP50": _metric(
            calibration.residual_p50,
            "pixels",
            source=(
                "frame-evidence" if calibration.residual_p50_sample_count else "missing"
            ),
            sample_count=calibration.residual_p50_sample_count,
        ),
        "calibrationResidualP95": _metric(
            calibration.residual_p95,
            "pixels",
            source=(
                "frame-evidence" if calibration.residual_p95_sample_count else "missing"
            ),
            sample_count=calibration.residual_p95_sample_count,
        ),
        "representativeReprojectionError": _metric(
            calibration.representative_error,
            "pixels",
            source=(
                "representative-calibration"
                if calibration.representative_error is not None
                else "missing"
            ),
            sample_count=1 if calibration.representative_error is not None else 0,
        ),
        "calibrationInlierRatioP10": _metric(
            calibration.inlier_ratio_p10,
            "ratio",
            source=(
                "frame-evidence" if calibration.inlier_ratio_sample_count else "missing"
            ),
            sample_count=calibration.inlier_ratio_sample_count,
        ),
        "semanticAlignmentF1P10": _metric(
            calibration.alignment_f1_p10,
            "ratio",
            source=(
                "frame-evidence" if calibration.alignment_f1_sample_count else "missing"
            ),
            sample_count=calibration.alignment_f1_sample_count,
        ),
        "visiblePitchSideAgreement": _metric(
            calibration.visible_side_agreement,
            "ratio",
            source=("frame-evidence" if calibration.side_vote_count else "missing"),
            sample_count=calibration.side_vote_count,
            visiblePitchSide=calibration.visible_side,
            sideVotes=calibration.side_votes,
        ),
        "groundErrorP50": _metric(
            calibration.ground_error_p50,
            "metres",
            source=(
                "frame-evidence"
                if calibration.ground_error_p50_sample_count
                else "missing"
            ),
            sample_count=calibration.ground_error_p50_sample_count,
        ),
        "groundErrorP95": _metric(
            calibration.ground_error_p95,
            "metres",
            source=(
                "frame-evidence"
                if calibration.ground_error_p95_sample_count
                else "missing"
            ),
            sample_count=calibration.ground_error_p95_sample_count,
        ),
        "projectionFallbackRatio": _metric(
            projection.fallback_ratio,
            "ratio",
            source=projection.fallback_source,
            sample_count=projection.projected_count,
            fallbackObservations=projection.fallback_count,
        ),
        "boundaryClampRatio": _metric(
            projection.clamp_ratio,
            "ratio",
            source=projection.clamp_source,
            sample_count=projection.position_count,
            clampedOrBoundaryObservations=projection.clamp_count,
            note=(
                "Exact boundary contacts are a conservative proxy when explicit clamp provenance is unavailable."
                if projection.clamp_source == "boundary-contact-inference"
                else None
            ),
        ),
        "playerSpeedViolationRatio": _metric(
            player_speed.ratio,
            "ratio",
            source=player_speed.source,
            sample_count=player_speed.segment_count,
            limitMetresPerSecond=thresholds.player_speed_limit_mps,
            violations=player_speed.violations,
            p95MetresPerSecond=_round(player_speed.p95_metres_per_second),
            maxMetresPerSecond=_round(player_speed.maximum_metres_per_second),
            violatingTrackCount=player_speed.violating_track_count,
            publishedRatio=_round(player_speed.published_ratio),
            publishedSampleCount=player_speed.published_segment_count,
        ),
        "ballSpeedViolationRatio": _metric(
            ball_speed.ratio,
            "ratio",
            source="trajectory" if ball_speed.segment_count else "missing",
            sample_count=ball_speed.segment_count,
            limitMetresPerSecond=thresholds.ball_speed_limit_mps,
            violations=ball_speed.violations,
            p95MetresPerSecond=_round(ball_speed.p95_metres_per_second),
            maxMetresPerSecond=_round(ball_speed.maximum_metres_per_second),
        ),
        "ballObservedCoverage": _metric(
            ball.observed_coverage,
            "ratio",
            source=ball_source,
            sample_count=ball.frame_count,
            observedFrames=ball.observed_frame_count,
            occludedFrames=ball.occluded_frame_count,
        ),
        "ballPublishedCoverage": _metric(
            ball.published_coverage,
            "ratio",
            source=ball_source,
            sample_count=ball.frame_count,
            inferredFrames=ball.inferred_frame_count,
        ),
        "ballLongestUnresolvedGap": _metric(
            ball.longest_gap_seconds,
            "seconds",
            source=ball_source,
            sample_count=ball.gap_count,
        ),
        "ballPathCostMargin": _metric(
            ball.path_cost_margin,
            "cost",
            source=ball_source,
            sample_count=1 if ball.path_cost_margin is not None else 0,
            note="A larger best-vs-runner-up margin indicates a less ambiguous global trajectory.",
        ),
        "trackContinuity": _metric(
            continuity.median_completeness,
            "ratio",
            source=("trajectory" if continuity.median_completeness is not None else "missing"),
            sample_count=continuity.track_count,
            sampleCadenceSeconds=_round(continuity.sample_cadence_seconds),
            gapThresholdSeconds=_round(continuity.gap_threshold_seconds),
        ),
        "trackFragmentationRatio": _metric(
            continuity.fragmented_track_ratio,
            "ratio",
            source=(
                "trajectory" if continuity.fragmented_track_ratio is not None else "missing"
            ),
            sample_count=continuity.track_count,
            fragmentCount=continuity.fragment_count,
        ),
        "identityIdf1": _metric(
            _number(identity.get("idf1")),
            "ratio",
            source=(
                "labelled-identity-assignments"
                if identity.get("groundTruthAvailable")
                else "ground-truth-unavailable"
            ),
            sample_count=int(identity.get("sampleCount") or 0),
            idPrecision=identity.get("idPrecision"),
            idRecall=identity.get("idRecall"),
            idSwitchCount=identity.get("idSwitchCount"),
            duplicateOverlapSeconds=identity.get("duplicateOverlapSeconds"),
            note=(
                None
                if identity.get("groundTruthAvailable")
                else "Identity accuracy cannot be inferred from observation coverage."
            ),
        ),
    }


__all__ = ["build_quality_metrics"]
