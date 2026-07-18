from __future__ import annotations

"""Typed measurements produced by reconstruction QA evidence collection.

The classes in this module deliberately contain no presentation or gate
policy.  They are the boundary between reading a scene document and deciding
how measurements are exposed or classified.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CalibrationMeasurements:
    evidence_count: int
    accepted_count: int
    direct_count: int
    temporal_count: int
    reconstruction_frame_count: int
    calibrated_frame_count: int
    coverage: float | None
    direct_coverage: float | None
    temporal_coverage: float | None
    coverage_source: str
    temporal_uncertainty_p95: float | None
    temporal_uncertainty_sample_count: int
    temporal_ambiguity_count: int
    max_gap_seconds: float | None
    residual_p50: float | None
    residual_p50_sample_count: int
    residual_p95: float | None
    residual_p95_sample_count: int
    representative_error: float | None
    inlier_ratio_p10: float | None
    inlier_ratio_sample_count: int
    alignment_f1_p10: float | None
    alignment_f1_sample_count: int
    visible_side_agreement: float | None
    visible_side: str | None
    side_votes: dict[str, int]
    side_vote_count: int
    ground_error_p50: float | None
    ground_error_p50_sample_count: int
    ground_error_p95: float | None
    ground_error_p95_sample_count: int
    manual_calibration: bool

    @property
    def has_evidence(self) -> bool:
        return self.evidence_count > 0


@dataclass(frozen=True)
class ProjectionMeasurements:
    fallback_ratio: float | None
    fallback_count: int
    projected_count: int
    fallback_source: str
    clamp_ratio: float | None
    clamp_count: int
    position_count: int
    clamp_source: str


@dataclass(frozen=True)
class SpeedMeasurements:
    ratio: float | None
    violations: int
    segment_count: int
    p95_metres_per_second: float | None
    maximum_metres_per_second: float | None
    violating_track_count: int
    source: str
    published_ratio: float | None = None
    published_segment_count: int | None = None


@dataclass(frozen=True)
class ContinuityMeasurements:
    median_completeness: float | None
    fragmented_track_ratio: float | None
    fragment_count: int
    track_count: int
    sample_cadence_seconds: float | None
    gap_threshold_seconds: float | None


@dataclass(frozen=True)
class MotionMeasurements:
    player_speed: SpeedMeasurements
    ball_speed: SpeedMeasurements
    continuity: ContinuityMeasurements
    player_track_count: int


@dataclass(frozen=True)
class BallTrackingMeasurements:
    available: bool
    observed_coverage: float | None
    published_coverage: float | None
    frame_count: int
    observed_frame_count: int
    inferred_frame_count: int
    occluded_frame_count: int
    gap_count: int
    longest_gap_seconds: float | None
    path_cost_margin: float | None


@dataclass(frozen=True)
class IdentityMeasurements:
    validation: dict[str, Any]


@dataclass(frozen=True)
class ReconstructionQualityMeasurements:
    processing_status: str
    calibration: CalibrationMeasurements
    projection: ProjectionMeasurements
    motion: MotionMeasurements
    ball_tracking: BallTrackingMeasurements
    identity: IdentityMeasurements


@dataclass(frozen=True)
class QualityGateAssessment:
    verdict: Literal["pass", "review", "reject"]
    summary: dict[str, Any]
    gates: tuple[dict[str, Any], ...]
