from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .pitch_calibration_contract import PitchCalibration
from .temporal_calibration_contract import CalibrationHypothesis, TemporalCalibrationFrame
from .temporal_homography import motion_path, normalize_homography


def anchor_uncertainty(calibration: PitchCalibration) -> float:
    error = calibration.reprojection_p95
    if error is None:
        error = calibration.reprojection_error
    if error is not None and np.isfinite(error):
        return max(0.25, min(4.0, float(error) * 0.18))
    return max(0.5, min(4.0, (1.0 - float(calibration.confidence)) * 5.0))


def make_temporal_hypothesis(
    target: TemporalCalibrationFrame,
    anchor: TemporalCalibrationFrame,
    anchor_calibration: PitchCalibration,
    motion_edges: dict[int, CameraMotionEstimate],
) -> CalibrationHypothesis | None:
    path = motion_path(motion_edges, target.sample_index, anchor.sample_index)
    if path is None:
        return None
    temporal_distance = abs(float(target.scene_time) - float(anchor.scene_time))
    matrix = normalize_homography(anchor_calibration.image_to_pitch @ path.target_to_anchor)
    direction = "forward" if anchor.sample_index < target.sample_index else "backward"
    anchor_score = max(0.0, min(1.0, float(anchor_calibration.confidence)))
    if anchor_calibration.inlier_ratio is not None:
        anchor_score *= 0.85 + 0.15 * max(
            0.0, min(1.0, float(anchor_calibration.inlier_ratio))
        )
    temporal_decay = exp(-temporal_distance / 7.0)
    score = anchor_score * (0.82 + 0.18 * path.confidence) * temporal_decay
    uncertainty = (
        anchor_uncertainty(anchor_calibration)
        + temporal_distance * 0.38
        + path.residual_sum * 0.035
        + (1.0 - path.confidence) * 3.0
    )
    propagated = replace(
        anchor_calibration,
        image_to_pitch=matrix,
        confidence=max(0.0, min(0.99, score)),
        supported_lines=0,
        mean_line_score=0.0,
        matched_curves=0,
        method=f"temporal-{direction}",
        keypoint_count=0,
        inlier_count=0,
        reprojection_error=None,
        frame_index=target.source_frame_index,
        detected_keypoint_count=0,
        completed_keypoint_count=0,
        inlier_ratio=None,
        reprojection_p95=None,
        raw_line_count=0,
        ground_error_p50=None,
        ground_error_p95=None,
        raw_keypoints=(),
        confidence_kind="heuristic-temporal-hypothesis-score",
    )
    return CalibrationHypothesis(
        id=f"temporal-{direction}-s{anchor.sample_index}-to-s{target.sample_index}",
        target_sample_index=target.sample_index,
        anchor_sample_index=anchor.sample_index,
        anchor_source_frame_index=anchor.source_frame_index,
        anchor_scene_time=anchor.scene_time,
        direction=direction,
        calibration=propagated,
        score=max(0.0, min(0.99, score)),
        uncertainty_metres=max(0.25, uncertainty),
        motion_confidence=path.confidence,
        temporal_distance_seconds=temporal_distance,
        motion_edge_indices=path.edge_indices,
    )
