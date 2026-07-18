from __future__ import annotations

import numpy as np

from .pitch_calibration_contract import (
    PitchCalibration,
    opposite_pitch_preset,
    pitch_side,
)
from .pitch_geometry import project_points


def flip_pitch_calibration(calibration: PitchCalibration) -> PitchCalibration:
    pitch_flip = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PitchCalibration(
        image_to_pitch=pitch_flip @ calibration.image_to_pitch,
        confidence=calibration.confidence,
        supported_lines=calibration.supported_lines,
        mean_line_score=calibration.mean_line_score,
        rectangle=opposite_pitch_preset(calibration.rectangle),
        matched_curves=calibration.matched_curves,
        method=calibration.method,
        keypoint_count=calibration.keypoint_count,
        inlier_count=calibration.inlier_count,
        reprojection_error=calibration.reprojection_error,
        frame_index=calibration.frame_index,
        detected_keypoint_count=calibration.detected_keypoint_count,
        completed_keypoint_count=calibration.completed_keypoint_count,
        inlier_ratio=calibration.inlier_ratio,
        reprojection_p95=calibration.reprojection_p95,
        raw_line_count=calibration.raw_line_count,
        ground_error_p50=calibration.ground_error_p50,
        ground_error_p95=calibration.ground_error_p95,
        raw_keypoints=calibration.raw_keypoints,
        raw_lines=calibration.raw_lines,
        confidence_kind=calibration.confidence_kind,
        backend_diagnostics=calibration.backend_diagnostics,
    )


def canonicalize_penalty_side(
    calibration: PitchCalibration,
    image_width: int,
) -> PitchCalibration:
    """Resolve the mirror ambiguity from the fitted landmark's image position."""
    side = pitch_side(calibration.rectangle)
    if side is None or not calibration.rectangle.startswith(
        ("penalty-area-", "goal-area-")
    ):
        return calibration
    center_magnitude = (
        49.75 if calibration.rectangle.startswith("goal-area-") else 44.25
    )
    pitch_center_x = -center_magnitude if side == "left" else center_magnitude
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return calibration
    image_center = project_points(
        np.array([[pitch_center_x, 0.0]]), pitch_to_image
    )[0]
    if not np.isfinite(image_center).all():
        return calibration
    offset = float(image_center[0]) - image_width / 2
    if abs(offset) < image_width * 0.04:
        return calibration
    expected_right = offset > 0
    current_right = side == "right"
    if expected_right == current_right:
        return calibration
    return flip_pitch_calibration(calibration)
