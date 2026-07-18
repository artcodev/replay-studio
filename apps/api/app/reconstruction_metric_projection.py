from __future__ import annotations

"""Metric pitch projection for reconstructed detections and ball samples."""

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_person_detection_contract import Detection


def project_metric_point(
    x: float,
    y: float,
    calibration: PitchCalibration,
    pitch: dict,
) -> tuple[float, float] | None:
    projected = calibration.image_to_pitch @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(float(projected[2])) < 1e-8:
        return None
    pitch_x = float(projected[0] / projected[2])
    pitch_z = float(projected[1] / projected[2])
    half_length = float(pitch["length"]) / 2.0
    half_width = float(pitch["width"]) / 2.0
    if not np.isfinite([pitch_x, pitch_z]).all():
        return None
    if not (-half_length - 4.0 <= pitch_x <= half_length + 4.0):
        return None
    if not (-half_width - 4.0 <= pitch_z <= half_width + 4.0):
        return None
    return (
        max(-half_length, min(half_length, pitch_x)),
        max(-half_width, min(half_width, pitch_z)),
    )


def attach_metric_positions(
    people: list[Detection],
    balls: list[dict],
    calibration: PitchCalibration | None,
    pitch: dict,
    *,
    projection_source: str = "direct",
    calibration_frame_index: int | None = None,
    position_uncertainty_metres: float | None = None,
) -> None:
    if calibration is None:
        return
    for detection in people:
        position = project_metric_point(detection.x, detection.y, calibration, pitch)
        if position is not None:
            detection.pitch_x, detection.pitch_z = position
            detection.projection_source = projection_source
            detection.calibration_frame_index = calibration_frame_index
            detection.position_uncertainty_metres = position_uncertainty_metres
    for ball in balls:
        position = project_metric_point(ball["x"], ball["y"], calibration, pitch)
        if position is not None:
            ball["pitchX"], ball["pitchZ"] = position
            ball["projectionSource"] = projection_source
            ball["calibrationFrameIndex"] = calibration_frame_index
            ball["positionUncertaintyMetres"] = position_uncertainty_metres


def calibration_person_support(
    people: list[Detection],
    calibration: PitchCalibration,
    pitch: dict,
) -> tuple[int, int]:
    if not people:
        return 0, 0
    supported = sum(
        project_metric_point(person.x, person.y, calibration, pitch) is not None
        for person in people
    )
    return supported, len(people)


def calibration_uncertainty_metres(
    calibration: PitchCalibration,
    alignment_error: float | None = None,
) -> float:
    """Return an explicit engineering estimate, not a statistical confidence interval."""
    pixel_error = (
        alignment_error
        if alignment_error is not None
        else calibration.reprojection_error
    )
    if pixel_error is not None:
        return round(max(0.25, min(12.0, float(pixel_error) * 0.25)), 2)
    return round(max(0.75, min(8.0, 1.0 + (1.0 - calibration.confidence) * 8.0)), 2)
