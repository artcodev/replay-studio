from __future__ import annotations

"""Project image points into the scene pitch coordinate system."""

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_calibration_policy import METRIC_CALIBRATION_THRESHOLD
from .reconstruction_metric_projection import project_metric_point


def project_pitch_point_unclamped(
    point_x: float,
    point_y: float,
    width: int,
    height: int,
    pitch: dict,
    calibration: PitchCalibration | None = None,
) -> tuple[float, float]:
    if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD:
        projected = calibration.image_to_pitch @ np.array([point_x, point_y, 1.0])
        if abs(float(projected[2])) > 1e-8:
            x, z = float(projected[0] / projected[2]), float(projected[1] / projected[2])
        else:
            x, z = 0.0, 0.0
    elif calibration is not None and calibration.rectangle in {
        "penalty-area-left",
        "penalty-area-right",
    }:
        progress = point_x / max(1.0, width)
        half_length = float(pitch["length"]) / 2
        x = (
            progress * half_length
            if calibration.rectangle.endswith("right")
            else -half_length + progress * half_length
        )
        z = (point_y / height - 0.5) * float(pitch["width"]) * 1.05
    else:
        x = (point_x / width - 0.5) * float(pitch["length"]) * 0.96
        z = (point_y / height - 0.5) * float(pitch["width"]) * 1.05
    return x, z


def project_pitch_point(
    point_x: float,
    point_y: float,
    width: int,
    height: int,
    pitch: dict,
    calibration: PitchCalibration | None = None,
) -> tuple[float, float]:
    if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD:
        metric = project_metric_point(point_x, point_y, calibration, pitch)
        if metric is not None:
            return metric
        # A trusted matrix can still be invalid for a distant frame after a cut,
        # zoom, or failed optical-flow transform. Do not pile every observation
        # onto a pitch corner by clamping an arbitrarily large projection.
        x, z = project_pitch_point_unclamped(
            point_x,
            point_y,
            width,
            height,
            pitch,
            None,
        )
    else:
        x, z = project_pitch_point_unclamped(
            point_x,
            point_y,
            width,
            height,
            pitch,
            calibration,
        )
    return (
        max(-float(pitch["length"]) / 2, min(float(pitch["length"]) / 2, x)),
        max(-float(pitch["width"]) / 2, min(float(pitch["width"]) / 2, z)),
    )


__all__ = ["project_pitch_point", "project_pitch_point_unclamped"]
