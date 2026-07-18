from __future__ import annotations

import cv2
import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_geometry import PITCH_LINES, project_points


def calibration_overlay(
    image: np.ndarray,
    calibration: PitchCalibration,
) -> np.ndarray:
    overlay = image.copy()
    pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    palette = (80, 240, 255)
    for _, start, end in PITCH_LINES:
        alpha = np.linspace(0.0, 1.0, 80)
        pitch_points = np.column_stack(
            [
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
            ]
        )
        image_points = project_points(pitch_points, pitch_to_image)
        valid = np.isfinite(image_points).all(axis=1)
        points = image_points[valid].round().astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, palette, 2, cv2.LINE_AA)
    return cv2.addWeighted(image, 0.72, overlay, 0.58, 0)
