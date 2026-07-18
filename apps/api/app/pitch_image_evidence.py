from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_geometry import projected_pitch_markings


@dataclass(frozen=True)
class AlignmentResiduals:
    model_to_observed: np.ndarray
    observed_to_model: np.ndarray
    model_sample_count: int
    observed_sample_count: int


def pitch_line_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 100, 255]))
    green = cv2.inRange(hsv, np.array([25, 25, 20]), np.array([105, 255, 255]))
    connected_green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected_green, 8
    )
    pitch_green = np.zeros_like(green)
    if component_count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch_green[labels == largest] = 255
    near_pitch = cv2.dilate(pitch_green, np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(white, near_pitch)
    mask[: int(image.shape[0] * 0.18)] = 0
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def alignment_residuals(
    image: np.ndarray,
    calibration: PitchCalibration,
) -> AlignmentResiduals | None:
    observed_mask = pitch_line_mask(image)
    height, width = observed_mask.shape
    model_mask = np.zeros_like(observed_mask)
    for marking in projected_pitch_markings(calibration, width, height):
        points = np.float32(
            [[point["x"], point["y"]] for point in marking["points"]]
        )
        inside = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= height * 0.16)
            & (points[:, 1] < height)
        )
        visible = points[inside].round().astype(np.int32)
        if len(visible) < (8 if marking["kind"] == "curve" else 2):
            continue
        cv2.polylines(model_mask, [visible], False, 255, 1, cv2.LINE_AA)

    # distanceTransform measures distance to zero pixels. Normalize the
    # anti-aliased raster first so every rendered model pixel becomes a target.
    model_mask = np.where(model_mask > 0, 255, 0).astype(np.uint8)
    observed_mask = np.where(observed_mask > 0, 255, 0).astype(np.uint8)
    model_pixels = model_mask > 0
    observed_pixels = observed_mask > 0
    model_count = int(model_pixels.sum())
    observed_count = int(observed_pixels.sum())
    if model_count < 20 or observed_count < 20:
        return None

    distance_to_observed = cv2.distanceTransform(
        cv2.bitwise_not(observed_mask), cv2.DIST_L2, 3
    )
    distance_to_model = cv2.distanceTransform(
        cv2.bitwise_not(model_mask), cv2.DIST_L2, 3
    )
    return AlignmentResiduals(
        model_to_observed=np.clip(distance_to_observed[model_pixels], 0.0, 80.0),
        observed_to_model=np.clip(distance_to_model[observed_pixels], 0.0, 80.0),
        model_sample_count=model_count,
        observed_sample_count=observed_count,
    )
