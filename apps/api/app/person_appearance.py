"""Pure image appearance features and broadcast-pitch person filtering."""

from __future__ import annotations

import cv2
import numpy as np

from .person_detection_policy import (
    MINIMUM_PERSON_FOOT_Y,
    SHALLOW_PERSON_CONFIDENCE,
    SHALLOW_PERSON_FOOT_Y,
    SHALLOW_PERSON_GRASS_RATIO,
)


def green_ratio(
    hsv: np.ndarray,
    x: float,
    y: float,
    radius_x: int,
    radius_y: int,
) -> float:
    height, width = hsv.shape[:2]
    x1, x2 = max(0, int(x) - radius_x), min(width, int(x) + radius_x)
    y1, y2 = max(0, int(y) - radius_y), min(height, int(y) + radius_y)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    green = (
        (patch[:, :, 0] > 25)
        & (patch[:, :, 0] < 100)
        & (patch[:, :, 1] > 35)
        & (patch[:, :, 2] > 25)
    )
    return float(green.mean())


def appearance_feature(
    image: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    crop = image[
        max(0, int(y1 + height * 0.12)):max(1, int(y1 + height * 0.62)),
        max(0, int(x1 + width * 0.18)):max(1, int(x2 - width * 0.18)),
    ]
    if crop.size == 0:
        return np.zeros(12, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    vivid = pixels[(pixels[:, 1] > 55) & (pixels[:, 2] > 45)]
    histogram = np.zeros(8, dtype=np.float32)
    if len(vivid):
        histogram, _ = np.histogram(vivid[:, 0], bins=8, range=(0, 180))
        histogram = histogram.astype(np.float32)
        histogram /= max(1.0, float(histogram.sum()))
    white_ratio = float(((pixels[:, 1] < 55) & (pixels[:, 2] > 135)).mean())
    dark_ratio = float((pixels[:, 2] < 72).mean())
    mean_saturation = float(pixels[:, 1].mean() / 255.0)
    mean_value = float(pixels[:, 2].mean() / 255.0)
    return np.concatenate(
        [
            histogram,
            np.array(
                [white_ratio, dark_ratio, mean_saturation, mean_value],
                dtype=np.float32,
            ),
        ]
    )


def is_pitch_person(
    hsv: np.ndarray,
    box: tuple[float, float, float, float],
    confidence: float,
) -> bool:
    """Reject spectators and graphics without losing small far-side players."""

    x1, y1, x2, y2 = box
    height, _ = hsv.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    if box_height < 14 or box_width < 5 or box_height < box_width * 1.05:
        return False
    if y2 < height * MINIMUM_PERSON_FOOT_Y:
        return False
    center_x = (x1 + x2) / 2
    pitch_ratio = green_ratio(
        hsv,
        center_x,
        y2 + min(4.0, box_height * 0.06),
        max(8, int(box_width)),
        max(6, int(box_height * 0.16)),
    )
    if pitch_ratio < 0.38:
        # Strong detector evidence may survive painted lines and crowded boxes.
        if not (
            confidence >= 0.55
            and y2 >= height * 0.24
            and pitch_ratio >= 0.15
        ):
            return False
    if y2 < height * SHALLOW_PERSON_FOOT_Y:
        return confidence >= SHALLOW_PERSON_CONFIDENCE and (
            pitch_ratio >= SHALLOW_PERSON_GRASS_RATIO or confidence >= 0.55
        )
    return True
