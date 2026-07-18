from __future__ import annotations

from math import pi

import numpy as np

from .pitch_calibration_contract import PitchCalibration


PITCH_LINES = [
    ("touch-top", (-52.5, -34.0), (52.5, -34.0)),
    ("touch-bottom", (-52.5, 34.0), (52.5, 34.0)),
    ("goal-left", (-52.5, -34.0), (-52.5, 34.0)),
    ("goal-right", (52.5, -34.0), (52.5, 34.0)),
    ("halfway", (0.0, -34.0), (0.0, 34.0)),
    ("penalty-left-main", (-36.0, -20.16), (-36.0, 20.16)),
    ("penalty-left-top", (-52.5, -20.16), (-36.0, -20.16)),
    ("penalty-left-bottom", (-52.5, 20.16), (-36.0, 20.16)),
    ("penalty-right-main", (36.0, -20.16), (36.0, 20.16)),
    ("penalty-right-top", (36.0, -20.16), (52.5, -20.16)),
    ("penalty-right-bottom", (36.0, 20.16), (52.5, 20.16)),
    ("goal-area-left-main", (-47.0, -9.16), (-47.0, 9.16)),
    ("goal-area-left-top", (-52.5, -9.16), (-47.0, -9.16)),
    ("goal-area-left-bottom", (-52.5, 9.16), (-47.0, 9.16)),
    ("goal-area-right-main", (47.0, -9.16), (47.0, 9.16)),
    ("goal-area-right-top", (47.0, -9.16), (52.5, -9.16)),
    ("goal-area-right-bottom", (47.0, 9.16), (52.5, 9.16)),
]


PNLCALIB_LINE_TO_PITCH_LINE = {
    "Big rect. left bottom": "penalty-left-bottom",
    "Big rect. left main": "penalty-left-main",
    "Big rect. left top": "penalty-left-top",
    "Big rect. right bottom": "penalty-right-bottom",
    "Big rect. right main": "penalty-right-main",
    "Big rect. right top": "penalty-right-top",
    "Middle line": "halfway",
    "Side line bottom": "touch-bottom",
    "Side line left": "goal-left",
    "Side line right": "goal-right",
    "Side line top": "touch-top",
    "Small rect. left bottom": "goal-area-left-bottom",
    "Small rect. left main": "goal-area-left-main",
    "Small rect. left top": "goal-area-left-top",
    "Small rect. right bottom": "goal-area-right-bottom",
    "Small rect. right main": "goal-area-right-main",
    "Small rect. right top": "goal-area-right-top",
}


def _curve_points(
    center_x: float,
    center_z: float,
    radius: float,
    side: str | None = None,
) -> np.ndarray:
    angles = np.linspace(0.0, pi * 2.0, 180)
    points = np.column_stack(
        [center_x + np.cos(angles) * radius, center_z + np.sin(angles) * radius]
    )
    if side == "left":
        points = points[points[:, 0] >= -36.0]
    elif side == "right":
        points = points[points[:, 0] <= 36.0]
    return points


PITCH_CURVES = [
    ("center-circle", _curve_points(0.0, 0.0, 9.15)),
    ("penalty-arc-left", _curve_points(-41.5, 0.0, 9.15, "left")),
    ("penalty-arc-right", _curve_points(41.5, 0.0, 9.15, "right")),
]


RECTANGLES = [
    ("penalty-area-right", (36.0, 52.5), (-20.16, 20.16), "penalty-arc-right"),
    ("penalty-area-left", (-52.5, -36.0), (-20.16, 20.16), "penalty-arc-left"),
]


ANCHOR_PRESETS: dict[str, list[tuple[str, str, tuple[float, float]]]] = {
    "penalty-area-right": [
        ("front-far", "Penalty front · far", (36.0, -20.16)),
        ("front-near", "Penalty front · near", (36.0, 20.16)),
        ("goal-far", "Goal line · far", (52.5, -20.16)),
        ("goal-near", "Goal line · near", (52.5, 20.16)),
    ],
    "goal-area-right": [
        ("front-far", "Goal area front · far", (47.0, -9.16)),
        ("front-near", "Goal area front · near", (47.0, 9.16)),
        ("goal-far", "Goal line · far", (52.5, -9.16)),
        ("goal-near", "Goal line · near", (52.5, 9.16)),
    ],
    "penalty-area-left": [
        ("goal-far", "Goal line · far", (-52.5, -20.16)),
        ("goal-near", "Goal line · near", (-52.5, 20.16)),
        ("front-far", "Penalty front · far", (-36.0, -20.16)),
        ("front-near", "Penalty front · near", (-36.0, 20.16)),
    ],
    "goal-area-left": [
        ("goal-far", "Goal line · far", (-52.5, -9.16)),
        ("goal-near", "Goal line · near", (-52.5, 9.16)),
        ("front-far", "Goal area front · far", (-47.0, -9.16)),
        ("front-near", "Goal area front · near", (-47.0, 9.16)),
    ],
    "center-circle": [
        ("circle-left", "Circle · left", (-9.15, 0.0)),
        ("circle-top", "Circle · far", (0.0, -9.15)),
        ("circle-right", "Circle · right", (9.15, 0.0)),
        ("circle-bottom", "Circle · near", (0.0, 9.15)),
    ],
}


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = source @ homography.T
    valid = np.abs(projected[:, 2]) > 1e-8
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = projected[valid, :2] / projected[valid, 2:3]
    return output


def projected_pitch_markings(
    calibration: PitchCalibration,
    width: int,
    height: int,
) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return []
    sources: list[tuple[str, str, np.ndarray]] = []
    for name, start, end in PITCH_LINES:
        alpha = np.linspace(0.0, 1.0, 90)
        points = np.column_stack(
            [
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
            ]
        )
        sources.append((name, "line", points))
    sources.extend((name, "curve", points) for name, points in PITCH_CURVES)

    markings = []
    for name, kind, pitch_points in sources:
        image_points = project_points(pitch_points, pitch_to_image)
        valid = (
            np.isfinite(image_points).all(axis=1)
            & (image_points[:, 0] > -width * 0.2)
            & (image_points[:, 0] < width * 1.2)
            & (image_points[:, 1] > -height * 0.2)
            & (image_points[:, 1] < height * 1.2)
        )
        visible = image_points[valid]
        if len(visible) < (8 if kind == "curve" else 2):
            continue
        markings.append(
            {
                "id": name,
                "kind": kind,
                "points": [
                    {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}
                    for point in visible
                ],
            }
        )
    return markings


def calibration_horizon(
    calibration: PitchCalibration,
    image_width: int,
) -> dict | None:
    """Return the image-space ground-plane horizon for calibration QA."""
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return None
    line = np.cross(pitch_to_image[:, 0], pitch_to_image[:, 1])
    if not np.isfinite(line).all() or abs(float(line[1])) < 1e-8:
        return None
    x1 = 0.0
    x2 = float(max(1, image_width - 1))
    y1 = float((-line[0] * x1 - line[2]) / line[1])
    y2 = float((-line[0] * x2 - line[2]) / line[1])
    if not np.isfinite([y1, y2]).all():
        return None
    return {
        "start": {"x": x1, "y": round(y1, 3)},
        "end": {"x": x2, "y": round(y2, 3)},
    }
