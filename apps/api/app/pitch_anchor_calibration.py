from __future__ import annotations

import cv2
import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_geometry import ANCHOR_PRESETS, project_points


def calibration_from_anchors(
    anchors: list[dict],
    preset: str,
    confidence: float = 0.9,
) -> PitchCalibration:
    if preset not in ANCHOR_PRESETS:
        raise ValueError("Unsupported pitch anchor preset")
    if len(anchors) < 4:
        raise ValueError("At least four pitch anchors are required")
    image_points = np.float64(
        [
            [float(anchor["image"]["x"]), float(anchor["image"]["y"])]
            for anchor in anchors
        ]
    )
    pitch_points = np.float64(
        [
            [float(anchor["pitch"]["x"]), float(anchor["pitch"]["z"])]
            for anchor in anchors
        ]
    )
    if cv2.contourArea(cv2.convexHull(np.float32(image_points))) < 80.0:
        raise ValueError("Pitch anchors are too close together")
    if cv2.contourArea(cv2.convexHull(np.float32(pitch_points))) < 8.0:
        raise ValueError("Pitch anchors do not cover a stable area of the pitch")
    method = cv2.RANSAC if len(anchors) > 4 else 0
    homography, mask = cv2.findHomography(image_points, pitch_points, method, 2.5)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise ValueError("Pitch anchors do not define a valid plane")
    determinant = float(np.linalg.det(homography))
    if abs(determinant) < 1e-10:
        raise ValueError("Pitch anchors produce a singular projection")
    homography /= homography[2, 2]
    try:
        pitch_to_image = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Pitch anchors produce a singular inverse projection") from exc
    image_reprojected = project_points(pitch_points, pitch_to_image)
    image_residuals = np.linalg.norm(image_reprojected - image_points, axis=1)
    if not np.isfinite(image_residuals).all():
        raise ValueError("Pitch anchors produce non-finite reprojection residuals")
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 4:
        raise ValueError("Pitch anchors do not contain four geometric inliers")
    return PitchCalibration(
        image_to_pitch=homography,
        confidence=max(0.0, min(1.0, confidence)),
        supported_lines=len(anchors),
        mean_line_score=0.0,
        rectangle=preset,
        matched_curves=1 if preset == "center-circle" else 0,
        keypoint_count=len(anchors),
        inlier_count=inlier_count,
        reprojection_error=float(np.median(image_residuals[inlier_mask])),
    )
