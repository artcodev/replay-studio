from __future__ import annotations

"""Camera motion estimation and image-space stabilization."""

from math import exp

import cv2
import numpy as np

from .reconstruction_person_detection_contract import Detection
from .camera_motion_contract import CameraMotionEstimate

def scene_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    previous_hist = cv2.calcHist([previous_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    current_hist = cv2.calcHist([current_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(previous_hist, previous_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA))


def unreliable_motion(
    reason: str,
    scene_change_score: float,
    *,
    tracked_count: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    residual_p50: float | None = None,
    residual_p95: float | None = None,
    forward_backward_p95: float | None = None,
    coverage_ratio: float = 0.0,
) -> CameraMotionEstimate:
    cut = scene_change_score > 0.18 and (
        tracked_count < 12
        or inlier_ratio < 0.20
        or (forward_backward_p95 is not None and forward_backward_p95 > 8.0)
    )
    return CameraMotionEstimate(
        matrix=np.eye(3, dtype=np.float64),
        status="cut" if cut else "unreliable",
        confidence=0.0,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        residual_p50=residual_p50,
        residual_p95=residual_p95,
        forward_backward_p95=forward_backward_p95,
        coverage_ratio=coverage_ratio,
        scene_change_score=scene_change_score,
        reason=reason,
    )


def camera_motion_estimate(previous: np.ndarray, current: np.ndarray) -> CameraMotionEstimate:
    """Estimate a QA-scored projective transform from current to previous.

    A successful static-camera estimate remains ``estimated`` even when its
    matrix is nearly identity. Failed flow and shot cuts are explicit graph
    barriers, so temporal calibration can never silently cross them.
    """

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    scene_change = scene_change_score(previous, current)
    field_mask = cv2.inRange(previous_hsv, np.array([25, 35, 25]), np.array([100, 255, 255]))
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.012,
        minDistance=7,
        mask=field_mask,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return unreliable_motion(
            "insufficient-pitch-features",
            scene_change,
            tracked_count=0 if points is None else len(points),
        )
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    if moved is None or status is None:
        return unreliable_motion("forward-optical-flow-failed", scene_change)
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        moved,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    if returned is None or backward_status is None:
        return unreliable_motion("backward-optical-flow-failed", scene_change)
    forward_valid = status.ravel() == 1
    backward_valid = backward_status.ravel() == 1
    forward_backward_error = np.linalg.norm(
        returned.reshape(-1, 2) - points.reshape(-1, 2),
        axis=1,
    )
    flow_candidates = (
        forward_valid & backward_valid & np.isfinite(forward_backward_error)
    )
    valid = flow_candidates & (forward_backward_error <= 2.5)
    previous_points = points.reshape(-1, 2)[valid]
    current_points = moved.reshape(-1, 2)[valid]
    tracked_count = len(previous_points)
    fb_p95 = (
        float(np.percentile(forward_backward_error[valid], 95))
        if int(valid.sum())
        else None
    )
    if tracked_count < 16:
        return unreliable_motion(
            "insufficient-forward-backward-tracks",
            scene_change,
            tracked_count=tracked_count,
            forward_backward_p95=fb_p95,
        )
    height, width = previous_gray.shape
    coverage = float(cv2.contourArea(cv2.convexHull(previous_points.astype(np.float32)))) / max(
        1.0, float(width * height)
    )
    matrix, inliers = cv2.findHomography(
        current_points,
        previous_points,
        cv2.RANSAC,
        2.5,
        maxIters=2000,
        confidence=0.995,
    )
    if (
        matrix is None
        or inliers is None
        or not np.isfinite(matrix).all()
        or abs(float(matrix[2, 2])) < 1e-10
        or abs(float(np.linalg.det(matrix))) < 1e-10
    ):
        return unreliable_motion(
            "projective-motion-fit-failed",
            scene_change,
            tracked_count=tracked_count,
            forward_backward_p95=fb_p95,
            coverage_ratio=coverage,
        )
    matrix /= matrix[2, 2]
    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(1, tracked_count)
    projected = cv2.perspectiveTransform(
        current_points[inlier_mask].reshape(-1, 1, 2).astype(np.float32),
        matrix,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(projected - previous_points[inlier_mask], axis=1)
    residual_p50 = float(np.percentile(residuals, 50)) if len(residuals) else None
    residual_p95 = float(np.percentile(residuals, 95)) if len(residuals) else None
    corners = np.float32(
        [[[0.0, 0.0]], [[width, 0.0]], [[width, height]], [[0.0, height]]]
    )
    warped_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
    area_ratio = warped_area / max(1.0, float(width * height))
    plausible_corners = bool(
        np.isfinite(warped_corners).all()
        and np.all(warped_corners[:, 0] > -width * 1.5)
        and np.all(warped_corners[:, 0] < width * 2.5)
        and np.all(warped_corners[:, 1] > -height * 1.5)
        and np.all(warped_corners[:, 1] < height * 2.5)
        and 0.40 <= area_ratio <= 2.50
    )
    rejection = None
    if inlier_count < 16:
        rejection = "insufficient-projective-inliers"
    elif inlier_ratio < 0.52:
        rejection = "projective-inlier-ratio-too-low"
    elif coverage < 0.02:
        rejection = "motion-features-too-concentrated"
    elif residual_p95 is None or residual_p95 > 3.5:
        rejection = "projective-motion-residual-too-high"
    elif fb_p95 is None:
        rejection = "forward-backward-flow-error-too-high"
    elif not plausible_corners:
        rejection = "implausible-projective-frame-warp"
    if rejection:
        return unreliable_motion(
            rejection,
            scene_change,
            tracked_count=tracked_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            residual_p50=residual_p50,
            residual_p95=residual_p95,
            forward_backward_p95=fb_p95,
            coverage_ratio=coverage,
        )

    confidence = (
        0.30 * min(1.0, inlier_ratio / 0.80)
        + 0.20 * min(1.0, inlier_count / 60.0)
        + 0.20 * exp(-float(residual_p95) / 2.5)
        + 0.20 * exp(-float(fb_p95) / 1.5)
        + 0.10 * min(1.0, coverage / 0.12)
    )
    return CameraMotionEstimate(
        matrix=matrix,
        status="estimated",
        confidence=max(0.0, min(0.99, confidence)),
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        residual_p50=residual_p50,
        residual_p95=residual_p95,
        forward_backward_p95=fb_p95,
        coverage_ratio=coverage,
        scene_change_score=scene_change,
    )


def camera_step(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    estimate = camera_motion_estimate(previous, current)
    return estimate.matrix if estimate.reliable else np.eye(3, dtype=np.float64)


def stabilize_point(x: float, y: float, transform: np.ndarray) -> tuple[float, float]:
    projected = transform @ np.array([x, y, 1.0], dtype=np.float64)
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def stabilize_detections(
    detections: list[Detection],
    balls: list[dict],
    transform: np.ndarray,
) -> None:
    for detection in detections:
        detection.x, detection.y = stabilize_point(detection.x, detection.y, transform)
    for ball in balls:
        # Preserve detector-space coordinates for video overlays and expose the
        # compensated position separately for temporal association.  Mutating
        # x/y used to make the saved candidate impossible to audit against the
        # source frame.
        stable_x, stable_y = stabilize_point(ball["x"], ball["y"], transform)
        ball["stabilizedX"] = stable_x
        ball["stabilizedY"] = stable_y

