from __future__ import annotations

from math import exp

import cv2
import numpy as np

from .pitch_calibration import PitchCalibration


# Semantic order used by the open Roboflow Sports pitch-keypoint model.  The
# coordinates use the same centred 105 x 68 metre pitch convention as the 3D
# scene.  Keeping the semantic IDs is what removes the rectangle/half ambiguity
# of the previous Hough-only fit.
PITCH_KEYPOINTS = np.float32(
    [
        (-52.5, -34.0),
        (-52.5, -20.16),
        (-52.5, -9.16),
        (-52.5, 9.16),
        (-52.5, 20.16),
        (-52.5, 34.0),
        (-47.0, -9.16),
        (-47.0, 9.16),
        (-41.5, 0.0),
        (-36.0, -20.16),
        (-36.0, -9.16),
        (-36.0, 9.16),
        (-36.0, 20.16),
        (0.0, -34.0),
        (0.0, -9.15),
        (0.0, 9.15),
        (0.0, 34.0),
        (36.0, -20.16),
        (36.0, -9.16),
        (36.0, 9.16),
        (36.0, 20.16),
        (41.5, 0.0),
        (47.0, -9.16),
        (47.0, 9.16),
        (52.5, -34.0),
        (52.5, -20.16),
        (52.5, -9.16),
        (52.5, 9.16),
        (52.5, 20.16),
        (52.5, 34.0),
        (-9.15, 0.0),
        (9.15, 0.0),
    ]
)


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = source @ homography.T
    valid = np.abs(projected[:, 2]) > 1e-8
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def _visible_side(indices: np.ndarray, confidences: np.ndarray) -> str | None:
    left = float(confidences[(indices <= 12)].sum())
    right = float(confidences[(indices >= 17) & (indices <= 29)].sum())
    if left > right * 1.15:
        return "left"
    if right > left * 1.15:
        return "right"
    weighted_x = float(np.average(PITCH_KEYPOINTS[indices, 0], weights=confidences))
    if abs(weighted_x) < 4.0:
        return None
    return "left" if weighted_x < 0 else "right"


def calibration_from_pose_result(
    result,
    frame_index: int,
    confidence_threshold: float = 0.35,
) -> PitchCalibration | None:
    """Fit an image-to-pitch homography from semantic field keypoints.

    The Ultralytics pose result can contain more than one pitch proposal.  Each
    proposal is fitted independently with RANSAC and the geometrically strongest
    one is returned.  Confidence is deliberately based on reprojection quality,
    not only on the neural-network score.
    """
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    xy = keypoints.xy.detach().cpu().numpy()
    if xy.shape[1] != len(PITCH_KEYPOINTS):
        return None
    if keypoints.conf is None:
        point_confidences = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        point_confidences = keypoints.conf.detach().cpu().numpy()
    boxes = getattr(result, "boxes", None)
    object_confidences = (
        boxes.conf.detach().cpu().numpy()
        if boxes is not None and boxes.conf is not None
        else np.ones(len(xy), dtype=np.float32)
    )

    best: PitchCalibration | None = None
    for pose_index, (points, scores) in enumerate(zip(xy, point_confidences)):
        valid = (
            (points[:, 0] > 1.0)
            & (points[:, 1] > 1.0)
            & (scores >= confidence_threshold)
        )
        if int(valid.sum()) < 6:
            valid = (
                (points[:, 0] > 1.0)
                & (points[:, 1] > 1.0)
                & (scores >= max(0.18, confidence_threshold * 0.6))
            )
        indices = np.flatnonzero(valid)
        if len(indices) < 6:
            continue
        image_points = np.float32(points[indices])
        pitch_points = np.float32(PITCH_KEYPOINTS[indices])
        if cv2.contourArea(cv2.convexHull(image_points)) < 80.0:
            continue
        if cv2.contourArea(cv2.convexHull(pitch_points)) < 18.0:
            continue
        homography, inlier_mask = cv2.findHomography(
            image_points,
            pitch_points,
            cv2.RANSAC,
            1.5,
            maxIters=5000,
            confidence=0.999,
        )
        if homography is None or inlier_mask is None or not np.isfinite(homography).all():
            continue
        if abs(float(np.linalg.det(homography))) < 1e-10:
            continue
        homography /= homography[2, 2]
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 6:
            continue
        inlier_ratio = inlier_count / len(indices)
        try:
            pitch_to_image = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            continue
        reprojected = _project(pitch_points[inliers], pitch_to_image)
        image_error = np.linalg.norm(reprojected - image_points[inliers], axis=1)
        reprojection_error = float(np.median(image_error))
        reprojection_p95 = float(np.percentile(image_error, 95))
        if not np.isfinite(reprojection_error) or reprojection_error > 18.0:
            continue
        keypoint_score = float(np.mean(scores[indices][inliers]))
        object_score = float(object_confidences[min(pose_index, len(object_confidences) - 1)])
        geometry_score = exp(-reprojection_error / 7.0)
        confidence = float(
            np.clip(
                0.16 * object_score
                + 0.22 * keypoint_score
                + 0.34 * inlier_ratio
                + 0.28 * geometry_score,
                0.0,
                0.99,
            )
        )
        side = _visible_side(indices[inliers], scores[indices][inliers])
        all_reprojected = _project(pitch_points, pitch_to_image)
        all_image_error = np.linalg.norm(all_reprojected - image_points, axis=1)
        raw_keypoints = tuple(
            {
                "id": int(semantic_index),
                "image": {
                    "x": round(float(point[0]), 3),
                    "y": round(float(point[1]), 3),
                },
                "pitch": {
                    "x": round(float(PITCH_KEYPOINTS[semantic_index, 0]), 4),
                    "z": round(float(PITCH_KEYPOINTS[semantic_index, 1]), 4),
                },
                "confidence": round(float(score), 5),
                "inlier": bool(is_inlier),
                "imageResidualPixels": (
                    round(float(residual), 4) if np.isfinite(residual) else None
                ),
            }
            for semantic_index, point, score, is_inlier, residual in zip(
                indices,
                image_points,
                scores[indices],
                inliers,
                all_image_error,
            )
        )
        candidate = PitchCalibration(
            image_to_pitch=homography,
            confidence=confidence,
            supported_lines=inlier_count,
            mean_line_score=inlier_ratio,
            rectangle=f"field-keypoints-{side}" if side else "field-keypoints",
            matched_curves=int(any(index in {14, 15, 30, 31} for index in indices[inliers])),
            method="roboflow-field-keypoints",
            keypoint_count=len(indices),
            inlier_count=inlier_count,
            reprojection_error=reprojection_error,
            frame_index=frame_index,
            detected_keypoint_count=len(indices),
            completed_keypoint_count=len(indices),
            inlier_ratio=inlier_ratio,
            reprojection_p95=reprojection_p95,
            raw_keypoints=raw_keypoints,
        )
        if best is None or (candidate.confidence, candidate.inlier_count) > (
            best.confidence,
            best.inlier_count,
        ):
            best = candidate
    return best


def calibration_from_worker_result(item: dict) -> PitchCalibration | None:
    matrix = np.asarray(item.get("imageToPitch"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        return None
    matrix /= matrix[2, 2]
    side = item.get("pitchSide")
    rectangle = f"field-keypoints-{side}" if side in {"left", "right"} else "field-keypoints"
    return PitchCalibration(
        image_to_pitch=matrix,
        confidence=float(np.clip(item.get("confidence", 0.0), 0.0, 0.99)),
        supported_lines=int(item.get("lineCount") or item.get("inlierCount") or 0),
        mean_line_score=float(item.get("inlierRatio") or 0.0),
        rectangle=rectangle,
        matched_curves=int(item.get("matchedCurves") or 0),
        method=str(item.get("method") or "pnlcalib-worker"),
        keypoint_count=int(item.get("keypointCount") or 0),
        inlier_count=int(item.get("inlierCount") or 0),
        reprojection_error=(
            float(item["reprojectionError"])
            if item.get("reprojectionError") is not None
            else None
        ),
        frame_index=int(item.get("frameIndex") or 0),
        detected_keypoint_count=int(
            item.get("detectedKeypointCount") or item.get("keypointCount") or 0
        ),
        completed_keypoint_count=int(
            item.get("completedKeypointCount") or item.get("keypointCount") or 0
        ),
        inlier_ratio=(
            float(item["inlierRatio"]) if item.get("inlierRatio") is not None else None
        ),
        reprojection_p95=(
            float(item["reprojectionP95"])
            if item.get("reprojectionP95") is not None
            else None
        ),
        raw_line_count=int(
            item.get("detectedLineCount") or item.get("lineCount") or 0
        ),
        ground_error_p50=(
            float(item["groundErrorP50Metres"])
            if item.get("groundErrorP50Metres") is not None
            else None
        ),
        ground_error_p95=(
            float(item["groundErrorP95Metres"])
            if item.get("groundErrorP95Metres") is not None
            else None
        ),
        raw_keypoints=tuple(item.get("rawKeypoints") or ()),
        raw_lines=tuple(item.get("rawLines") or ()),
        confidence_kind=str(item.get("confidenceKind") or "heuristic-quality-score"),
        backend_diagnostics=(
            dict(item["backendDiagnostics"])
            if isinstance(item.get("backendDiagnostics"), dict)
            else None
        ),
    )
