from __future__ import annotations

"""Strict conversion of the PnLCalib worker wire result into the domain DTO."""

import numpy as np

from .pitch_calibration_contract import PitchCalibration


def calibration_from_worker_result(item: dict) -> PitchCalibration | None:
    matrix = np.asarray(item.get("imageToPitch"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        return None
    matrix /= matrix[2, 2]
    side = item.get("pitchSide")
    rectangle = (
        f"field-keypoints-{side}"
        if side in {"left", "right"}
        else "field-keypoints"
    )
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
            float(item["inlierRatio"])
            if item.get("inlierRatio") is not None
            else None
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
        confidence_kind=str(
            item.get("confidenceKind") or "heuristic-quality-score"
        ),
        backend_diagnostics=(
            dict(item["backendDiagnostics"])
            if isinstance(item.get("backendDiagnostics"), dict)
            else None
        ),
    )


__all__ = ("calibration_from_worker_result",)
