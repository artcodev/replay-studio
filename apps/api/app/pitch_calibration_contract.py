from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pitch_side(rectangle: str | None) -> str | None:
    if rectangle and rectangle.endswith("-left"):
        return "left"
    if rectangle and rectangle.endswith("-right"):
        return "right"
    return None


def opposite_pitch_preset(preset: str) -> str:
    if preset.endswith("-left"):
        return f"{preset[:-5]}-right"
    if preset.endswith("-right"):
        return f"{preset[:-6]}-left"
    return preset


@dataclass(frozen=True)
class PitchCalibration:
    image_to_pitch: np.ndarray
    confidence: float
    supported_lines: int
    mean_line_score: float
    rectangle: str
    matched_curves: int = 0
    method: str = "pitch-lines-ransac"
    keypoint_count: int = 0
    inlier_count: int = 0
    reprojection_error: float | None = None
    frame_index: int | None = None
    detected_keypoint_count: int = 0
    completed_keypoint_count: int = 0
    inlier_ratio: float | None = None
    reprojection_p95: float | None = None
    raw_line_count: int = 0
    ground_error_p50: float | None = None
    ground_error_p95: float | None = None
    raw_keypoints: tuple[dict, ...] = ()
    raw_lines: tuple[dict, ...] = ()
    confidence_kind: str = "heuristic-quality-score"
    backend_diagnostics: dict | None = None

    def as_dict(self) -> dict:
        return {
            "status": "ready",
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "supportedLines": self.supported_lines,
            "matchedCurves": self.matched_curves,
            "meanLineScore": round(self.mean_line_score, 3),
            "rectangle": self.rectangle,
            "pitchSide": pitch_side(self.rectangle),
            "keypointCount": self.keypoint_count,
            "inlierCount": self.inlier_count,
            "reprojectionError": (
                round(self.reprojection_error, 3)
                if self.reprojection_error is not None
                else None
            ),
            "frameIndex": self.frame_index,
            "detectedKeypointCount": self.detected_keypoint_count,
            "completedKeypointCount": self.completed_keypoint_count,
            "inlierRatio": (
                round(self.inlier_ratio, 5) if self.inlier_ratio is not None else None
            ),
            "reprojectionP95": (
                round(self.reprojection_p95, 3) if self.reprojection_p95 is not None else None
            ),
            "rawLineCount": self.raw_line_count,
            "groundErrorP50Metres": (
                round(self.ground_error_p50, 4) if self.ground_error_p50 is not None else None
            ),
            "groundErrorP95Metres": (
                round(self.ground_error_p95, 4) if self.ground_error_p95 is not None else None
            ),
            "rawKeypoints": [dict(item) for item in self.raw_keypoints],
            "rawLines": [dict(item) for item in self.raw_lines],
            "confidenceKind": self.confidence_kind,
            "backendDiagnostics": self.backend_diagnostics,
            "imageToPitch": [
                [round(float(value), 8) for value in row]
                for row in self.image_to_pitch
            ],
        }


@dataclass(frozen=True)
class CalibrationAlignmentMetrics:
    """Bidirectional image-space agreement between evidence and a camera fit."""

    precision: float
    recall: float
    f1: float
    residual_p50: float
    residual_p95: float
    model_sample_count: int
    observed_sample_count: int
    tolerance_pixels: float

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "f1": round(self.f1, 5),
            "residualP50": round(self.residual_p50, 3),
            "residualP95": round(self.residual_p95, 3),
            "modelSampleCount": self.model_sample_count,
            "observedSampleCount": self.observed_sample_count,
            "tolerancePixels": round(self.tolerance_pixels, 2),
        }
