from __future__ import annotations

from math import hypot

import numpy as np

from .pitch_calibration_contract import (
    CalibrationAlignmentMetrics,
    PitchCalibration,
)
from .pitch_geometry import (
    PITCH_LINES,
    PNLCALIB_LINE_TO_PITCH_LINE,
    project_points,
)
from .pitch_image_evidence import alignment_residuals


def calibration_alignment_metrics(
    image: np.ndarray,
    calibration: PitchCalibration,
    tolerance_pixels: float = 3.0,
) -> CalibrationAlignmentMetrics | None:
    residuals = alignment_residuals(image, calibration)
    if residuals is None:
        return None
    precision = float(np.mean(residuals.model_to_observed <= tolerance_pixels))
    recall = float(np.mean(residuals.observed_to_model <= tolerance_pixels))
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return CalibrationAlignmentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        residual_p50=float(np.median(residuals.model_to_observed)),
        residual_p95=float(np.percentile(residuals.model_to_observed, 95)),
        model_sample_count=residuals.model_sample_count,
        observed_sample_count=residuals.observed_sample_count,
        tolerance_pixels=float(tolerance_pixels),
    )


def calibration_alignment_error(
    image: np.ndarray,
    calibration: PitchCalibration,
) -> float | None:
    metrics = calibration_alignment_metrics(image, calibration)
    return round(metrics.residual_p50, 2) if metrics is not None else None


def semantic_line_evidence(calibration: PitchCalibration) -> list[dict]:
    """Add an image residual to every observed PnL semantic line."""
    if not calibration.raw_lines:
        return []
    pitch_lines = {
        name: (np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
        for name, start, end in PITCH_LINES
    }
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return [dict(line) for line in calibration.raw_lines]

    result: list[dict] = []
    for raw_line in calibration.raw_lines:
        evidence = dict(raw_line)
        evidence["residualP50"] = None
        evidence["residualP95"] = None
        if evidence.get("groundPlane") is False:
            evidence["residualStatus"] = "not-scored-3d"
            result.append(evidence)
            continue
        pitch_line_name = PNLCALIB_LINE_TO_PITCH_LINE.get(
            str(evidence.get("name") or "")
        )
        pitch_segment = pitch_lines.get(pitch_line_name or "")
        start = evidence.get("start")
        end = evidence.get("end")
        if (
            pitch_segment is None
            or not isinstance(start, dict)
            or not isinstance(end, dict)
        ):
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        model_image = project_points(np.vstack(pitch_segment), pitch_to_image)
        if not np.isfinite(model_image).all():
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        direction = model_image[1] - model_image[0]
        denominator = hypot(float(direction[0]), float(direction[1]))
        if denominator < 1e-7:
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        observed = np.asarray(
            [
                [float(start["x"]), float(start["y"])],
                [float(end["x"]), float(end["y"])],
            ],
            dtype=np.float64,
        )
        relative = observed - model_image[0]
        residual_values = np.abs(
            direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
        ) / denominator
        evidence["residualP50"] = round(float(np.median(residual_values)), 3)
        evidence["residualP95"] = round(
            float(np.percentile(residual_values, 95)), 3
        )
        evidence["residualStatus"] = "scored"
        result.append(evidence)
    return result
