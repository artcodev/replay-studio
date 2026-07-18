"""Pitch-calibration benchmark evaluation."""

from __future__ import annotations

from math import hypot
from typing import Any, Sequence

from .quality_benchmark_context import BenchmarkSampleContext
from .quality_benchmark_statistics import distribution, point, ratio


def evaluate_calibration(
    contexts: Sequence[BenchmarkSampleContext],
) -> dict[str, Any]:
    """Measure labelled image and pitch point errors and coverage."""

    reprojection_errors: list[float] = []
    metric_errors: list[float] = []
    ground_truth_point_count = 0
    predicted_image_count = predicted_pitch_count = 0
    for context in contexts:
        for frame_index, ground_truth_frame in context.ground_truth_frames.items():
            prediction_frame = context.prediction_frames.get(frame_index) or {}
            predicted_points = {
                str(calibration_point.get("id")): calibration_point
                for calibration_point in prediction_frame.get("calibrationPoints") or []
                if isinstance(calibration_point, dict)
                and str(calibration_point.get("id") or "").strip()
            }
            for ground_truth_point in ground_truth_frame.get("calibrationPoints") or []:
                if not isinstance(ground_truth_point, dict):
                    continue
                point_id = str(ground_truth_point.get("id") or "")
                true_image = point(ground_truth_point.get("image"))
                true_pitch = point(ground_truth_point.get("pitch"))
                if true_image is None or true_pitch is None:
                    continue
                ground_truth_point_count += 1
                predicted_point = predicted_points.get(point_id) or {}
                predicted_image = point(predicted_point.get("image"))
                predicted_pitch = point(predicted_point.get("pitch"))
                if predicted_image is not None:
                    predicted_image_count += 1
                    reprojection_errors.append(
                        hypot(
                            predicted_image[0] - true_image[0],
                            predicted_image[1] - true_image[1],
                        )
                    )
                if predicted_pitch is not None:
                    predicted_pitch_count += 1
                    metric_errors.append(
                        hypot(
                            predicted_pitch[0] - true_pitch[0],
                            predicted_pitch[1] - true_pitch[1],
                        )
                    )
    return {
        "available": ground_truth_point_count > 0,
        "groundTruthPointCount": ground_truth_point_count,
        "reprojectionCoverage": ratio(predicted_image_count, ground_truth_point_count),
        "metricProjectionCoverage": ratio(predicted_pitch_count, ground_truth_point_count),
        "reprojectionError": distribution(reprojection_errors, "px"),
        "metricProjectionError": distribution(metric_errors, "m"),
    }
