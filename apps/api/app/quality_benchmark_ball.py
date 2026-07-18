"""Ball localisation benchmark evaluation."""

from __future__ import annotations

from math import hypot
from typing import Any, Sequence

from .quality_benchmark_context import BenchmarkSampleContext
from .quality_benchmark_statistics import distribution, f1, point, ratio


def evaluate_ball(
    contexts: Sequence[BenchmarkSampleContext], threshold: float
) -> dict[str, Any]:
    """Measure labelled ball point precision, recall, and localisation error."""

    visible_ground_truth_count = predicted_count = true_positive = 0
    false_positive = false_negative = 0
    errors: list[float] = []
    invalid_prediction_count = 0
    for context in contexts:
        frame_indices = sorted(
            set(context.ground_truth_frames) | set(context.prediction_frames)
        )
        for frame_index in frame_indices:
            ground_truth_ball = (
                context.ground_truth_frames.get(frame_index) or {}
            ).get("ball")
            prediction_ball = (
                context.prediction_frames.get(frame_index) or {}
            ).get("ball")
            true_point = (
                point(ground_truth_ball.get("center"))
                if isinstance(ground_truth_ball, dict)
                and ground_truth_ball.get("visible") is True
                else None
            )
            predicted_point = (
                point(prediction_ball.get("center"))
                if isinstance(prediction_ball, dict)
                else None
            )
            if isinstance(prediction_ball, dict) and predicted_point is None:
                invalid_prediction_count += 1
            if true_point is not None:
                visible_ground_truth_count += 1
            if predicted_point is not None:
                predicted_count += 1
            if true_point is not None and predicted_point is not None:
                error = hypot(
                    predicted_point[0] - true_point[0],
                    predicted_point[1] - true_point[1],
                )
                errors.append(error)
                if error <= threshold:
                    true_positive += 1
                else:
                    false_positive += 1
                    false_negative += 1
            elif true_point is not None:
                false_negative += 1
            elif predicted_point is not None:
                false_positive += 1
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, visible_ground_truth_count)
    return {
        "available": visible_ground_truth_count > 0,
        "pointThresholdPx": threshold,
        "visibleGroundTruthCount": visible_ground_truth_count,
        "predictionCount": predicted_count,
        "invalidPredictionCount": invalid_prediction_count,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "pointError": distribution(errors, "px"),
    }
