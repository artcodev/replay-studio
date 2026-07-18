"""Person detection and post-association identity benchmark evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .identity_metrics import evaluate_identity_assignments
from .quality_benchmark_context import BenchmarkSampleContext
from .quality_benchmark_statistics import (
    bbox,
    distribution,
    f1,
    finite_number,
    intersection_over_union,
    ratio,
    rounded,
)


def _valid_people(frame: dict[str, Any], *, prediction: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, person in enumerate(frame.get("persons") or []):
        if not isinstance(person, dict):
            continue
        person_bbox = bbox(person.get("bbox"))
        if person_bbox is None:
            continue
        identifier_key = "trackId" if prediction else "id"
        identifier = str(person.get(identifier_key) or "").strip()
        confidence = finite_number(person.get("confidence")) if prediction else 1.0
        result.append(
            {
                "identifier": identifier or None,
                "bbox": person_bbox,
                "confidence": confidence if confidence is not None else 1.0,
                "ordinal": index,
            }
        )
    return result


def _match_people(
    ground_truth: Sequence[dict[str, Any]],
    predicted: Sequence[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    if not ground_truth or not predicted:
        return [], set(range(len(ground_truth))), set(range(len(predicted)))
    overlaps = np.asarray(
        [
            [intersection_over_union(gt["bbox"], pred["bbox"]) for pred in predicted]
            for gt in ground_truth
        ],
        dtype=float,
    )
    # A valid edge is worth more than the maximum possible IoU gain, so the
    # assignment first maximises match count and then overlap quality.
    rewards = np.where(overlaps >= threshold, 2.0 + overlaps, 0.0)
    gt_indices, prediction_indices = linear_sum_assignment(-rewards)
    matches = [
        (int(gt_index), int(prediction_index), float(overlaps[gt_index, prediction_index]))
        for gt_index, prediction_index in zip(gt_indices, prediction_indices)
        if overlaps[gt_index, prediction_index] >= threshold
    ]
    matched_gt = {item[0] for item in matches}
    matched_prediction = {item[1] for item in matches}
    return (
        matches,
        set(range(len(ground_truth))) - matched_gt,
        set(range(len(predicted))) - matched_prediction,
    )


def _average_precision(
    detections: Sequence[dict[str, Any]],
    ground_truth_by_frame: dict[tuple[str, int], list[dict[str, Any]]],
    threshold: float,
) -> float | None:
    ground_truth_count = sum(len(items) for items in ground_truth_by_frame.values())
    if not ground_truth_count:
        return None
    ordered = sorted(
        detections,
        key=lambda item: (
            -item["confidence"],
            item["sampleId"],
            item["frameIndex"],
            item["ordinal"],
        ),
    )
    matched: dict[tuple[str, int], set[int]] = {}
    true_positive_flags: list[int] = []
    for prediction in ordered:
        frame_key = (prediction["sampleId"], prediction["frameIndex"])
        frame_ground_truth = ground_truth_by_frame.get(frame_key, [])
        used = matched.setdefault(frame_key, set())
        candidates = [
            (intersection_over_union(gt["bbox"], prediction["bbox"]), index)
            for index, gt in enumerate(frame_ground_truth)
            if index not in used
        ]
        overlap, gt_index = max(candidates, default=(0.0, -1))
        is_true_positive = gt_index >= 0 and overlap >= threshold
        if is_true_positive:
            used.add(gt_index)
        true_positive_flags.append(1 if is_true_positive else 0)
    if not ordered:
        return 0.0
    cumulative_true_positive = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for rank, flag in enumerate(true_positive_flags, start=1):
        cumulative_true_positive += flag
        precisions.append(cumulative_true_positive / rank)
        recalls.append(cumulative_true_positive / ground_truth_count)
    envelope = list(precisions)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    area = 0.0
    previous_recall = 0.0
    for recall, precision, flag in zip(recalls, envelope, true_positive_flags):
        if flag:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return rounded(area)


def evaluate_people_and_identity(
    contexts: Sequence[BenchmarkSampleContext], threshold: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate spatial person detection and IDs derived from the same matches."""

    ground_truth_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    confidence_detections: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    matched_overlaps: list[float] = []
    ground_truth_count = prediction_count = true_positive = 0
    invalid_prediction_count = 0
    for context in contexts:
        frame_indices = sorted(
            set(context.ground_truth_frames) | set(context.prediction_frames)
        )
        for frame_index in frame_indices:
            ground_truth_frame = context.ground_truth_frames.get(frame_index) or {}
            prediction_frame = context.prediction_frames.get(frame_index) or {}
            ground_truth = _valid_people(ground_truth_frame, prediction=False)
            predicted = _valid_people(prediction_frame, prediction=True)
            raw_predictions = prediction_frame.get("persons") or []
            invalid_prediction_count += max(0, len(raw_predictions) - len(predicted))
            frame_key = (context.sample_id, frame_index)
            ground_truth_by_frame[frame_key] = ground_truth
            ground_truth_count += len(ground_truth)
            prediction_count += len(predicted)
            for item in predicted:
                confidence_detections.append(
                    {**item, "sampleId": context.sample_id, "frameIndex": frame_index}
                )
            matches, unmatched_gt, unmatched_prediction = _match_people(
                ground_truth, predicted, threshold
            )
            true_positive += len(matches)
            matched_overlaps.extend(match[2] for match in matches)
            global_frame_index = context.sample_index * 1_000_000_000 + frame_index
            for gt_index, prediction_index, _ in matches:
                gt_id = ground_truth[gt_index]["identifier"]
                predicted_id = predicted[prediction_index]["identifier"]
                identity_rows.append(
                    {
                        "frameIndex": global_frame_index,
                        "groundTruthId": f"{context.sample_id}:{gt_id}" if gt_id else None,
                        "predictedId": (
                            f"{context.sample_id}:{predicted_id}" if predicted_id else None
                        ),
                    }
                )
            for gt_index in sorted(unmatched_gt):
                gt_id = ground_truth[gt_index]["identifier"]
                identity_rows.append(
                    {
                        "frameIndex": global_frame_index,
                        "groundTruthId": f"{context.sample_id}:{gt_id}" if gt_id else None,
                        "predictedId": None,
                    }
                )
            for prediction_index in sorted(unmatched_prediction):
                predicted_id = predicted[prediction_index]["identifier"]
                identity_rows.append(
                    {
                        "frameIndex": global_frame_index,
                        "groundTruthId": None,
                        "predictedId": (
                            f"{context.sample_id}:{predicted_id}" if predicted_id else None
                        ),
                    }
                )
    false_positive = prediction_count - true_positive
    false_negative = ground_truth_count - true_positive
    precision = ratio(true_positive, prediction_count)
    recall = ratio(true_positive, ground_truth_count)
    person_metrics = {
        "available": ground_truth_count > 0,
        "iouThreshold": threshold,
        "groundTruthCount": ground_truth_count,
        "predictionCount": prediction_count,
        "invalidPredictionCount": invalid_prediction_count,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "averagePrecisionAtIou": _average_precision(
            confidence_detections, ground_truth_by_frame, threshold
        ),
        "matchedIou": distribution(matched_overlaps, "ratio"),
    }
    identity = evaluate_identity_assignments(identity_rows or None)
    ground_truth_identity_count = int(identity.get("groundTruthIdentityCount") or 0)
    fragment_count = int(identity.get("fragmentCount") or 0)
    identity_metrics = {
        **identity,
        "fragmentationRate": ratio(fragment_count, ground_truth_identity_count),
        "matchingIouThreshold": threshold,
        "note": (
            "Detections are spatially matched before IDF1 evaluation. HOTA and GS-HOTA "
            "require their official dataset evaluator."
        ),
    }
    return person_metrics, identity_metrics
