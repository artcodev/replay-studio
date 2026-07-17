"""Label-dependent identity metrics for reconstructed player tracks.

Runtime evidence coverage is not identity accuracy.  This module only emits
accuracy values when explicit ground-truth assignments are supplied.  Rows
represent an already spatially matched observation; either side may be null to
encode a false negative or false positive.

Official HOTA/GS-HOTA still belongs to the SoccerNet evaluator because it also
scores detection and game-state association.  We expose that limitation
instead of deriving a similarly named heuristic from smooth trajectories.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import median
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _time_coordinate(
    row: dict[str, Any],
    fallback: int,
    *,
    frame_rate: float | None,
) -> tuple[float, float | None, str]:
    """Return a sortable coordinate and, only when known, seconds.

    A frame index is not a duration.  It becomes seconds only when the caller
    supplies the labelled assignment timebase explicitly.  This keeps switch
    ordering available for sparse frame-only labels without publishing frames
    under a `...Seconds` field.
    """

    for key in ("sceneTime", "time"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            return value, value, "seconds"
    try:
        frame_index = float(row.get("frameIndex"))
    except (TypeError, ValueError):
        frame_index = float(fallback)
    if not isfinite(frame_index):
        frame_index = float(fallback)
    if frame_rate is not None:
        seconds = frame_index / frame_rate
        # Use normalized seconds for ordering and same-instant grouping too.
        # Otherwise a mixed labelled set would treat ``sceneTime=1`` and
        # ``frameIndex=25`` at 25 fps as different moments.
        return seconds, seconds, "frame-index+explicit-fps"
    return frame_index, None, "frame-index"


def evaluate_identity_assignments(
    rows: Iterable[dict[str, Any]] | None,
    *,
    frame_rate: float | None = None,
) -> dict[str, Any]:
    """Compute global IDF1 and transparent identity error counts.

    Supported row keys are ``groundTruthId``/``gtId`` and
    ``predictedId``/``canonicalPersonId``.  The global bipartite mapping is the
    standard protection against arbitrary label names: only the best one-to-one
    GT/predicted assignment contributes ID true positives.
    """

    observations = [row for row in (rows or []) if isinstance(row, dict)]
    if not observations:
        return {
            "groundTruthAvailable": False,
            "status": "unavailable",
            "reason": "No labelled identity assignments were supplied.",
            "idf1": None,
            "idPrecision": None,
            "idRecall": None,
            "hota": None,
            "gsHota": None,
        }

    try:
        normalized_frame_rate = float(frame_rate) if frame_rate is not None else None
    except (TypeError, ValueError, OverflowError):
        normalized_frame_rate = None
    if normalized_frame_rate is not None and (
        not isfinite(normalized_frame_rate) or normalized_frame_rate <= 0.0
    ):
        normalized_frame_rate = None

    normalized: list[tuple[float, float | None, str, str | None, str | None]] = []
    for index, row in enumerate(observations):
        ground_truth = _identifier(row.get("groundTruthId", row.get("gtId")))
        predicted = _identifier(row.get("predictedId", row.get("canonicalPersonId")))
        if ground_truth is None and predicted is None:
            continue
        coordinate, seconds, timebase = _time_coordinate(
            row,
            index,
            frame_rate=normalized_frame_rate,
        )
        normalized.append((coordinate, seconds, timebase, ground_truth, predicted))

    deduplicated: list[
        tuple[float, float | None, str, str | None, str | None]
    ] = []
    prediction_by_gt_sample: dict[tuple[float, str], str | None] = {}
    conflicting_samples: list[dict[str, Any]] = []
    for row in normalized:
        coordinate, _, _, ground_truth, predicted = row
        if ground_truth is None:
            deduplicated.append(row)
            continue
        key = (coordinate, ground_truth)
        if key not in prediction_by_gt_sample:
            prediction_by_gt_sample[key] = predicted
            deduplicated.append(row)
            continue
        previous = prediction_by_gt_sample[key]
        if previous != predicted:
            conflicting_samples.append(
                {
                    "time": coordinate,
                    "groundTruthId": ground_truth,
                    "predictedIds": sorted(
                        {
                            "<missing>" if value is None else value
                            for value in (previous, predicted)
                        }
                    ),
                }
            )
    if conflicting_samples:
        return {
            "groundTruthAvailable": True,
            "status": "invalid",
            "reason": (
                "One labelled ground-truth identity has conflicting predictions "
                "at the same timestamp."
            ),
            "invalidAssignmentCount": len(conflicting_samples),
            "invalidAssignments": conflicting_samples,
            "idf1": None,
            "idPrecision": None,
            "idRecall": None,
            "hota": None,
            "gsHota": None,
            "officialEvaluatorRequired": ["HOTA", "GS-HOTA"],
        }
    normalized = deduplicated

    ground_truth_ids = sorted({gt for _, _, _, gt, _ in normalized if gt is not None})
    predicted_ids = sorted({pred for _, _, _, _, pred in normalized if pred is not None})
    gt_index = {value: index for index, value in enumerate(ground_truth_ids)}
    predicted_index = {value: index for index, value in enumerate(predicted_ids)}
    contingency = np.zeros((len(ground_truth_ids), len(predicted_ids)), dtype=np.int64)
    for _, _, _, ground_truth, predicted in normalized:
        if ground_truth is not None and predicted is not None:
            contingency[gt_index[ground_truth], predicted_index[predicted]] += 1

    if contingency.size:
        maximum = int(contingency.max(initial=0))
        rows_index, columns_index = linear_sum_assignment(maximum - contingency)
        id_true_positives = int(contingency[rows_index, columns_index].sum())
        assignment = {
            ground_truth_ids[row]: predicted_ids[column]
            for row, column in zip(rows_index, columns_index)
            if contingency[row, column] > 0
        }
    else:
        id_true_positives = 0
        assignment = {}

    gt_observations = sum(
        ground_truth is not None for _, _, _, ground_truth, _ in normalized
    )
    predicted_observations = sum(
        predicted is not None for _, _, _, _, predicted in normalized
    )
    id_false_negatives = gt_observations - id_true_positives
    id_false_positives = predicted_observations - id_true_positives
    precision = (
        id_true_positives / predicted_observations if predicted_observations else 0.0
    )
    recall = id_true_positives / gt_observations if gt_observations else 0.0
    denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    idf1 = 2 * id_true_positives / denominator if denominator else 0.0

    by_ground_truth: dict[str, list[tuple[float, str | None]]] = defaultdict(list)
    by_time_and_prediction: dict[tuple[float, str], set[str]] = defaultdict(set)
    seconds_by_coordinate: dict[float, float | None] = {}
    timebases: set[str] = set()
    for timestamp, seconds, timebase, ground_truth, predicted in normalized:
        timebases.add(timebase)
        previous_seconds = seconds_by_coordinate.get(timestamp)
        if timestamp not in seconds_by_coordinate:
            seconds_by_coordinate[timestamp] = seconds
        elif previous_seconds != seconds:
            # Mixed/inconsistent time units for the same evaluation sample are
            # still usable for counts, but never for duration.
            seconds_by_coordinate[timestamp] = None
        if ground_truth is not None:
            by_ground_truth[ground_truth].append((timestamp, predicted))
        if ground_truth is not None and predicted is not None:
            by_time_and_prediction[(timestamp, predicted)].add(ground_truth)

    id_switches = fragments = 0
    for values in by_ground_truth.values():
        previous: str | None = None
        in_predicted_run = False
        has_predicted_run = False
        for _, predicted in sorted(values):
            if predicted is None:
                in_predicted_run = False
                continue
            if previous is not None and predicted != previous:
                id_switches += 1
            if not in_predicted_run and has_predicted_run:
                fragments += 1
            previous = predicted
            in_predicted_run = True
            has_predicted_run = True

    duplicate_frames = len(
        {
            timestamp
            for (timestamp, _), identities in by_time_and_prediction.items()
            if len(identities) > 1
        }
    )
    ordered_seconds = sorted(
        {
            seconds
            for seconds in seconds_by_coordinate.values()
            if seconds is not None and isfinite(seconds)
        }
    )
    duration_cadence = (
        median(
            [
                right - left
                for left, right in zip(ordered_seconds, ordered_seconds[1:])
                if right > left
            ]
        )
        if len(ordered_seconds) > 1
        and all(value is not None for value in seconds_by_coordinate.values())
        else None
    )
    duplicate_overlap_seconds = (
        0.0
        if duplicate_frames == 0
        else round(duplicate_frames / normalized_frame_rate, 6)
        if normalized_frame_rate is not None
        and "frame-index+explicit-fps" in timebases
        else round(duplicate_frames * duration_cadence, 6)
        if duration_cadence is not None
        else None
    )
    if timebases == {"seconds"}:
        timebase = "seconds"
    elif timebases == {"frame-index+explicit-fps"}:
        timebase = "frame-index+explicit-fps"
    elif timebases == {"frame-index"}:
        timebase = "frame-index-without-fps"
    else:
        timebase = "mixed"

    return {
        "groundTruthAvailable": True,
        "status": "evaluated",
        "sampleCount": len(normalized),
        "groundTruthIdentityCount": len(ground_truth_ids),
        "predictedIdentityCount": len(predicted_ids),
        "idTruePositives": id_true_positives,
        "idFalsePositives": id_false_positives,
        "idFalseNegatives": id_false_negatives,
        "idPrecision": round(precision, 6),
        "idRecall": round(recall, 6),
        "idf1": round(idf1, 6),
        "idSwitchCount": id_switches,
        "fragmentCount": fragments,
        "duplicateAssignmentFrameCount": duplicate_frames,
        "duplicateOverlapSeconds": duplicate_overlap_seconds,
        "duplicateOverlapTimebase": timebase,
        "identityAssignmentFrameRate": normalized_frame_rate,
        "globalAssignment": assignment,
        "hota": None,
        "gsHota": None,
        "officialEvaluatorRequired": ["HOTA", "GS-HOTA"],
    }


__all__ = ["evaluate_identity_assignments"]
