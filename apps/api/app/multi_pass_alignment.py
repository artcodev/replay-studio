from __future__ import annotations

"""Temporal alignment of independently recorded views of one event."""

from math import isfinite

import cv2
import numpy as np

from .reconstruction_inputs import frame_paths


def motion_dtw(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) < 2 or len(candidate) < 2:
        return {
            "cost": 1.0,
            "anchors": [
                {"reference": 0.0, "pass": 0.0},
                {"reference": 1.0, "pass": 1.0},
            ],
        }
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    distances = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    distances[0, 0] = 0.0
    for left_index in range(1, len(left) + 1):
        for right_index in range(1, len(right) + 1):
            distances[left_index, right_index] = abs(
                left[left_index - 1] - right[right_index - 1]
            ) + min(
                distances[left_index - 1, right_index],
                distances[left_index, right_index - 1],
                distances[left_index - 1, right_index - 1],
            )

    path: list[tuple[int, int]] = []
    left_index, right_index = len(left), len(right)
    while left_index > 0 and right_index > 0:
        path.append((left_index - 1, right_index - 1))
        choices = (
            (
                distances[left_index - 1, right_index - 1],
                left_index - 1,
                right_index - 1,
            ),
            (distances[left_index - 1, right_index], left_index - 1, right_index),
            (distances[left_index, right_index - 1], left_index, right_index - 1),
        )
        _, left_index, right_index = min(choices, key=lambda item: item[0])
    path.reverse()

    anchors = []
    pass_indexes = np.linspace(0, len(right) - 1, min(7, len(right))).round().astype(int)
    for pass_index in pass_indexes:
        matches = [
            left_position
            for left_position, right_position in path
            if right_position == pass_index
        ]
        if not matches:
            matches = [min(path, key=lambda item: abs(item[1] - pass_index))[0]]
        anchors.append(
            {
                "reference": round(
                    float(np.median(matches)) / max(1, len(left) - 1), 4
                ),
                "pass": round(float(pass_index) / max(1, len(right) - 1), 4),
            }
        )
    return {
        "cost": round(float(distances[-1, -1]) / (len(left) + len(right)), 4),
        "anchors": anchors,
    }


def classify_pass_relation(
    motion_cost: float,
    segment: dict,
    reference_segment: dict,
) -> str:
    if motion_cost <= 0.07:
        return "replay-overlap"
    before_gap = float(reference_segment.get("start", 0)) - float(
        segment.get("end", 0)
    )
    after_gap = float(segment.get("start", 0)) - float(
        reference_segment.get("end", 0)
    )
    if -0.15 <= before_gap <= 0.4:
        return "continuation-before"
    if -0.15 <= after_gap <= 0.4:
        return "continuation-after"
    return "independent"


def motion_signature(scene: dict, bins: int = 24) -> list[float]:
    values: list[float] = []
    times: list[float] = []
    previous: np.ndarray | None = None
    duration = max(0.1, float(scene.get("duration") or 0.1))
    for path, time in frame_paths(scene):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        if previous is not None:
            values.append(float(cv2.absdiff(previous, image).mean() / 255.0))
            times.append(min(1.0, max(0.0, time / duration)))
        previous = image
    if len(values) < 2:
        return [0.0] * bins
    series = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(series, [10, 90])
    series = np.clip((series - low) / max(1e-6, high - low), 0.0, 1.0)
    return np.interp(
        np.linspace(0.0, 1.0, bins),
        times,
        series,
        left=series[0],
        right=series[-1],
    ).tolist()


def manual_clock_alignment(
    saved_anchors: object,
    reference_scene: dict,
    pass_scene: dict,
    pass_segment: dict,
) -> tuple[dict | None, dict | None]:
    """Validate a pass-scoped manual clock map and return diagnostics."""

    if isinstance(saved_anchors, dict):
        records = [saved_anchors]
    elif isinstance(saved_anchors, list):
        records = saved_anchors
    else:
        return None, None

    source_scene_id = str(pass_scene.get("id") or "")
    source_segment_id = str(pass_segment.get("id") or "")
    matched_records = 0
    candidates: list[object] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_scene_id = str(record.get("sourceSceneId") or "").strip()
        record_segment_id = str(record.get("segmentId") or "").strip()
        if not record_scene_id and not record_segment_id:
            continue
        if record_scene_id and record_scene_id != source_scene_id:
            continue
        if record_segment_id and record_segment_id != source_segment_id:
            continue
        matched_records += 1
        grouped = record.get("anchors")
        candidates.extend(grouped if isinstance(grouped, list) else [record])

    if matched_records == 0:
        return None, None

    reference_duration = max(0.0, float(reference_scene.get("duration") or 0.0))
    pass_duration = max(0.0, float(pass_scene.get("duration") or 0.0))
    valid: list[dict[str, float]] = []
    rejection_reasons: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejection_reasons.add("anchor-not-an-object")
            continue
        reference_time = candidate.get("referenceTime")
        pass_time = candidate.get("passTime")
        if isinstance(reference_time, bool) or isinstance(pass_time, bool):
            rejection_reasons.add("anchor-time-not-numeric")
            continue
        try:
            reference_value = float(reference_time)
            pass_value = float(pass_time)
        except (TypeError, ValueError):
            rejection_reasons.add("anchor-time-not-numeric")
            continue
        if not isfinite(reference_value) or not isfinite(pass_value):
            rejection_reasons.add("anchor-time-not-finite")
            continue
        if not 0.0 <= reference_value <= reference_duration:
            rejection_reasons.add("reference-time-out-of-range")
            continue
        if not 0.0 <= pass_value <= pass_duration:
            rejection_reasons.add("pass-time-out-of-range")
            continue
        valid.append(
            {
                "referenceTime": round(reference_value, 6),
                "passTime": round(pass_value, 6),
            }
        )

    diagnostics = {
        "status": "rejected",
        "matchedRecordCount": matched_records,
        "providedAnchorCount": len(candidates),
        "validAnchorCount": len(valid),
        "rejectionReasons": sorted(rejection_reasons),
    }
    if len(valid) < 2:
        diagnostics["rejectionReasons"] = sorted(
            {*rejection_reasons, "at-least-two-valid-anchors-required"}
        )
        return None, diagnostics

    ordered = sorted(valid, key=lambda item: (item["referenceTime"], item["passTime"]))
    if any(
        right["referenceTime"] <= left["referenceTime"]
        or right["passTime"] <= left["passTime"]
        for left, right in zip(ordered, ordered[1:])
    ):
        diagnostics["rejectionReasons"] = sorted(
            {*rejection_reasons, "anchors-not-strictly-monotonic"}
        )
        return None, diagnostics

    return {
        "relation": "replay-overlap",
        "method": "manual-clock-anchors",
        "confidence": 1.0,
        "motionCost": None,
        "overlap": True,
        "anchors": ordered,
        "manualAlignment": {
            **diagnostics,
            "status": "accepted",
            "rejectionReasons": sorted(rejection_reasons),
        },
    }, None


def temporal_alignment(
    reference_scene: dict,
    pass_scene: dict,
    reference_segment: dict,
    pass_segment: dict,
    manual_alignment_anchors: object = None,
) -> dict:
    if reference_scene["id"] == pass_scene["id"]:
        return {
            "relation": "reference",
            "method": "identity",
            "confidence": 1.0,
            "motionCost": 0.0,
            "overlap": True,
            "anchors": [
                {"referenceTime": 0.0, "passTime": 0.0},
                {
                    "referenceTime": reference_scene["duration"],
                    "passTime": pass_scene["duration"],
                },
            ],
        }
    manual_alignment, manual_diagnostics = manual_clock_alignment(
        manual_alignment_anchors,
        reference_scene,
        pass_scene,
        pass_segment,
    )
    if manual_alignment is not None:
        return manual_alignment
    result = motion_dtw(motion_signature(reference_scene), motion_signature(pass_scene))
    relation = classify_pass_relation(result["cost"], pass_segment, reference_segment)
    if relation == "replay-overlap":
        confidence = max(0.55, min(0.95, 1.0 - result["cost"] / 0.12))
        method = "motion-dtw"
    elif relation.startswith("continuation"):
        confidence = 0.9
        method = "source-continuity"
    else:
        confidence = 0.2
        method = "phase-normalized"
    alignment = {
        "relation": relation,
        "method": method,
        "confidence": round(confidence, 3),
        "motionCost": result["cost"],
        "overlap": relation == "replay-overlap",
        "anchors": [
            {
                "referenceTime": round(
                    item["reference"] * reference_scene["duration"], 3
                ),
                "passTime": round(item["pass"] * pass_scene["duration"], 3),
            }
            for item in result["anchors"]
        ],
    }
    if manual_diagnostics is not None:
        alignment["manualAlignment"] = manual_diagnostics
    return alignment


def map_reference_time(anchors: list[dict], reference_time: float) -> float:
    ordered = sorted(anchors, key=lambda item: item["referenceTime"])
    if reference_time <= ordered[0]["referenceTime"]:
        return float(ordered[0]["passTime"])
    if reference_time >= ordered[-1]["referenceTime"]:
        return float(ordered[-1]["passTime"])
    for left, right in zip(ordered, ordered[1:]):
        if left["referenceTime"] <= reference_time <= right["referenceTime"]:
            width = max(1e-6, right["referenceTime"] - left["referenceTime"])
            progress = (reference_time - left["referenceTime"]) / width
            return float(
                left["passTime"] + (right["passTime"] - left["passTime"]) * progress
            )
    return reference_time
