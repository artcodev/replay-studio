from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np


def _score_signature(frame: np.ndarray) -> np.ndarray | None:
    """Return a normalized two-digit scoreboard fingerprint when a cyan score box is visible."""
    height, width = frame.shape[:2]
    top = frame[: max(80, int(height * 0.18)), : int(width * 0.7)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([70, 80, 100]), np.array([100, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            width * 0.045 <= box_width <= width * 0.13
            and height * 0.045 <= box_height <= height * 0.13
            and 1.4 <= box_width / max(1, box_height) <= 2.4
        ):
            boxes.append((x, y, box_width, box_height))
    if not boxes:
        return None
    x, y, box_width, box_height = max(boxes, key=lambda item: item[2] * item[3])
    fingerprints: list[np.ndarray] = []
    for start_ratio, end_ratio in ((0.05, 0.48), (0.52, 0.95)):
        start = x + int(box_width * start_ratio)
        end = x + int(box_width * end_ratio)
        crop = cv2.cvtColor(top[y : y + box_height, start:end], cv2.COLOR_BGR2GRAY)
        digit_mask = (crop < 80).astype(np.uint8)
        digit_contours, _ = cv2.findContours(
            digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        usable = [item for item in digit_contours if cv2.contourArea(item) >= 12]
        if not usable:
            return None
        digit = max(usable, key=cv2.contourArea)
        digit_x, digit_y, digit_width, digit_height = cv2.boundingRect(digit)
        normalized = cv2.resize(
            digit_mask[digit_y : digit_y + digit_height, digit_x : digit_x + digit_width],
            (16, 24),
            interpolation=cv2.INTER_NEAREST,
        )
        fingerprints.append(normalized)
    return np.concatenate(fingerprints, axis=1)


def detect_score_changes(source: Path, duration: float) -> tuple[list[float], float]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return [], 0.0
    samples: list[tuple[float, np.ndarray]] = []
    sample_times = np.arange(0.75, max(0.76, duration - 0.25), 1.0)
    for time in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time) * 1000.0)
        success, frame = capture.read()
        if not success:
            continue
        signature = _score_signature(frame)
        if signature is not None:
            samples.append((float(time), signature))
    capture.release()
    coverage = len(samples) / max(1, len(sample_times))
    if not samples:
        return [], coverage

    current = samples[0][1]
    candidate: np.ndarray | None = None
    candidate_start = 0.0
    candidate_count = 0
    changes: list[float] = []
    for time, signature in samples[1:]:
        if float(np.mean(signature != current)) <= 0.08:
            candidate = None
            candidate_count = 0
            continue
        if candidate is not None and float(np.mean(signature != candidate)) <= 0.08:
            candidate_count += 1
        else:
            candidate = signature
            candidate_start = time
            candidate_count = 1
        if candidate_count >= 2 and candidate is not None:
            changes.append(round(candidate_start, 2))
            current = candidate
            candidate = None
            candidate_count = 0
    return changes, coverage


def _segment_motion_signature(
    capture: cv2.VideoCapture, segment: dict, bins: int = 24
) -> list[float]:
    start = float(segment["start"])
    end = float(segment["end"])
    sample_times = np.linspace(start + 0.08, max(start + 0.09, end - 0.08), bins + 1)
    values: list[float] = []
    previous: np.ndarray | None = None
    for time in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time) * 1000.0)
        success, frame = capture.read()
        if not success:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if previous is not None:
            values.append(float(cv2.absdiff(previous, gray).mean() / 255.0))
        previous = gray
    if len(values) < 2:
        return [0.0] * bins
    series = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(series, [10, 90])
    return np.clip((series - low) / max(1e-6, high - low), 0.0, 1.0).tolist()


def _dtw_cost(reference: list[float], candidate: list[float]) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    distances = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    distances[0, 0] = 0.0
    for left_index in range(1, len(left) + 1):
        for right_index in range(1, len(right) + 1):
            distances[left_index, right_index] = abs(left[left_index - 1] - right[right_index - 1]) + min(
                distances[left_index - 1, right_index],
                distances[left_index, right_index - 1],
                distances[left_index - 1, right_index - 1],
            )
    return round(float(distances[-1, -1]) / max(1, len(left) + len(right)), 4)


def _variant(index: int) -> str:
    value = index
    output = ""
    while value >= 0:
        output = chr(65 + value % 26) + output
        value = value // 26 - 1
    return output


def build_segment_layout(
    segments: list[dict],
    score_change_times: list[float],
    motion_costs: dict[tuple[str, str], float] | None = None,
    scoreboard_coverage: float = 0.0,
) -> dict:
    if not segments:
        return {
            "status": "proposed",
            "method": "empty",
            "confidence": 0.0,
            "scoreChangeTimes": [],
            "groups": [],
            "warnings": ["No continuous shots were detected."],
        }
    ordered = sorted(segments, key=lambda item: float(item["start"]))
    if score_change_times:
        boundaries = [
            (left + right) / 2.0
            for left, right in zip(score_change_times, score_change_times[1:])
        ]
        group_indexes = [
            bisect_right(boundaries, (float(segment["start"]) + float(segment["end"])) / 2.0) + 1
            for segment in ordered
        ]
        method = "scoreboard-change+motion-dtw"
        base_confidence = min(0.95, 0.68 + scoreboard_coverage * 0.2)
        warnings = ["Event boundaries are inferred from stable scoreboard changes and require review."]
    else:
        chunk_size = 3 if len(ordered) <= 9 else 4
        group_indexes = [index // chunk_size + 1 for index in range(len(ordered))]
        method = "shot-order-fallback"
        base_confidence = 0.38
        warnings = ["No stable scoreboard was found; shots are grouped by order and require manual review."]

    costs = motion_costs or {}
    grouped: dict[int, list[dict]] = {}
    for segment, group_index in zip(ordered, group_indexes):
        grouped.setdefault(group_index, []).append(segment)

    groups = []
    for group_index, group_segments in grouped.items():
        replay_count = 0
        for variant_index, segment in enumerate(group_segments):
            earlier = group_segments[:variant_index]
            pair_costs = [
                costs.get((earlier_segment["id"], segment["id"]), 1.0)
                for earlier_segment in earlier
            ]
            best_cost = min(pair_costs, default=1.0)
            if variant_index == 0:
                role = "original"
                role_confidence = base_confidence
            elif best_cost <= 0.095:
                role = "replay"
                replay_count += 1
                role_confidence = max(0.55, min(0.95, 1.0 - best_cost / 0.16))
            else:
                role = "continuation"
                role_confidence = 0.55
            variant = _variant(variant_index)
            segment["layout"] = {
                "group": group_index,
                "variant": variant,
                "label": f"{group_index}-{variant}",
                "role": role,
                "confidence": round(role_confidence, 3),
                "motionCost": None if variant_index == 0 else round(best_cost, 4),
            }
        groups.append(
            {
                "id": f"event-{group_index}",
                "index": group_index,
                "label": str(group_index),
                "segmentIds": [item["id"] for item in group_segments],
                "replayCount": replay_count,
            }
        )

    return {
        "status": "proposed",
        "method": method,
        "confidence": round(base_confidence, 3),
        "scoreChangeTimes": [round(item, 2) for item in score_change_times],
        "groups": groups,
        "warnings": warnings,
    }


def propose_segment_layout(source: Path, segments: list[dict], duration: float) -> dict:
    score_change_times, coverage = detect_score_changes(source, duration)
    capture = cv2.VideoCapture(str(source))
    signatures = {
        segment["id"]: _segment_motion_signature(capture, segment)
        for segment in segments
    }
    capture.release()
    motion_costs: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(segments):
        for right in segments[left_index + 1 :]:
            motion_costs[(left["id"], right["id"])] = _dtw_cost(
                signatures[left["id"]], signatures[right["id"]]
            )
    return build_segment_layout(
        segments,
        score_change_times,
        motion_costs=motion_costs,
        scoreboard_coverage=coverage,
    )
