"""Provider-neutral person and generic-ball candidate selection."""

from __future__ import annotations

from math import hypot

import cv2

from .bounding_box_geometry import intersection_over_union
from .person_appearance import appearance_feature, green_ratio, is_pitch_person
from .person_detection_policy import (
    GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS,
    GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS,
    GENERIC_BALL_MINIMUM_CENTER_Y_RATIO,
    GENERIC_BALL_MINIMUM_GRASS_RATIO,
    PERSON_LOCAL_NMS_IOU,
)
from .person_detection_provider_contract import RawFramePrediction
from .reconstruction_person_detection_contract import Detection


# Football-tuned checkpoints publish domain classes instead of COCO indices;
# class semantics are therefore resolved by NAME with the COCO ids as the
# fallback for checkpoints that expose no usable name map.
PERSON_CLASS_NAMES = frozenset(
    {"person", "player", "goalkeeper", "referee", "staff"}
)
BALL_CLASS_NAMES = frozenset({"ball", "sports ball"})


def detection_class_ids(names: object) -> tuple[frozenset[int], frozenset[int]]:
    """Resolve (person ids, ball ids) from a model/result class-name map."""

    person_ids: set[int] = set()
    ball_ids: set[int] = set()
    if isinstance(names, dict):
        for index, name in names.items():
            lowered = str(name).strip().lower()
            if lowered in PERSON_CLASS_NAMES:
                person_ids.add(int(index))
            elif lowered in BALL_CLASS_NAMES:
                ball_ids.add(int(index))
    if not person_ids:
        person_ids = {0}
        ball_ids = {32}
    return frozenset(person_ids), frozenset(ball_ids)


def parse_person_prediction(
    prediction: RawFramePrediction,
    *,
    debug_log: list[dict] | None = None,
) -> tuple[list[Detection], list[dict]]:
    """Select usable people and balls while recording every raw-box verdict."""

    image = prediction.image_bgr
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, _ = image.shape[:2]
    names = prediction.names
    person_ids, ball_ids = detection_class_ids(names)
    people: list[tuple[float, tuple[float, float, float, float], dict | None]] = []
    balls: list[tuple[dict, dict | None]] = []

    for index, box in enumerate(prediction.boxes):
        class_id = box.class_id
        confidence = box.confidence
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        box_width, box_height = x2 - x1, y2 - y1
        center_x = (x1 + x2) / 2
        record: dict | None = None
        if debug_log is not None:
            record = {
                "index": index,
                "classId": int(class_id),
                "className": str(names.get(int(class_id), int(class_id))),
                "confidence": round(float(confidence), 3),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
            debug_log.append(record)
        if class_id in person_ids:
            gate: dict | None = {} if record is not None else None
            accepted = is_pitch_person(
                hsv, (x1, y1, x2, y2), float(confidence), debug=gate
            )
            if record is not None:
                record["gate"] = gate
                record["verdict"] = (
                    "person-candidate"
                    if accepted
                    else (gate or {}).get("verdict") or "rejected-off-pitch"
                )
            if not accepted:
                continue
            people.append((float(confidence), (x1, y1, x2, y2), record))
        elif class_id in ball_ids:
            center_y = (y1 + y2) / 2
            radius = max(7, int(max(box_width, box_height) * 1.8))
            if not (
                center_y > height * GENERIC_BALL_MINIMUM_CENTER_Y_RATIO
                and max(box_width, box_height)
                < GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS
            ):
                if record is not None:
                    record["verdict"] = "rejected-ball-geometry"
                continue
            context = green_ratio(hsv, center_x, center_y, radius, radius)
            if record is not None:
                record["ballGrassRatio"] = round(context, 3)
            if context >= GENERIC_BALL_MINIMUM_GRASS_RATIO:
                if record is not None:
                    record["verdict"] = "ball-candidate"
                balls.append(
                    (
                        {
                            "x": center_x,
                            "y": center_y,
                            "confidence": float(confidence),
                        },
                        record,
                    )
                )
            elif record is not None:
                record["verdict"] = "rejected-ball-off-grass"

    kept: list[tuple[float, tuple[float, float, float, float], dict | None]] = []
    for confidence, box, record in sorted(
        people, key=lambda item: (item[0], item[1]), reverse=True
    ):
        if all(
            intersection_over_union(box, existing) < PERSON_LOCAL_NMS_IOU
            for _, existing, _record in kept
        ):
            kept.append((confidence, box, record))
            if record is not None:
                record["verdict"] = "accepted-person"
        elif record is not None:
            record["verdict"] = "rejected-person-nms"

    detections = [
        Detection(
            x=(x1 + x2) / 2,
            y=y2,
            width=x2 - x1,
            height=y2 - y1,
            confidence=confidence,
            feature=appearance_feature(image, (x1, y1, x2, y2)),
        )
        for confidence, (x1, y1, x2, y2), _record in kept
    ]
    unique_balls: list[dict] = []
    for ball, record in sorted(
        balls, key=lambda item: item[0]["confidence"], reverse=True
    ):
        if any(
            hypot(ball["x"] - kept_ball["x"], ball["y"] - kept_ball["y"])
            < GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS
            for kept_ball in unique_balls
        ):
            if record is not None:
                record["verdict"] = "rejected-ball-duplicate"
            continue
        if record is not None:
            record["verdict"] = "accepted-ball"
        unique_balls.append(ball)
    return detections, unique_balls
