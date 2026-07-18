"""Ultralytics person inference and parsing for sampled broadcast frames."""

from __future__ import annotations

from math import hypot
from pathlib import Path

import cv2

from .bounding_box_geometry import intersection_over_union
from .config import get_settings
from .person_appearance import appearance_feature, green_ratio, is_pitch_person
from .person_detection_policy import (
    DETECTOR_MAX_DETECTIONS,
    DETECTOR_PROVIDER_NMS_IOU,
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
    GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS,
    GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS,
    GENERIC_BALL_MINIMUM_CENTER_Y_RATIO,
    GENERIC_BALL_MINIMUM_GRASS_RATIO,
    PERSON_LOCAL_NMS_IOU,
)
from .reconstruction_person_detection_contract import Detection


def predict_frame(model, path: Path | str):
    return model.predict(
        str(path),
        imgsz=GENERIC_ULTRALYTICS_IMAGE_SIZE,
        conf=GENERIC_ULTRALYTICS_CONFIDENCE,
        classes=[0, 32],
        iou=DETECTOR_PROVIDER_NMS_IOU,
        max_det=DETECTOR_MAX_DETECTIONS,
        device=get_settings().reconstruction_device,
        verbose=False,
    )[0]


def parse_person_detections(result) -> tuple[list[Detection], list[dict]]:
    image = result.orig_img
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, _ = image.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    people: list[tuple[float, tuple[float, float, float, float]]] = []
    balls: list[dict] = []

    for box, class_id, confidence in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = (float(value) for value in box)
        box_width, box_height = x2 - x1, y2 - y1
        center_x = (x1 + x2) / 2
        if class_id == 0:
            if not is_pitch_person(hsv, (x1, y1, x2, y2), float(confidence)):
                continue
            people.append((float(confidence), (x1, y1, x2, y2)))
        elif class_id == 32:
            center_y = (y1 + y2) / 2
            radius = max(7, int(max(box_width, box_height) * 1.8))
            if (
                center_y > height * GENERIC_BALL_MINIMUM_CENTER_Y_RATIO
                and max(box_width, box_height)
                < GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS
            ):
                context = green_ratio(hsv, center_x, center_y, radius, radius)
                if context >= GENERIC_BALL_MINIMUM_GRASS_RATIO:
                    balls.append(
                        {
                            "x": center_x,
                            "y": center_y,
                            "confidence": float(confidence),
                        }
                    )

    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for confidence, box in sorted(people, reverse=True):
        if all(
            intersection_over_union(box, existing) < PERSON_LOCAL_NMS_IOU
            for _, existing in kept
        ):
            kept.append((confidence, box))

    detections = [
        Detection(
            x=(x1 + x2) / 2,
            y=y2,
            width=x2 - x1,
            height=y2 - y1,
            confidence=confidence,
            feature=appearance_feature(image, (x1, y1, x2, y2)),
        )
        for confidence, (x1, y1, x2, y2) in kept
    ]
    unique_balls: list[dict] = []
    for ball in sorted(balls, key=lambda item: item["confidence"], reverse=True):
        if any(
            hypot(ball["x"] - kept_ball["x"], ball["y"] - kept_ball["y"])
            < GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS
            for kept_ball in unique_balls
        ):
            continue
        unique_balls.append(ball)
    return detections, unique_balls
