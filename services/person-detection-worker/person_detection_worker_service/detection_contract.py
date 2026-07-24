from __future__ import annotations

"""Strict request parsing for the person-detection worker."""

import json
from math import isfinite


class DetectionRequestError(ValueError):
    pass


FIELDS = frozenset(
    {
        "contractVersion",
        "imageSize",
        "confidence",
        "nmsIou",
        "maxDetections",
    }
)


def parse_manifest(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DetectionRequestError("Detection manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DetectionRequestError("Detection manifest must be an object")
    unknown = sorted(set(value) - FIELDS)
    if unknown:
        raise DetectionRequestError(
            "Detection manifest has unknown fields: " + ", ".join(unknown)
        )
    image_size = value.get("imageSize")
    confidence = value.get("confidence")
    nms_iou = value.get("nmsIou")
    max_detections = value.get("maxDetections")
    if value.get("contractVersion") != 1:
        raise DetectionRequestError("Unsupported detection contractVersion")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or not 320 <= image_size <= 4096
    ):
        raise DetectionRequestError("imageSize must be between 320 and 4096")
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or not 1 <= max_detections <= 1000
    ):
        raise DetectionRequestError(
            "maxDetections must be between 1 and 1000"
        )
    for field, number in (("confidence", confidence), ("nmsIou", nms_iou)):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or not 0.0 < float(number) <= 1.0
        ):
            raise DetectionRequestError(f"{field} must be in (0, 1]")
    return {
        "imageSize": int(image_size),
        "confidence": float(confidence),
        "nmsIou": float(nms_iou),
        "maxDetections": int(max_detections),
    }
