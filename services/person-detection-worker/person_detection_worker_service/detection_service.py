from __future__ import annotations

"""Decode one binary frame and execute the configured detector."""

from time import perf_counter

import cv2
import numpy as np

from .detection_contract import DetectionRequestError, parse_manifest
from .ultralytics_engine import UltralyticsDetectionEngine


class FrameDecodeError(DetectionRequestError):
    pass


class PersonDetectionService:
    def __init__(self, engine: UltralyticsDetectionEngine) -> None:
        self.engine = engine

    def process(self, frame_bytes: bytes, manifest: str) -> dict:
        request_started = perf_counter()
        policy = parse_manifest(manifest)
        decode_started = perf_counter()
        encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FrameDecodeError("Uploaded frame is not a decodable image")
        decode_seconds = perf_counter() - decode_started
        names, boxes, inference_seconds, degenerate_box_count = (
            self.engine.predict(image, policy)
        )
        info = self.engine.info()
        info.pop("modelLoadSeconds", None)
        return {
            **info,
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "names": {str(index): name for index, name in names.items()},
            "boxes": boxes,
            "diagnostics": {
                "decodeSeconds": round(decode_seconds, 6),
                "inferenceSeconds": round(inference_seconds, 6),
                "requestSeconds": round(perf_counter() - request_started, 6),
                "boxCount": len(boxes),
                "degenerateBoxCount": degenerate_box_count,
            },
        }
