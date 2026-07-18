"""Cache adapter and orchestration for immutable base person detections."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

import app.ultralytics_person_inference as ultralytics_person_inference

from .person_detection_cache import (
    PERSON_DETECTION_CACHE_SCHEMA_VERSION,
    PersonDetectionCacheError,
    frame_content_sha256,
    lookup_person_detection_cache,
    store_person_detection_cache,
)
from .reconstruction_person_detection_contract import Detection


def base_detection_cache_diagnostics(frame_count: int, detector_input: dict) -> dict:
    return {
        "schemaVersion": 1,
        "artifactSchemaVersion": PERSON_DETECTION_CACHE_SCHEMA_VERSION,
        "frameCount": int(frame_count),
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "errors": 0,
        "corruptArtifacts": 0,
        "providerCalls": 0,
        "input": deepcopy(detector_input),
    }


def _base_detection_payload(detection: Detection) -> dict:
    """Serialize only immutable pre-annotation detector evidence."""

    return {
        "x": float(detection.x),
        "y": float(detection.y),
        "width": float(detection.width),
        "height": float(detection.height),
        "confidence": float(detection.confidence),
        "feature": np.asarray(detection.feature, dtype=np.float32).tolist(),
    }


def _base_detection_from_payload(payload: Mapping) -> Detection:
    return Detection(
        x=float(payload["x"]),
        y=float(payload["y"]),
        width=float(payload["width"]),
        height=float(payload["height"]),
        confidence=float(payload["confidence"]),
        feature=np.asarray(payload["feature"], dtype=np.float32).copy(),
    )


def cached_base_frame_detections(
    model: object,
    path: Path,
    asset_directory: Path,
    detector_input: Mapping,
    diagnostics: dict,
) -> tuple[np.ndarray, list[Detection], list[dict]]:
    """Load or compute one frame's base detections before all manual state."""

    frame_digest: str | None = None
    try:
        frame_digest = frame_content_sha256(path)
        lookup = lookup_person_detection_cache(
            asset_directory,
            frame_sha256=frame_digest,
            detector_input=detector_input,
        )
    except (OSError, PersonDetectionCacheError, ValueError) as exc:
        lookup = None
        diagnostics["errors"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {"frame": path.name, "stage": "lookup", "detail": str(exc)}
        )

    if lookup is not None and lookup.entry is not None:
        image = cv2.imread(str(path))
        if image is not None and (image.shape[1], image.shape[0]) == lookup.entry.image_size:
            people_payload, balls, _ = lookup.entry.as_pipeline_data()
            diagnostics["hits"] += 1
            return (
                image,
                [_base_detection_from_payload(item) for item in people_payload],
                balls,
            )
        diagnostics["errors"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {
                "frame": path.name,
                "stage": "decode",
                "detail": "cached image size does not match the current decoded JPEG",
            }
        )
    elif lookup is not None and lookup.status in {"corrupt", "error"}:
        diagnostics["errors"] += 1
        if lookup.status == "corrupt":
            diagnostics["corruptArtifacts"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {
                "frame": path.name,
                "stage": "lookup",
                "detail": lookup.error or lookup.status,
            }
        )

    diagnostics["misses"] += 1
    diagnostics["providerCalls"] += 1
    result = ultralytics_person_inference.predict_frame(model, path)
    image = result.orig_img
    people, balls = ultralytics_person_inference.parse_person_detections(result)
    if frame_digest is not None:
        try:
            stored = store_person_detection_cache(
                asset_directory,
                frame_sha256=frame_digest,
                detector_input=detector_input,
                image_size=(int(image.shape[1]), int(image.shape[0])),
                people=[_base_detection_payload(item) for item in people],
                generic_ball_candidates=balls,
            )
        except (OSError, PersonDetectionCacheError, ValueError) as exc:
            diagnostics["errors"] += 1
            diagnostics.setdefault("errorDetails", []).append(
                {"frame": path.name, "stage": "write", "detail": str(exc)}
            )
        else:
            if stored is not None:
                diagnostics["writes"] += 1
    return image, people, balls
