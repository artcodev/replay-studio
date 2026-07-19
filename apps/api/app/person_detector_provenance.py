"""Auditable, cache-invalidating provenance for person detector evidence."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import cv2
import numpy as np

from .checkpoint_identity import file_content_sha256
from .config import get_settings
from .person_detection_policy import (
    APPEARANCE_FEATURE_SCHEMA_VERSION,
    DETECTOR_MAX_DETECTIONS,
    DETECTOR_PROVIDER_NMS_IOU,
    GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS,
    GENERIC_BALL_FILTER_POLICY_VERSION,
    GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS,
    GENERIC_BALL_MINIMUM_CENTER_Y_RATIO,
    GENERIC_BALL_MINIMUM_GRASS_RATIO,
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
    MINIMUM_PERSON_FOOT_Y,
    PERSON_FILTER_POLICY_VERSION,
    PERSON_LOCAL_NMS_IOU,
    SHALLOW_PERSON_CONFIDENCE,
    SHALLOW_PERSON_FOOT_Y,
    SHALLOW_PERSON_GRASS_RATIO,
)


def _person_checkpoint_identity(
    model_name: str,
    model: object | None = None,
) -> dict:
    """Resolve checkpoint provenance without Ultralytics implementation details."""

    candidates: list[Path] = []
    for raw in (model_name, getattr(model, "ckpt_path", None)):
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        candidate = (
            (Path.cwd() / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        if candidate not in candidates:
            candidates.append(candidate)
    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    identity: dict = {"requested": str(model_name)}
    if checkpoint is None:
        identity["contentAvailable"] = False
        return identity
    stat = checkpoint.stat()
    identity.update(
        {
            "contentAvailable": True,
            "name": checkpoint.name,
            "size": int(stat.st_size),
            "sha256": file_content_sha256(checkpoint),
        }
    )
    return identity


def _installed_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def person_detection_input(model_name: str, model: object | None = None) -> dict:
    """Return the complete cache identity for sampled-frame base inference."""

    return {
        "schemaVersion": 1,
        "provider": {
            "backend": "ultralytics-yolo",
            "version": _installed_package_version("ultralytics"),
        },
        "postprocessRuntime": {
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        },
        "checkpoint": _person_checkpoint_identity(model_name, model),
        "classes": {"person": 0, "genericBallFallback": 32},
        "inference": {
            "imageSize": int(GENERIC_ULTRALYTICS_IMAGE_SIZE),
            "confidence": float(GENERIC_ULTRALYTICS_CONFIDENCE),
            "providerNmsIou": float(DETECTOR_PROVIDER_NMS_IOU),
            "maxDetections": int(DETECTOR_MAX_DETECTIONS),
            "device": str(get_settings().reconstruction_device),
        },
        "personFilter": {
            "version": PERSON_FILTER_POLICY_VERSION,
            "localNmsIou": float(PERSON_LOCAL_NMS_IOU),
            "minimumFootYRatio": float(MINIMUM_PERSON_FOOT_Y),
            "shallowFootYRatio": float(SHALLOW_PERSON_FOOT_Y),
            "shallowConfidence": float(SHALLOW_PERSON_CONFIDENCE),
            "shallowGrassRatio": float(SHALLOW_PERSON_GRASS_RATIO),
            "appearanceFeatureSchema": APPEARANCE_FEATURE_SCHEMA_VERSION,
        },
        "genericBallFallbackFilter": {
            "version": GENERIC_BALL_FILTER_POLICY_VERSION,
            "minimumCenterYRatio": GENERIC_BALL_MINIMUM_CENTER_Y_RATIO,
            "maximumBoxSizePixels": GENERIC_BALL_MAXIMUM_BOX_SIZE_PIXELS,
            "minimumGrassRatio": GENERIC_BALL_MINIMUM_GRASS_RATIO,
            "deduplicationRadiusPixels": GENERIC_BALL_DEDUPLICATION_RADIUS_PIXELS,
        },
    }
