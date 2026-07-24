"""Auditable, cache-invalidating provenance for person detector evidence."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Mapping

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


def local_person_detection_provider_info(
    model_name: str,
    model: object | None = None,
) -> dict:
    checkpoint = _person_checkpoint_identity(model_name, model)
    digest = str(checkpoint.get("sha256") or "content-unavailable")
    return {
        "backend": "ultralytics-yolo",
        "providerVersion": _installed_package_version("ultralytics"),
        "modelVersion": digest[:16],
        "checkpoint": checkpoint,
        "device": str(get_settings().reconstruction_device),
        "batchSize": 1,
        "torchVersion": _installed_package_version("torch"),
        "mpsFallbackEnabled": False,
    }


def person_detection_input(
    model_name: str,
    model: object | None = None,
    *,
    provider_info: Mapping | None = None,
) -> dict:
    """Return the complete cache identity for sampled-frame base inference."""

    runtime = dict(
        provider_info
        or local_person_detection_provider_info(model_name, model)
    )
    return {
        "schemaVersion": 2,
        "provider": {
            "backend": runtime.get("backend"),
            "version": runtime.get("providerVersion"),
            "modelVersion": runtime.get("modelVersion"),
        },
        "postprocessRuntime": {
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        },
        "checkpoint": dict(
            runtime.get("checkpoint")
            or _person_checkpoint_identity(model_name, model)
        ),
        "classes": {"person": 0, "genericBallFallback": 32},
        "inference": {
            "imageSize": int(GENERIC_ULTRALYTICS_IMAGE_SIZE),
            "confidence": float(GENERIC_ULTRALYTICS_CONFIDENCE),
            "providerNmsIou": float(DETECTOR_PROVIDER_NMS_IOU),
            "maxDetections": int(DETECTOR_MAX_DETECTIONS),
            "device": str(runtime.get("device") or "unknown"),
            "batchSize": int(runtime.get("batchSize") or 1),
            "torchVersion": runtime.get("torchVersion"),
            "mpsFallbackEnabled": bool(
                runtime.get("mpsFallbackEnabled", False)
            ),
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
