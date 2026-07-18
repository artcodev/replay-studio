from __future__ import annotations

"""Immutable reconstruction inputs for selecting a ball detector backend."""

from pathlib import Path

from .config import get_settings
from .person_detection_policy import (
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
)
from .reconstruction_errors import ReconstructionError


BALL_DETECTION_BACKENDS = frozenset(
    {
        "generic-ultralytics",
        "dedicated-ultralytics",
        "wasb-service",
    }
)


def ball_checkpoint_identity(path: str | Path) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    identity: dict = {"name": checkpoint.name}
    if checkpoint.is_file():
        stat = checkpoint.stat()
        identity.update({"size": int(stat.st_size), "mtimeNs": int(stat.st_mtime_ns)})
    return identity


def verify_queued_ball_checkpoint(path: str | Path, expected: object) -> None:
    if not isinstance(expected, dict):
        return
    actual = ball_checkpoint_identity(path)
    mismatches = [
        key
        for key in ("name", "size", "mtimeNs")
        if expected.get(key) is not None and expected.get(key) != actual.get(key)
    ]
    if mismatches:
        raise ReconstructionError(
            "Queued ball checkpoint no longer matches the local file "
            f"({', '.join(mismatches)} changed); queue a new reconstruction run."
        )


def ball_detection_input(backend: str | None = None) -> dict:
    settings = get_settings()
    selected = str(backend or settings.ball_detection_backend)
    if selected not in BALL_DETECTION_BACKENDS:
        raise ReconstructionError(f"Unsupported ball detection backend: {selected}")

    generic_input = {
        "backend": "generic-ultralytics",
        "modelSource": "reconstruction-model",
        "classId": 32,
        "confidence": float(GENERIC_ULTRALYTICS_CONFIDENCE),
        "imageSize": int(GENERIC_ULTRALYTICS_IMAGE_SIZE),
        "nmsIou": float(settings.ball_detection_nms_iou),
    }
    dedicated_input = {
        "backend": "dedicated-ultralytics",
        "checkpoint": ball_checkpoint_identity(settings.ball_detection_model),
        "classId": 0,
        "confidence": float(settings.ball_detection_confidence),
        "imageSize": int(settings.ball_detection_image_size),
        "tileSize": int(settings.ball_detection_tile_size),
        "tileOverlap": float(settings.ball_detection_tile_overlap),
        "inferenceBatchSize": int(settings.ball_detection_inference_batch_size),
        "nmsIou": float(settings.ball_detection_nms_iou),
        "adaptiveRoi": {
            "enabled": True,
            "algorithmVersion": "adaptive-roi-v1",
            "fullScanIntervalFrames": int(
                settings.ball_detection_full_scan_interval
            ),
            "maxRegions": int(settings.ball_detection_roi_region_count),
            "paddingPixels": int(settings.ball_detection_roi_padding),
            # Changing this policy must invalidate raw-candidate caches even
            # when all numeric settings remain identical.
            "reacquirePolicy": "same-frame-global-on-miss",
        },
    }

    value = {
        "schemaVersion": 1,
        "backend": selected,
        "maxCandidates": int(settings.ball_detection_max_candidates),
        "analysisFrameRate": float(settings.ball_analysis_frame_rate),
        "failurePolicy": str(settings.ball_detection_failure_policy),
    }
    if selected == "generic-ultralytics":
        value.update(
            {key: item for key, item in generic_input.items() if key != "backend"}
        )
    elif selected == "dedicated-ultralytics":
        value.update(
            {key: item for key, item in dedicated_input.items() if key != "backend"}
        )
        if str(settings.ball_detection_failure_policy) == "fallback":
            value["fallback"] = generic_input
    else:
        value.update(
            {
                "workerEndpoint": settings.ball_wasb_worker_url,
                "timeoutSeconds": float(settings.ball_wasb_timeout),
                "temporalWindowFrames": 3,
                "temporalContext": "previous-current-next",
                "fallback": (
                    dedicated_input
                    if str(settings.ball_detection_failure_policy) == "fallback"
                    else None
                ),
            }
        )
    return value


__all__ = (
    "BALL_DETECTION_BACKENDS",
    "ball_checkpoint_identity",
    "ball_detection_input",
    "verify_queued_ball_checkpoint",
)
