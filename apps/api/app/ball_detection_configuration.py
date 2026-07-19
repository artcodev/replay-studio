from __future__ import annotations

"""Immutable reconstruction inputs for selecting a ball detector backend."""

from pathlib import Path

from .checkpoint_identity import checkpoint_content_identity
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
    return checkpoint_content_identity(path)


def verify_queued_ball_checkpoint(path: str | Path, expected: object) -> None:
    if not isinstance(expected, dict):
        raise ReconstructionError(
            "Queued ball checkpoint identity is missing; "
            "queue a new reconstruction run."
        )
    if not expected.get("sha256"):
        raise ReconstructionError(
            "Queued ball checkpoint identity has no content hash; "
            "queue a new reconstruction run."
        )
    actual = ball_checkpoint_identity(path)
    mismatches = [
        key
        for key in ("name", "size", "sha256")
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
    failure_policy = str(settings.ball_detection_failure_policy)
    requires_dedicated_checkpoint = selected == "dedicated-ultralytics" or (
        selected == "wasb-service" and failure_policy == "fallback"
    )
    checkpoint = (
        ball_checkpoint_identity(settings.ball_detection_model)
        if requires_dedicated_checkpoint
        else {}
    )
    if requires_dedicated_checkpoint and not checkpoint.get("sha256"):
        raise ReconstructionError(
            "Required ball detection checkpoint does not exist: "
            f"{Path(settings.ball_detection_model).expanduser().resolve()}"
        )
    dedicated_input = {
        "backend": "dedicated-ultralytics",
        "checkpoint": checkpoint,
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
        "failurePolicy": failure_policy,
    }
    if selected == "generic-ultralytics":
        value.update(
            {key: item for key, item in generic_input.items() if key != "backend"}
        )
    elif selected == "dedicated-ultralytics":
        value.update(
            {key: item for key, item in dedicated_input.items() if key != "backend"}
        )
        if failure_policy == "fallback":
            value["fallback"] = generic_input
    else:
        wasb_transport = str(settings.ball_wasb_transport)
        if wasb_transport not in {"per-frame-window", "batched-sequence"}:
            raise ReconstructionError(
                f"Unsupported WASB transport: {wasb_transport}"
            )
        value.update(
            {
                "workerEndpoint": settings.ball_wasb_worker_url,
                "timeoutSeconds": float(settings.ball_wasb_timeout),
                "temporalWindowFrames": 3,
                # "per-frame-window" centers a fresh (prev, current, next)
                # window on every dense frame; "batched-sequence" sends runs
                # of frames in one request and accepts the worker's fixed
                # window tiling at run boundaries in exchange for ~3x fewer
                # uploads and inferences. The choice is part of the cache
                # contract, so switching transports never reuses evidence
                # produced by the other windowing.
                "temporalContext": (
                    "previous-current-next"
                    if wasb_transport == "per-frame-window"
                    else "tiled-window-sequence"
                ),
                "wasbTransport": wasb_transport,
                **(
                    {"wasbBatchSize": int(settings.ball_wasb_batch_size)}
                    if wasb_transport == "batched-sequence"
                    else {}
                ),
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
