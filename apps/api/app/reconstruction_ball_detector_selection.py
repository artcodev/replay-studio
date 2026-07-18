from __future__ import annotations

"""Construct the detector pair requested by one immutable reconstruction run."""

from copy import deepcopy

from .ball_detection_configuration import (
    ball_detection_input,
    verify_queued_ball_checkpoint,
)
from .ball_detection_contract import BallDetector, BallDetectorConfig
from .ball_detector_factory import build_ball_detector
from .config import get_settings
from .person_detection_policy import (
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import load_model


def configured_ball_detectors(
    person_model: object,
    backend: str,
    detection_input: dict | None = None,
) -> tuple[BallDetector, BallDetector | None]:
    """Build the requested detector and its explicit failure-policy fallback."""

    settings = get_settings()
    contract = (
        deepcopy(detection_input)
        if isinstance(detection_input, dict)
        else ball_detection_input(backend)
    )
    if str(contract.get("backend") or "") != backend:
        raise ReconstructionError(
            "Queued ball detector input does not match its requested backend"
        )
    policy = str(
        contract.get("failurePolicy") or settings.ball_detection_failure_policy
    )
    if policy not in {"raise", "fallback"}:
        raise ReconstructionError(
            "BALL_DETECTION_FAILURE_POLICY must be raise or fallback"
        )
    max_candidates = int(
        contract.get("maxCandidates", settings.ball_detection_max_candidates)
    )
    fallback_contract = contract.get("fallback")
    if not isinstance(fallback_contract, dict):
        fallback_contract = {}
    generic_contract = (
        contract
        if backend == "generic-ultralytics"
        else fallback_contract
        if fallback_contract.get("backend") == "generic-ultralytics"
        else {}
    )
    generic = build_ball_detector(
        BallDetectorConfig(
            backend="generic-ultralytics",
            device=settings.reconstruction_device,
            confidence=float(
                generic_contract.get(
                    "confidence", GENERIC_ULTRALYTICS_CONFIDENCE
                )
            ),
            image_size=int(
                generic_contract.get("imageSize", GENERIC_ULTRALYTICS_IMAGE_SIZE)
            ),
            max_candidates=max_candidates,
            nms_iou=float(
                generic_contract.get("nmsIou", settings.ball_detection_nms_iou)
            ),
        ),
        model=person_model,
    )
    if backend == "generic-ultralytics":
        return generic, None

    dedicated: BallDetector | None = None
    if backend == "dedicated-ultralytics" or policy == "fallback":
        dedicated_contract = (
            contract if backend == "dedicated-ultralytics" else fallback_contract
        )
        verify_queued_ball_checkpoint(
            settings.ball_detection_model,
            dedicated_contract.get("checkpoint"),
        )
        dedicated = build_ball_detector(
            BallDetectorConfig(
                backend="dedicated-ultralytics",
                checkpoint_path=settings.ball_detection_model,
                device=settings.reconstruction_device,
                confidence=float(
                    dedicated_contract.get(
                        "confidence", settings.ball_detection_confidence
                    )
                ),
                image_size=int(
                    dedicated_contract.get(
                        "imageSize", settings.ball_detection_image_size
                    )
                ),
                max_candidates=max_candidates,
                tile_size=(
                    int(
                        dedicated_contract.get(
                            "tileSize", settings.ball_detection_tile_size
                        )
                    ),
                    int(
                        dedicated_contract.get(
                            "tileSize", settings.ball_detection_tile_size
                        )
                    ),
                ),
                tile_overlap=float(
                    dedicated_contract.get(
                        "tileOverlap", settings.ball_detection_tile_overlap
                    )
                ),
                inference_batch_size=int(
                    dedicated_contract.get(
                        "inferenceBatchSize",
                        settings.ball_detection_inference_batch_size,
                    )
                ),
                nms_iou=float(
                    dedicated_contract.get(
                        "nmsIou", settings.ball_detection_nms_iou
                    )
                ),
            ),
            model=load_model(settings.ball_detection_model),
        )
    if backend == "dedicated-ultralytics":
        assert dedicated is not None
        return dedicated, generic if policy == "fallback" else None
    if backend == "wasb-service":
        # The service adapter stays strict. Reconstruction owns the fallback
        # and circuit breaker, so a worker outage is reported once rather than
        # producing one long timeout per dense frame.
        detector = build_ball_detector(
            BallDetectorConfig(
                backend="wasb-service",
                device=settings.reconstruction_device,
                max_candidates=max_candidates,
                nms_iou=float(
                    contract.get("nmsIou", settings.ball_detection_nms_iou)
                ),
                wasb_service_url=(
                    contract.get("workerEndpoint") or settings.ball_wasb_worker_url
                ),
                wasb_timeout=float(
                    contract.get("timeoutSeconds", settings.ball_wasb_timeout)
                ),
                failure_policy="raise",
            ),
        )
        return detector, dedicated if policy == "fallback" else None
    raise ReconstructionError(f"Unsupported ball detection backend: {backend}")


__all__ = ("configured_ball_detectors",)
