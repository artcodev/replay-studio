from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ball_detection_contract import (
    BallDetector,
    BallDetectorConfig,
    BallDetectorConfigurationError,
    UltralyticsBallDetectorConfig,
)
from .ultralytics_ball_detector import UltralyticsBallDetector
from .wasb_ball_detector import WasbServiceBallDetector, WasbServiceTransport


def build_ball_detector(
    config: BallDetectorConfig,
    *,
    model: Any | None = None,
    fallback: BallDetector | None = None,
    model_loader: Callable[[str], Any] | None = None,
    service_transport: WasbServiceTransport | None = None,
) -> BallDetector:
    """Build one configured backend without implicit downloads or fallback."""

    if config.backend in ("generic-ultralytics", "dedicated-ultralytics"):
        dedicated = config.backend == "dedicated-ultralytics"
        detector_config = UltralyticsBallDetectorConfig(
            backend_name=config.backend,
            class_ids=(0,) if dedicated else (32,),
            confidence=config.confidence,
            image_size=(
                config.image_size
                if config.image_size is not None
                else (640 if dedicated else 1280)
            ),
            device=config.device,
            max_candidates=config.max_candidates,
            tile_size=config.tile_size if dedicated else None,
            tile_overlap=config.tile_overlap,
            inference_batch_size=config.inference_batch_size,
            nms_iou=config.nms_iou,
        )
        return UltralyticsBallDetector(
            detector_config,
            model=model,
            checkpoint_path=config.checkpoint_path,
            model_loader=model_loader,
        )
    if config.backend == "wasb-service":
        if config.wasb_service_url is None:
            raise BallDetectorConfigurationError(
                "wasb_service_url is required"
            )
        return WasbServiceBallDetector(
            config.wasb_service_url,
            timeout=config.wasb_timeout,
            max_candidates=config.max_candidates,
            nms_iou=config.nms_iou,
            failure_policy=config.failure_policy,
            fallback=fallback,
            transport=service_transport,
        )
    raise BallDetectorConfigurationError(
        f"unsupported ball detector backend: {config.backend}"
    )
