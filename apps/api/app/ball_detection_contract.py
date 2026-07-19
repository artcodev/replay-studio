from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np


FrameInput: TypeAlias = str | Path | np.ndarray
FailurePolicy: TypeAlias = Literal["raise", "fallback"]
BackendName: TypeAlias = Literal[
    "generic-ultralytics",
    "dedicated-ultralytics",
    "wasb-service",
]

# Evidence backend recorded when every detector failed and only sampled
# generic COCO candidates remain for a dense frame.
GENERIC_FALLBACK_BACKEND = "generic-coco-fallback"


class BallDetectionError(RuntimeError):
    """Base error for the ball detector boundary."""


class BallDetectorConfigurationError(BallDetectionError):
    """Raised when a detector cannot be constructed safely."""


class BallDetectorUnavailable(BallDetectionError):
    """Raised when an external detector cannot produce a result."""


@dataclass(frozen=True, slots=True)
class BallCandidate:
    """One full-frame image-space ball hypothesis."""

    bbox: tuple[float, float, float, float]
    confidence: float
    backend: str
    class_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def as_reconstruction_detection(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "detectorBackend": self.backend,
            "detectorClassId": self.class_id,
            "detectorMetadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BallDetectionBatch:
    candidates: tuple[BallCandidate, ...]
    image_size: tuple[int, int]
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_reconstruction_detections(self) -> list[dict[str, Any]]:
        return [
            candidate.as_reconstruction_detection()
            for candidate in self.candidates
        ]


class BallDetector(Protocol):
    backend_name: str

    def detect(
        self,
        frame: FrameInput,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch: ...


@dataclass(frozen=True, slots=True)
class UltralyticsBallDetectorConfig:
    backend_name: str = "generic-ultralytics"
    class_ids: tuple[int, ...] | None = (32,)
    confidence: float = 0.035
    image_size: int | None = 1280
    device: str | int | None = "cpu"
    max_candidates: int = 12
    tile_size: tuple[int, int] | None = None
    tile_overlap: float = 0.2
    inference_batch_size: int = 4
    nms_iou: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise BallDetectorConfigurationError(
                "confidence must be between 0 and 1"
            )
        if self.image_size is not None and self.image_size <= 0:
            raise BallDetectorConfigurationError("image_size must be positive")
        if self.max_candidates <= 0:
            raise BallDetectorConfigurationError(
                "max_candidates must be positive"
            )
        if self.tile_size is not None and any(
            value <= 0 for value in self.tile_size
        ):
            raise BallDetectorConfigurationError(
                "tile dimensions must be positive"
            )
        if not 0.0 <= self.tile_overlap < 1.0:
            raise BallDetectorConfigurationError("tile_overlap must be in [0, 1)")
        if self.inference_batch_size <= 0:
            raise BallDetectorConfigurationError(
                "inference_batch_size must be positive"
            )
        if not 0.0 <= self.nms_iou <= 1.0:
            raise BallDetectorConfigurationError(
                "nms_iou must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class BallDetectorConfig:
    """Provider-neutral detector factory settings."""

    backend: BackendName = "generic-ultralytics"
    checkpoint_path: str | Path | None = None
    device: str | int | None = "cpu"
    confidence: float = 0.035
    image_size: int | None = None
    max_candidates: int = 12
    tile_size: tuple[int, int] = (640, 640)
    tile_overlap: float = 0.2
    inference_batch_size: int = 4
    nms_iou: float = 0.1
    wasb_service_url: str | None = None
    wasb_timeout: float = 30.0
    failure_policy: FailurePolicy = "raise"
