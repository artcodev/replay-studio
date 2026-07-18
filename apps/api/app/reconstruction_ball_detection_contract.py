from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .ball_detection_contract import BallDetectionBatch


BallFrameDetections = list[tuple[list[dict], float]]
BallDetectionProgress = Callable[[int, int, str], None]
BallDetectionResult = tuple[BallFrameDetections, dict, list[dict], list[str]]


@dataclass(slots=True)
class DenseBallDetectionSource:
    frames: list[tuple[Path, float]]
    metadata: dict
    dense_cache_key: str | None
    cache_asset_directory: Path | None
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class BallDetectionAttempt:
    batch: BallDetectionBatch | None
    failure_detail: str | None
    circuit_reason: str | None


@dataclass(frozen=True, slots=True)
class BallFrameEvidence:
    detections: list[dict]
    backend: str
    metadata: dict
    image_size: tuple[int, int] | None
    detector_failed: bool
