from __future__ import annotations

"""Provider-neutral raw object-detection boundary for one source frame."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class RawDetectionBox:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class RawFramePrediction:
    image_bgr: np.ndarray
    names: dict[int, str]
    boxes: tuple[RawDetectionBox, ...]
    diagnostics: dict = field(default_factory=dict)


class PersonDetectionProvider(Protocol):
    def info(self) -> dict:
        """Return immutable model/runtime provenance used by the cache."""

    def predict(self, path: Path) -> RawFramePrediction:
        """Run raw detection without applying football-specific postprocess."""

