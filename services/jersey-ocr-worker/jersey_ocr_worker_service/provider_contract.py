from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class ProviderUnavailable(RuntimeError):
    """The configured OCR runtime or its model assets are unavailable."""


@dataclass(frozen=True, slots=True)
class OcrSample:
    crop_id: str
    image_rgb: np.ndarray
    tracklet_id: str | None = None
    observation_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class RawTextCandidate:
    text: str
    confidence: float
    polygon: list[list[float]] | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    crop_id: str
    candidates: tuple[RawTextCandidate, ...]


class JerseyOcrProvider(Protocol):
    backend: str

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> dict[str, Any]: ...

    def recognize(self, samples: Sequence[OcrSample]) -> list[OcrResult]: ...
