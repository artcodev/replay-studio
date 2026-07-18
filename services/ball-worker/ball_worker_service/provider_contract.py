from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


class ProviderUnavailable(RuntimeError):
    """The configured ball model or its verified assets are unavailable."""


@dataclass(frozen=True, slots=True)
class BallCandidate:
    x: float
    y: float
    confidence: float
    heatmap_peak: float
    component_score: float
    component_area: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BallProviderInfo:
    backend: str
    model_version: str
    checkpoint_sha256: str | None
    device: str
    frames_in: int
    frames_out: int
    input_size: tuple[int, int]
    score_threshold: float
    model_load_seconds: float | None

    def to_wire(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "modelVersion": self.model_version,
            "checkpointSha256": self.checkpoint_sha256,
            "device": self.device,
            "framesIn": self.frames_in,
            "framesOut": self.frames_out,
            "inputSize": list(self.input_size),
            "scoreThreshold": self.score_threshold,
            "modelLoadSeconds": (
                round(self.model_load_seconds, 4)
                if self.model_load_seconds is not None
                else None
            ),
        }


class BallDetectionProvider(Protocol):
    backend: str
    frames_in: int
    frames_out: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> BallProviderInfo: ...

    def detect_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        max_candidates: int,
    ) -> list[list[BallCandidate]]: ...
