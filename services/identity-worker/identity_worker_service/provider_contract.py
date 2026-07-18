from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


EMBEDDING_DIMENSION = 256


class ProviderUnavailable(RuntimeError):
    """The configured ReID model is missing, invalid, or could not load."""


@dataclass(frozen=True)
class EmbeddingSample:
    observation_id: str
    image_rgb: np.ndarray


@dataclass(frozen=True)
class ProviderEmbedding:
    observation_id: str
    embedding: np.ndarray
    visibility_scores: np.ndarray | None = None
    role: str | None = None
    role_confidence: float | None = None


class IdentityEmbeddingProvider(Protocol):
    backend: str
    dimension: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> dict: ...

    def embed(self, samples: Sequence[EmbeddingSample]) -> list[ProviderEmbedding]: ...
