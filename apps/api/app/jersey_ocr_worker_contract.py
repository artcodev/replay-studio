from __future__ import annotations

"""Typed application contract for the optional jersey OCR worker."""

from dataclasses import dataclass, field
from pathlib import Path


CONTRACT_VERSION = "jersey-ocr.v1"
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
VALID_STATUSES = frozenset(
    {"recognized", "no-number", "low-confidence", "ambiguous", "rejected"}
)


class JerseyOcrWorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JerseyCropRequest:
    crop_id: str
    path: Path
    observation_id: str | None = None
    tracklet_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None


@dataclass(slots=True)
class JerseyOcrBatchResult:
    items_by_crop_id: dict[str, dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
