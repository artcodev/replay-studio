"""Provider-neutral contracts for jersey OCR evidence and fusion results."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Mapping


JerseyEvidenceScope = Literal["tracklet", "canonical"]
JerseyEvidenceStatus = Literal["reliable", "provisional", "conflict", "no-evidence"]


def normalize_jersey_number(
    value: str | int | None,
    *,
    minimum: int = 0,
    maximum: int = 99,
) -> str | None:
    """Return a canonical decimal jersey number, or ``None`` when unsafe."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if text.startswith("#"):
            text = text[1:].strip()
        if not text or not text.isascii() or not text.isdigit():
            return None
        number = int(text)
    if number < minimum or number > maximum:
        return None
    return str(number)


def probability(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return result


def identifier(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True)
class JerseyOcrObservation:
    """One OCR reading from one crop of a local person tracklet."""

    id: str
    tracklet_id: str
    timestamp_seconds: float
    raw_number: str | int | None
    ocr_confidence: float
    frame_quality: float = 1.0
    back_visibility: float = 1.0
    frame_index: int | None = None
    source: str = "jersey-ocr"
    evidence_fingerprint: str | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_seconds)
        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        frame_index = None if self.frame_index is None else int(self.frame_index)
        if frame_index is not None and frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        object.__setattr__(self, "id", identifier(self.id, "JerseyOcrObservation.id"))
        object.__setattr__(
            self,
            "tracklet_id",
            identifier(self.tracklet_id, "JerseyOcrObservation.tracklet_id"),
        )
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(
            self,
            "ocr_confidence",
            probability(self.ocr_confidence, "ocr_confidence"),
        )
        object.__setattr__(
            self,
            "frame_quality",
            probability(self.frame_quality, "frame_quality"),
        )
        object.__setattr__(
            self,
            "back_visibility",
            probability(self.back_visibility, "back_visibility"),
        )
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "source", str(self.source).strip() or "jersey-ocr")
        fingerprint = self.evidence_fingerprint
        if fingerprint is not None:
            if (
                not isinstance(fingerprint, str)
                or not fingerprint
                or len(fingerprint) > 160
                or not fingerprint.isascii()
                or any(character.isspace() for character in fingerprint)
            ):
                raise ValueError("evidence_fingerprint must be a stable opaque string")
        object.__setattr__(self, "evidence_fingerprint", fingerprint)

    @property
    def effective_score(self) -> float:
        return self.ocr_confidence * self.frame_quality * self.back_visibility


@dataclass(frozen=True)
class JerseyFusionConfig:
    """Thresholds for conservative sampling and evidence publication."""

    sampling_window_seconds: float = 0.50
    min_ocr_confidence: float = 0.55
    min_frame_quality: float = 0.35
    min_back_visibility: float = 0.35
    min_effective_score: float = 0.30
    max_selected_frames: int = 16
    reliable_confidence: float = 0.80
    reliable_support_count: int = 2
    conflict_vote_share: float = 0.34
    minimum_vote_margin: float = 0.20
    minimum_number: int = 0
    maximum_number: int = 99

    def __post_init__(self) -> None:
        if not isfinite(float(self.sampling_window_seconds)) or self.sampling_window_seconds <= 0:
            raise ValueError("sampling_window_seconds must be finite and positive")
        for name in (
            "min_ocr_confidence",
            "min_frame_quality",
            "min_back_visibility",
            "min_effective_score",
            "reliable_confidence",
            "conflict_vote_share",
            "minimum_vote_margin",
        ):
            probability(getattr(self, name), name)
        if int(self.max_selected_frames) < 1:
            raise ValueError("max_selected_frames must be positive")
        if int(self.reliable_support_count) < 1:
            raise ValueError("reliable_support_count must be positive")
        if int(self.minimum_number) < 0 or int(self.maximum_number) < int(
            self.minimum_number
        ):
            raise ValueError("jersey number range is invalid")


@dataclass(frozen=True)
class JerseyVote:
    number: str
    support_count: int
    weight: float
    weight_share: float
    mean_effective_score: float
    observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class JerseyEvidenceSummary:
    """Fail-closed fused evidence for one tracklet or canonical person."""

    subject_id: str
    scope: JerseyEvidenceScope
    status: JerseyEvidenceStatus
    jersey_number: str | None
    candidate_number: str | None
    confidence: float
    support_count: int
    selected_sample_count: int
    selected_observations: tuple[JerseyOcrObservation, ...]
    votes: tuple[JerseyVote, ...]
    tracklet_ids: tuple[str, ...]
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    conflict_reasons: tuple[str, ...] = ()

    def identity_resolver_fields(self) -> dict[str, str | float | int | None]:
        if self.jersey_number is None:
            return {
                "jersey_number": None,
                "jersey_confidence": 0.0,
                "jersey_sample_count": 0,
            }
        return {
            "jersey_number": self.jersey_number,
            "jersey_confidence": self.confidence,
            "jersey_sample_count": self.support_count,
        }

    def to_payload(self) -> dict:
        return {
            "subjectId": self.subject_id,
            "scope": self.scope,
            "status": self.status,
            "jerseyNumber": self.jersey_number,
            "candidateNumber": self.candidate_number,
            "confidence": round(self.confidence, 6),
            "supportCount": self.support_count,
            "selectedSampleCount": self.selected_sample_count,
            "selectedObservationIds": [item.id for item in self.selected_observations],
            "trackletIds": list(self.tracklet_ids),
            "votes": [
                {
                    "number": item.number,
                    "supportCount": item.support_count,
                    "weight": round(item.weight, 6),
                    "weightShare": round(item.weight_share, 6),
                    "meanEffectiveScore": round(item.mean_effective_score, 6),
                    "observationIds": list(item.observation_ids),
                }
                for item in self.votes
            ],
            "rejectionCounts": dict(sorted(self.rejection_counts.items())),
            "conflictReasons": list(self.conflict_reasons),
        }
