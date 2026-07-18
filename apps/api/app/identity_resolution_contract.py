"""Provider-neutral contracts for offline canonical identity resolution.

The contracts validate and normalize evidence at the subsystem boundary.  They
do not select candidate links, solve assignments, or know about reconstruction
and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Sequence

import numpy as np


EdgeStatus = Literal["accepted", "review", "rejected"]
GroupStatus = Literal["resolved", "provisional", "excluded"]


def normalize_reid_embedding(
    value: Sequence[float] | np.ndarray | None,
) -> tuple[float, ...] | None:
    """Return a finite unit embedding, or ``None`` for unusable evidence."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return None
    return tuple(float(item) for item in array / norm)


def _normalized_jersey(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(int(text)) if text.isdigit() else text.upper()


def _pitch_position(
    value: Sequence[float] | None,
    field_name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain finite coordinates")
    return result


@dataclass(frozen=True)
class IdentityTracklet:
    """Immutable evidence summary for one local online-tracker trajectory."""

    id: str
    start_time: float
    end_time: float
    team_id: str | None = None
    role: str | None = None
    external_player_id: str | None = None
    jersey_number: str | int | None = None
    jersey_confidence: float = 0.0
    jersey_sample_count: int = 0
    mean_reid_embedding: Sequence[float] | np.ndarray | None = None
    reid_embeddings: Sequence[Sequence[float] | np.ndarray] = field(default_factory=tuple)
    start_pitch: Sequence[float] | None = None
    end_pitch: Sequence[float] | None = None
    start_uncertainty_metres: float | None = None
    end_uncertainty_metres: float | None = None
    observation_count: int = 0
    manual_confirmed: bool = False
    manual_excluded: bool = False
    manual_identity_id: str | None = None
    manual_team: bool = False
    manual_role: bool = False
    manual_jersey: bool = False

    def __post_init__(self) -> None:
        identifier = str(self.id).strip()
        if not identifier:
            raise ValueError("IdentityTracklet.id must not be empty")
        start, end = float(self.start_time), float(self.end_time)
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError("IdentityTracklet times must be finite and end_time >= start_time")
        confidence = float(self.jersey_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("jersey_confidence must be between 0 and 1")
        if int(self.jersey_sample_count) < 0 or int(self.observation_count) < 0:
            raise ValueError("sample and observation counts must be non-negative")

        mean = normalize_reid_embedding(self.mean_reid_embedding)
        samples = tuple(
            item
            for value in self.reid_embeddings
            if (item := normalize_reid_embedding(value)) is not None
        )
        dimensions = {len(item) for item in samples}
        if mean is not None:
            dimensions.add(len(mean))
        if len(dimensions) > 1:
            raise ValueError("All ReID embeddings in one tracklet must have the same dimension")
        if mean is None and samples:
            mean = normalize_reid_embedding(
                np.mean(np.asarray(samples, dtype=np.float64), axis=0)
            )

        manual_identity_id = (
            str(self.manual_identity_id).strip()
            if self.manual_identity_id is not None
            else None
        )
        external_player_id = (
            str(self.external_player_id).strip()
            if self.external_player_id is not None
            else None
        )
        team_id = str(self.team_id).strip() if self.team_id is not None else None
        role = str(self.role).strip() if self.role is not None else None

        for value, name in (
            (self.start_uncertainty_metres, "start_uncertainty_metres"),
            (self.end_uncertainty_metres, "end_uncertainty_metres"),
        ):
            if value is not None and (not isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "team_id", team_id or None)
        object.__setattr__(self, "role", role or None)
        object.__setattr__(self, "external_player_id", external_player_id or None)
        object.__setattr__(self, "manual_identity_id", manual_identity_id or None)
        object.__setattr__(self, "jersey_number", _normalized_jersey(self.jersey_number))
        object.__setattr__(self, "jersey_confidence", confidence)
        object.__setattr__(self, "jersey_sample_count", int(self.jersey_sample_count))
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(self, "mean_reid_embedding", mean)
        object.__setattr__(self, "reid_embeddings", samples)
        object.__setattr__(self, "start_pitch", _pitch_position(self.start_pitch, "start_pitch"))
        object.__setattr__(self, "end_pitch", _pitch_position(self.end_pitch, "end_pitch"))
        object.__setattr__(
            self,
            "start_uncertainty_metres",
            float(self.start_uncertainty_metres)
            if self.start_uncertainty_metres is not None
            else None,
        )
        object.__setattr__(
            self,
            "end_uncertainty_metres",
            float(self.end_uncertainty_metres)
            if self.end_uncertainty_metres is not None
            else None,
        )


@dataclass(frozen=True)
class IdentityResolverConfig:
    reliable_jersey_confidence: float = 0.80
    reliable_jersey_samples: int = 2
    strong_reid_distance: float = 0.18
    review_reid_distance: float = 0.30
    strong_sample_reid_distance: float = 0.10
    min_strong_reid_samples: int = 2
    accept_score: float = 0.78
    ambiguity_margin: float = 0.08
    ambiguity_gap_penalty_per_second: float = 0.20
    ambiguity_gap_penalty_cap: float = 0.25
    max_player_speed_metres_per_second: float = 12.0
    motion_slack_metres: float = 2.0
    temporal_epsilon_seconds: float = 1e-6

    def __post_init__(self) -> None:
        probability_values = (
            self.reliable_jersey_confidence,
            self.strong_reid_distance,
            self.review_reid_distance,
            self.strong_sample_reid_distance,
            self.accept_score,
            self.ambiguity_margin,
            self.ambiguity_gap_penalty_per_second,
            self.ambiguity_gap_penalty_cap,
        )
        if any(not isfinite(float(item)) or float(item) < 0.0 for item in probability_values):
            raise ValueError("Identity resolver thresholds must be finite and non-negative")
        if self.strong_reid_distance > self.review_reid_distance:
            raise ValueError("strong_reid_distance must not exceed review_reid_distance")
        if not 0.0 <= self.accept_score <= 1.0:
            raise ValueError("accept_score must be between 0 and 1")
        if self.reliable_jersey_samples < 1:
            raise ValueError("reliable_jersey_samples must be positive")
        if self.min_strong_reid_samples < 2:
            raise ValueError("min_strong_reid_samples must be at least two")
        if self.max_player_speed_metres_per_second <= 0.0 or self.motion_slack_metres < 0.0:
            raise ValueError("Motion limits must be positive")


@dataclass(frozen=True)
class IdentityEdge:
    predecessor_id: str
    successor_id: str
    status: EdgeStatus
    score: float | None
    source: str
    reasons: tuple[str, ...] = ()
    gap_seconds: float | None = None
    reid_mean_distance: float | None = None
    reid_best_sample_distance: float | None = None
    reid_robust_sample_distance: float | None = None
    reid_strong_support_left: int = 0
    reid_strong_support_right: int = 0
    pitch_distance_metres: float | None = None
    reachable_distance_metres: float | None = None


@dataclass(frozen=True)
class IdentityGroup:
    id: str
    tracklet_ids: tuple[str, ...]
    status: GroupStatus
    confidence: float
    source: str
    team_id: str | None
    role: str | None
    external_player_id: str | None
    jersey_number: str | None
    manual_identity_id: str | None
    observation_count: int


@dataclass(frozen=True)
class IdentityResolution:
    groups: tuple[IdentityGroup, ...]
    accepted_edges: tuple[IdentityEdge, ...]
    review_edges: tuple[IdentityEdge, ...]
    rejected_edges: tuple[IdentityEdge, ...]
    diagnostics: dict
    tracklet_to_identity: dict[str, str]


__all__ = [
    "EdgeStatus",
    "GroupStatus",
    "IdentityEdge",
    "IdentityGroup",
    "IdentityResolution",
    "IdentityResolverConfig",
    "IdentityTracklet",
    "normalize_reid_embedding",
]
