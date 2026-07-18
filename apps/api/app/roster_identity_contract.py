"""Provider-neutral contracts for closed-set roster identity review."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Mapping, Sequence

from .jersey_ocr_contract import normalize_jersey_number
from .roster_identity_temporal import TimeInterval, merge_intervals


ResolutionStatus = Literal["confirmed", "suggested", "abstain"]
ProposalStatus = Literal[
    "confirmed",
    "selected",
    "alternative",
    "ambiguous",
    "blocked",
]


def _identifier(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _probability(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return result


def _optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


@dataclass(frozen=True)
class AttributeEvidence:
    """A fused canonical attribute such as team, role, or shirt number."""

    value: str | int
    confidence: float
    source: str
    support_count: int = 1
    confirmed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("AttributeEvidence.value must not be empty")
        if isinstance(self.value, bool):
            raise ValueError("AttributeEvidence.value must not be boolean")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(
            self, "source", _identifier(self.source, "AttributeEvidence.source")
        )
        if int(self.support_count) < 1:
            raise ValueError("AttributeEvidence.support_count must be positive")
        object.__setattr__(self, "support_count", int(self.support_count))


@dataclass(frozen=True)
class ParticipationEvidence:
    """Player-specific match-event evidence shared by roster and video analysis."""

    event_id: str
    kind: str
    match_time_seconds: float | None = None
    confidence: float = 1.0
    source: str = "match-event"

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind").casefold())
        if self.match_time_seconds is not None:
            value = float(self.match_time_seconds)
            if not isfinite(value) or value < 0.0:
                raise ValueError("match_time_seconds must be finite and non-negative")
            object.__setattr__(self, "match_time_seconds", value)
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "ParticipationEvidence.confidence"),
        )
        object.__setattr__(self, "source", _identifier(self.source, "source").casefold())


def _validated_participation(
    values: Sequence[ParticipationEvidence], field_name: str
) -> tuple[ParticipationEvidence, ...]:
    rows = tuple(values)
    keys = [(item.source, item.event_id) for item in rows]
    duplicates = sorted(
        f"{source}:{event_id}"
        for source, event_id in set(keys)
        if keys.count((source, event_id)) > 1
    )
    if duplicates:
        raise ValueError(
            f"{field_name} contains duplicate namespaced event IDs: "
            + ", ".join(duplicates)
        )
    return rows


@dataclass(frozen=True)
class PlayerLikelihoodEvidence:
    """A provider-neutral player score (face, name OCR, gallery ReID)."""

    external_player_id: str
    confidence: float
    source: str
    evidence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_player_id",
            _identifier(self.external_player_id, "external_player_id"),
        )
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "PlayerLikelihoodEvidence.confidence"),
        )
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )


@dataclass(frozen=True)
class CanonicalPersonEvidence:
    canonical_person_id: str
    visible_intervals: Sequence[TimeInterval] = field(default_factory=tuple)
    team: AttributeEvidence | None = None
    role: AttributeEvidence | None = None
    jersey_number: AttributeEvidence | None = None
    participation: Sequence[ParticipationEvidence] = field(default_factory=tuple)
    player_likelihoods: Sequence[PlayerLikelihoodEvidence] = field(default_factory=tuple)
    confirmed_external_player_id: str | None = None
    excluded_external_player_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_person_id",
            _identifier(self.canonical_person_id, "canonical_person_id"),
        )
        object.__setattr__(
            self, "visible_intervals", merge_intervals(self.visible_intervals)
        )
        object.__setattr__(
            self,
            "participation",
            _validated_participation(
                self.participation, "CanonicalPersonEvidence.participation"
            ),
        )
        object.__setattr__(self, "player_likelihoods", tuple(self.player_likelihoods))
        confirmed = _optional_identifier(self.confirmed_external_player_id)
        excluded = tuple(
            sorted(
                {
                    _identifier(value, "excluded_external_player_ids")
                    for value in self.excluded_external_player_ids
                }
            )
        )
        if confirmed is not None and confirmed in excluded:
            raise ValueError("A confirmed player cannot also be manually excluded")
        if self.jersey_number is not None and normalize_jersey_number(
            self.jersey_number.value
        ) is None:
            raise ValueError("Canonical jersey-number evidence is invalid")
        object.__setattr__(self, "confirmed_external_player_id", confirmed)
        object.__setattr__(self, "excluded_external_player_ids", excluded)


@dataclass(frozen=True)
class PersistedRosterPlayer:
    external_player_id: str
    display_name: str
    team_id: str | None = None
    jersey_number: str | int | None = None
    role: str | None = None
    active_intervals: Sequence[TimeInterval] = field(default_factory=tuple)
    participation: Sequence[ParticipationEvidence] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_player_id",
            _identifier(self.external_player_id, "external_player_id"),
        )
        object.__setattr__(
            self, "display_name", _identifier(self.display_name, "display_name")
        )
        object.__setattr__(self, "team_id", _optional_identifier(self.team_id))
        object.__setattr__(self, "role", _optional_identifier(self.role))
        if self.jersey_number is not None:
            number = normalize_jersey_number(self.jersey_number)
            if number is None:
                raise ValueError("Persisted roster jersey_number is invalid")
            object.__setattr__(self, "jersey_number", number)
        object.__setattr__(
            self, "active_intervals", merge_intervals(self.active_intervals)
        )
        object.__setattr__(
            self,
            "participation",
            _validated_participation(
                self.participation, "PersistedRosterPlayer.participation"
            ),
        )


@dataclass(frozen=True)
class RosterResolverConfig:
    team_weight: float = 0.14
    jersey_weight: float = 0.48
    role_weight: float = 0.10
    availability_weight: float = 0.08
    participation_weight: float = 0.32
    direct_player_weight: float = 0.62
    soft_team_penalty: float = 0.12
    soft_jersey_penalty: float = 0.18
    soft_role_penalty: float = 0.08
    hard_team_confidence: float = 0.85
    hard_jersey_confidence: float = 0.80
    hard_jersey_support_count: int = 2
    hard_role_confidence: float = 0.90
    hard_direct_player_confidence: float = 0.93
    direct_player_gate_margin: float = 0.10
    minimum_active_coverage: float = 0.98
    availability_tolerance_seconds: float = 2.0
    event_time_tolerance_seconds: float = 3.0
    event_visibility_tolerance_seconds: float = 3.0
    min_candidate_score: float = 0.52
    min_identity_signal_score: float = 0.30
    assignment_margin: float = 0.05
    candidate_limit: int = 10

    def __post_init__(self) -> None:
        non_negative = (
            "team_weight",
            "jersey_weight",
            "role_weight",
            "availability_weight",
            "participation_weight",
            "direct_player_weight",
            "soft_team_penalty",
            "soft_jersey_penalty",
            "soft_role_penalty",
            "hard_team_confidence",
            "hard_jersey_confidence",
            "hard_role_confidence",
            "hard_direct_player_confidence",
            "direct_player_gate_margin",
            "minimum_active_coverage",
            "availability_tolerance_seconds",
            "event_time_tolerance_seconds",
            "event_visibility_tolerance_seconds",
            "min_candidate_score",
            "min_identity_signal_score",
            "assignment_margin",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        probabilities = (
            "hard_team_confidence",
            "hard_jersey_confidence",
            "hard_role_confidence",
            "hard_direct_player_confidence",
            "direct_player_gate_margin",
            "minimum_active_coverage",
            "min_candidate_score",
            "min_identity_signal_score",
            "assignment_margin",
        )
        if any(float(getattr(self, name)) > 1.0 for name in probabilities):
            raise ValueError("probability and margin thresholds must not exceed 1")
        if int(self.hard_jersey_support_count) < 1 or int(self.candidate_limit) < 1:
            raise ValueError("support count and candidate limit must be positive")


@dataclass(frozen=True)
class CandidateEvidence:
    code: str
    score_delta: float
    confidence: float
    source: str
    details: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        return {
            "code": self.code,
            "scoreDelta": round(self.score_delta, 6),
            "confidence": round(self.confidence, 6),
            "source": self.source,
            "details": list(self.details),
        }


@dataclass(frozen=True)
class RosterIdentityCandidate:
    external_player_id: str
    display_name: str
    team_id: str | None
    jersey_number: str | None
    role: str | None
    score: float
    identity_signal_score: float
    evidence: tuple[CandidateEvidence, ...]
    reasons: tuple[str, ...]
    conflicts: tuple[str, ...]
    eligible: bool
    proposal_status: ProposalStatus = "alternative"
    rank: int = 0

    @property
    def requires_manual_confirmation(self) -> bool:
        return self.eligible and self.proposal_status in {
            "selected",
            "alternative",
            "ambiguous",
        }

    def to_payload(self) -> dict:
        return {
            "rank": self.rank,
            "externalPlayerId": self.external_player_id,
            "name": self.display_name,
            "teamId": self.team_id,
            "number": self.jersey_number,
            "position": self.role,
            "score": round(self.score, 6),
            "identitySignalScore": round(self.identity_signal_score, 6),
            "eligible": self.eligible,
            "proposalStatus": self.proposal_status,
            "evidence": [item.to_payload() for item in self.evidence],
            "reasons": list(self.reasons),
            "conflicts": list(self.conflicts),
            "requiresManualConfirmation": self.requires_manual_confirmation,
        }


@dataclass(frozen=True)
class PersonRosterResolution:
    canonical_person_id: str
    status: ResolutionStatus
    confirmed_external_player_id: str | None
    suggested_external_player_id: str | None
    candidates: tuple[RosterIdentityCandidate, ...]
    reasons: tuple[str, ...]
    conflicts: tuple[str, ...]

    def to_payload(self) -> dict:
        return {
            "canonicalPersonId": self.canonical_person_id,
            "status": self.status,
            "confirmedExternalPlayerId": self.confirmed_external_player_id,
            "suggestedExternalPlayerId": self.suggested_external_player_id,
            "requiresManualConfirmation": self.status == "suggested",
            "reasons": list(self.reasons),
            "conflicts": list(self.conflicts),
            "candidates": [item.to_payload() for item in self.candidates],
        }


@dataclass(frozen=True)
class RosterResolutionConflict:
    code: str
    message: str
    canonical_person_ids: tuple[str, ...]
    external_player_id: str | None = None

    def to_payload(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "canonicalPersonIds": list(self.canonical_person_ids),
            "externalPlayerId": self.external_player_id,
        }


@dataclass(frozen=True)
class ClosedSetRosterResolution:
    people: tuple[PersonRosterResolution, ...]
    conflicts: tuple[RosterResolutionConflict, ...]
    diagnostics: Mapping[str, object]

    def to_payload(self) -> dict:
        return {
            "schemaVersion": 1,
            "people": [item.to_payload() for item in self.people],
            "conflicts": [item.to_payload() for item in self.conflicts],
            "diagnostics": dict(self.diagnostics),
        }
