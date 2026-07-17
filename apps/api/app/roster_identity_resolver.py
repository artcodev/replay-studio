"""Closed-set roster suggestions for canonical football identities.

This module is deliberately pure and persistence-agnostic.  It receives the
canonical people already produced by tracking/ReID and a *persisted* match
roster, evaluates all player candidates, and solves one global assignment.
The output is review data only: there is no automatic binding field and no
mutation of either input.

The resolver separates three kinds of information:

* identity signals (repeated jersey OCR, player-specific event evidence, or a
  direct face/appearance candidate) may support a suggestion;
* team, role, and availability constrain or weight that suggestion; and
* a confirmed roster binding is a hard manual fact.  Contradicting evidence is
  reported, but never silently changes the binding.

Weak closed-set classifiers otherwise tend to choose the least-wrong roster
row even when the real person is unknown.  Here an explicit abstain column is
part of the global assignment and every suggestion requires both sufficient
identity evidence and a margin over the next global solution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import isfinite
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .jersey_ocr_fusion import normalize_jersey_number


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


@dataclass(frozen=True, order=True)
class TimeInterval:
    """One interval on the caller's normalized match clock, in seconds.

    Positive-duration intervals are treated as half-open ``[start, end)``.
    A zero-duration interval is a point observation and is considered visible
    when it lies inside the other interval.  The resolver never attempts to
    convert broadcast/source time into match time; that is an integration
    responsibility.
    """

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start, end = float(self.start_seconds), float(self.end_seconds)
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError("TimeInterval must be finite and end_seconds >= start_seconds")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_payload(self) -> dict:
        return {"startTime": self.start_seconds, "endTime": self.end_seconds}


def _merge_intervals(values: Iterable[TimeInterval]) -> tuple[TimeInterval, ...]:
    rows = sorted(tuple(values))
    if not rows:
        return ()
    merged: list[TimeInterval] = [rows[0]]
    for row in rows[1:]:
        previous = merged[-1]
        if previous.duration > 0.0 and row.duration > 0.0 and (
            row.start_seconds <= previous.end_seconds
        ):
            merged[-1] = TimeInterval(
                previous.start_seconds,
                max(previous.end_seconds, row.end_seconds),
            )
        elif previous.duration > 0.0 and row.duration == 0.0 and (
            previous.start_seconds <= row.start_seconds < previous.end_seconds
        ):
            # The point is already contained by the half-open span. A point at
            # ``previous.end_seconds`` is *not* contained and must be retained.
            continue
        elif previous.duration == 0.0 and row.duration > 0.0 and (
            row.start_seconds <= previous.start_seconds < row.end_seconds
        ):
            # Sorting can put a point before a span with the same start.
            merged[-1] = row
        elif (
            previous.duration == 0.0
            and row.duration == 0.0
            and row.start_seconds == previous.start_seconds
        ):
            continue
        else:
            merged.append(row)
    return tuple(merged)


def _intervals_overlap(
    left: Sequence[TimeInterval], right: Sequence[TimeInterval]
) -> bool:
    for first in left:
        for second in right:
            if first.duration == 0.0 and second.duration == 0.0:
                if first.start_seconds == second.start_seconds:
                    return True
            elif first.duration == 0.0:
                if second.start_seconds <= first.start_seconds < second.end_seconds:
                    return True
            elif second.duration == 0.0:
                if first.start_seconds <= second.start_seconds < first.end_seconds:
                    return True
            elif max(first.start_seconds, second.start_seconds) < min(
                first.end_seconds, second.end_seconds
            ):
                return True
    return False


def _expanded_intervals(
    values: Sequence[TimeInterval], tolerance_seconds: float
) -> tuple[TimeInterval, ...]:
    tolerance = max(0.0, float(tolerance_seconds))
    if tolerance == 0.0:
        return tuple(values)
    return _merge_intervals(
        TimeInterval(
            max(0.0, item.start_seconds - tolerance),
            item.end_seconds + tolerance,
        )
        for item in values
    )


def _point_inside_intervals(
    timestamp: float,
    intervals: Sequence[TimeInterval],
    tolerance_seconds: float,
) -> bool:
    point = TimeInterval(float(timestamp), float(timestamp))
    return _intervals_overlap(
        (point,), _expanded_intervals(intervals, tolerance_seconds)
    )


def _interval_coverage(
    observed: Sequence[TimeInterval], active: Sequence[TimeInterval]
) -> float:
    if not observed or not active:
        return 0.0
    coverage_parts: list[float] = []
    positive_duration = sum(item.duration for item in observed if item.duration > 0.0)
    if positive_duration > 0.0:
        overlap = 0.0
        for first in observed:
            if first.duration <= 0.0:
                continue
            for second in active:
                overlap += max(
                    0.0,
                    min(first.end_seconds, second.end_seconds)
                    - max(first.start_seconds, second.start_seconds),
                )
        coverage_parts.append(min(1.0, overlap / positive_duration))
    points = tuple(item for item in observed if item.duration == 0.0)
    if points:
        contained = sum(
            any(_intervals_overlap((point,), (candidate,)) for candidate in active)
            for point in points
        )
        coverage_parts.append(contained / len(points))
    return min(coverage_parts) if coverage_parts else 0.0


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
        object.__setattr__(self, "source", _identifier(self.source, "AttributeEvidence.source"))
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
        object.__setattr__(
            self, "source", _identifier(self.source, "source").casefold()
        )


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
    """A provider-neutral player-specific score (face, name OCR, gallery ReID)."""

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
            self, "visible_intervals", _merge_intervals(self.visible_intervals)
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
            self, "active_intervals", _merge_intervals(self.active_intervals)
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
        for name in (
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
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "hard_team_confidence",
            "hard_jersey_confidence",
            "hard_role_confidence",
            "hard_direct_player_confidence",
            "direct_player_gate_margin",
            "minimum_active_coverage",
            "min_candidate_score",
            "min_identity_signal_score",
            "assignment_margin",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must not exceed 1")
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

    @property
    def auto_bind_external_player_id(self) -> None:
        return None

    def to_payload(self) -> dict:
        return {
            "canonicalPersonId": self.canonical_person_id,
            "status": self.status,
            "confirmedExternalPlayerId": self.confirmed_external_player_id,
            "suggestedExternalPlayerId": self.suggested_external_player_id,
            "autoBindExternalPlayerId": None,
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

    @property
    def auto_bindings(self) -> tuple[()]:
        """An explicit empty contract: this module never authorizes writes."""

        return ()

    def to_payload(self) -> dict:
        return {
            "schemaVersion": 1,
            "people": [item.to_payload() for item in self.people],
            "conflicts": [item.to_payload() for item in self.conflicts],
            "autoBindings": [],
            "diagnostics": dict(self.diagnostics),
        }


def _role_family(value: str | int | None) -> str | None:
    if value is None:
        return None
    role = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    if not role:
        return None
    if role in {"gk", "goalkeeper", "goalie", "keeper"} or "goalkeeper" in role:
        return "goalkeeper"
    if role in {"ref", "referee", "official", "assistant referee"} or "referee" in role:
        return "official"
    if any(
        token in role
        for token in (
            "coach",
            "manager",
            "staff",
            "spectator",
            "supporter",
            "fan",
            "medical",
            "physio",
            "ball boy",
            "ball girl",
            "non player",
            "other person",
        )
    ) or role == "other":
        return "non-player"
    if any(
        token in role
        for token in (
            "player",
            "defender",
            "back",
            "midfield",
            "wing",
            "forward",
            "striker",
            "attacker",
            "sweeper",
        )
    ):
        return "outfield"
    # Unknown provider positions are not equivalent to an explicit coach,
    # official, or spectator classification.
    return None


def _attribute_value(evidence: AttributeEvidence | None) -> str | None:
    if evidence is None:
        return None
    return str(evidence.value).strip()


def _candidate_evidence(
    code: str,
    score_delta: float,
    confidence: float,
    source: str,
    *details: str,
) -> CandidateEvidence:
    return CandidateEvidence(
        code=code,
        score_delta=float(score_delta),
        confidence=float(confidence),
        source=source,
        details=tuple(details),
    )


def _evaluate_candidate(
    person: CanonicalPersonEvidence,
    player: PersistedRosterPlayer,
    config: RosterResolverConfig,
    *,
    confirmed_binding: bool = False,
) -> RosterIdentityCandidate:
    evidence: list[CandidateEvidence] = []
    reasons: list[str] = []
    conflicts: list[str] = []
    identity_signal_score = 0.0

    if confirmed_binding:
        evidence.append(
            _candidate_evidence(
                "manual-confirmed-binding", 1.0, 1.0, "manual-roster-binding"
            )
        )
        identity_signal_score = 1.0
        reasons.append("manual-confirmed-binding-authoritative")
    elif player.external_player_id in person.excluded_external_player_ids:
        conflicts.append("manually-excluded-player")

    team_value = _attribute_value(person.team)
    if team_value is not None and player.team_id is not None:
        confidence = person.team.confidence
        if team_value == player.team_id:
            evidence.append(
                _candidate_evidence(
                    "team-match",
                    config.team_weight * confidence,
                    confidence,
                    person.team.source,
                    player.team_id,
                )
            )
            reasons.append("team-match")
        elif person.team.confirmed or confidence >= config.hard_team_confidence:
            conflicts.append("team-mismatch-hard")
        else:
            evidence.append(
                _candidate_evidence(
                    "team-mismatch-soft",
                    -config.soft_team_penalty * confidence,
                    confidence,
                    person.team.source,
                    team_value,
                    player.team_id,
                )
            )
            reasons.append("team-mismatch-soft")
    elif team_value is not None:
        reasons.append("roster-team-unavailable")
    else:
        reasons.append("canonical-team-unavailable")

    canonical_number = (
        normalize_jersey_number(person.jersey_number.value)
        if person.jersey_number is not None
        else None
    )
    if canonical_number is not None and player.jersey_number is not None:
        jersey = person.jersey_number
        support_factor = 1.0 if jersey.confirmed else min(1.0, jersey.support_count / 2.0)
        effective_confidence = jersey.confidence * support_factor
        if canonical_number == player.jersey_number:
            delta = config.jersey_weight * effective_confidence
            evidence.append(
                _candidate_evidence(
                    "jersey-number-match",
                    delta,
                    jersey.confidence,
                    jersey.source,
                    canonical_number,
                    f"support:{jersey.support_count}",
                )
            )
            identity_signal_score += max(0.0, delta)
            reasons.append("jersey-number-match")
        elif jersey.confirmed or (
            jersey.confidence >= config.hard_jersey_confidence
            and jersey.support_count >= config.hard_jersey_support_count
        ):
            conflicts.append("jersey-number-mismatch-hard")
        else:
            evidence.append(
                _candidate_evidence(
                    "jersey-number-mismatch-soft",
                    -config.soft_jersey_penalty * effective_confidence,
                    jersey.confidence,
                    jersey.source,
                    canonical_number,
                    player.jersey_number,
                )
            )
            reasons.append("jersey-number-mismatch-soft")
    elif canonical_number is not None:
        reasons.append("roster-jersey-number-unavailable")
    else:
        reasons.append("canonical-jersey-number-unavailable")

    canonical_role = _role_family(person.role.value if person.role is not None else None)
    roster_role = _role_family(player.role)
    if (
        person.role is not None
        and person.role.confirmed
        and canonical_role in {"official", "non-player"}
    ):
        # PersistedRosterPlayer is a player-only closed set. An explicitly
        # confirmed referee/coach/spectator cannot become a roster player even
        # when that roster row has no position metadata.
        conflicts.append("confirmed-non-player-role")
    elif canonical_role is not None and roster_role is not None:
        confidence = person.role.confidence
        if canonical_role == roster_role:
            evidence.append(
                _candidate_evidence(
                    "role-match",
                    config.role_weight * confidence,
                    confidence,
                    person.role.source,
                    canonical_role,
                )
            )
            reasons.append("role-match")
        elif person.role.confirmed or confidence >= config.hard_role_confidence:
            conflicts.append("role-mismatch-hard")
        else:
            evidence.append(
                _candidate_evidence(
                    "role-mismatch-soft",
                    -config.soft_role_penalty * confidence,
                    confidence,
                    person.role.source,
                    canonical_role,
                    roster_role,
                )
            )
            reasons.append("role-mismatch-soft")
    elif canonical_role is not None:
        reasons.append("roster-role-unavailable")
    else:
        reasons.append("canonical-role-unavailable")

    if person.visible_intervals and player.active_intervals:
        active_with_tolerance = _expanded_intervals(
            player.active_intervals, config.availability_tolerance_seconds
        )
        coverage = _interval_coverage(person.visible_intervals, active_with_tolerance)
        if not _intervals_overlap(person.visible_intervals, active_with_tolerance):
            conflicts.append("player-inactive-at-observation-time")
        elif coverage < config.minimum_active_coverage:
            conflicts.append("player-not-active-for-full-visible-interval")
        else:
            evidence.append(
                _candidate_evidence(
                    "active-time-coverage",
                    config.availability_weight * coverage,
                    coverage,
                    "match-participation-window",
                    f"minimum:{config.minimum_active_coverage:.3f}",
                    f"tolerance-seconds:{config.availability_tolerance_seconds:.3f}",
                )
            )
            reasons.append("active-time-coverage")
    elif person.visible_intervals:
        reasons.append("roster-active-interval-unavailable")
    else:
        reasons.append("canonical-visible-interval-unavailable")

    canonical_events = {
        (item.source, item.event_id): item for item in person.participation
    }
    roster_events = {
        (item.source, item.event_id): item for item in player.participation
    }
    shared_event_keys = sorted(set(canonical_events) & set(roster_events))
    compatible_events: list[
        tuple[tuple[str, str], ParticipationEvidence, ParticipationEvidence]
    ] = []
    for event_key in shared_event_keys:
        canonical_event = canonical_events[event_key]
        roster_event = roster_events[event_key]
        if canonical_event.kind != roster_event.kind:
            conflicts.append("event-kind-mismatch")
            continue
        if (
            canonical_event.match_time_seconds is None
            or roster_event.match_time_seconds is None
        ):
            reasons.append("event-time-unavailable")
            continue
        if (
            abs(
                canonical_event.match_time_seconds
                - roster_event.match_time_seconds
            )
            > config.event_time_tolerance_seconds
        ):
            conflicts.append("event-time-mismatch")
            continue
        if not person.visible_intervals or not _point_inside_intervals(
            canonical_event.match_time_seconds,
            person.visible_intervals,
            config.event_visibility_tolerance_seconds,
        ):
            reasons.append("event-outside-canonical-visible-interval")
            continue
        if not player.active_intervals or not _point_inside_intervals(
            roster_event.match_time_seconds,
            player.active_intervals,
            config.availability_tolerance_seconds,
        ):
            conflicts.append("event-outside-player-active-interval")
            continue
        compatible_events.append((event_key, canonical_event, roster_event))

    if compatible_events:
        confidence = max(
            min(canonical_event.confidence, roster_event.confidence)
            for _, canonical_event, roster_event in compatible_events
        )
        delta = config.participation_weight * confidence
        event_namespaces = [
            f"{source}:{event_id}" for (source, event_id), _, _ in compatible_events
        ]
        evidence.append(
            _candidate_evidence(
                "player-event-match",
                delta,
                confidence,
                "matched-namespaced-event",
                *event_namespaces,
            )
        )
        identity_signal_score += delta
        reasons.append("player-event-match")
    elif canonical_events:
        canonical_ids = {event_id for _, event_id in canonical_events}
        roster_ids = {event_id for _, event_id in roster_events}
        if canonical_ids & roster_ids and not shared_event_keys:
            reasons.append("event-source-mismatch")
        if not any(reason.startswith("event-") for reason in reasons):
            reasons.append("no-player-event-match")

    likelihoods_by_player: dict[str, list[PlayerLikelihoodEvidence]] = defaultdict(list)
    for item in person.player_likelihoods:
        likelihoods_by_player[item.external_player_id].append(item)
    strongest_by_player = {
        identifier: max(items, key=lambda item: (item.confidence, item.evidence_id))
        for identifier, items in likelihoods_by_player.items()
    }
    direct = strongest_by_player.get(player.external_player_id)
    if direct is not None:
        delta = config.direct_player_weight * direct.confidence
        evidence.append(
            _candidate_evidence(
                "direct-player-evidence",
                delta,
                direct.confidence,
                direct.source,
                direct.evidence_id,
            )
        )
        identity_signal_score += delta
        reasons.append("direct-player-evidence")
    if strongest_by_player:
        strongest = max(
            strongest_by_player.values(),
            key=lambda item: (item.confidence, item.external_player_id),
        )
        candidate_confidence = direct.confidence if direct is not None else 0.0
        if (
            strongest.external_player_id != player.external_player_id
            and strongest.confidence >= config.hard_direct_player_confidence
            and strongest.confidence - candidate_confidence
            >= config.direct_player_gate_margin
        ):
            conflicts.append("direct-player-evidence-conflict")

    score = max(0.0, min(1.0, sum(item.score_delta for item in evidence)))
    identity_signal_score = max(0.0, min(1.0, identity_signal_score))
    conflicts = list(dict.fromkeys(conflicts))
    reasons = list(dict.fromkeys(reasons))
    return RosterIdentityCandidate(
        external_player_id=player.external_player_id,
        display_name=player.display_name,
        team_id=player.team_id,
        jersey_number=(
            str(player.jersey_number) if player.jersey_number is not None else None
        ),
        role=player.role,
        score=score,
        identity_signal_score=identity_signal_score,
        evidence=tuple(evidence),
        reasons=tuple(reasons),
        conflicts=tuple(conflicts),
        eligible=not conflicts,
        proposal_status=(
            "confirmed" if confirmed_binding else "blocked" if conflicts else "alternative"
        ),
    )


def _rank_candidates(
    candidates: Iterable[RosterIdentityCandidate],
    *,
    limit: int,
    force_ids: Iterable[str] = (),
) -> tuple[RosterIdentityCandidate, ...]:
    priority = {
        "confirmed": 0,
        "selected": 1,
        "ambiguous": 2,
        "alternative": 3,
        "blocked": 4,
    }
    rows = sorted(
        candidates,
        key=lambda item: (
            priority[item.proposal_status],
            not item.eligible,
            -item.score,
            -item.identity_signal_score,
            item.display_name.casefold(),
            item.external_player_id,
        ),
    )
    forced = set(force_ids)
    selected = rows[:limit]
    selected_ids = {item.external_player_id for item in selected}
    selected.extend(
        item
        for item in rows[limit:]
        if item.external_player_id in forced and item.external_player_id not in selected_ids
    )
    return tuple(replace(item, rank=index) for index, item in enumerate(selected, start=1))


def _assignment(
    people: Sequence[CanonicalPersonEvidence],
    players: Sequence[PersistedRosterPlayer],
    candidates: Mapping[str, Mapping[str, RosterIdentityCandidate]],
    config: RosterResolverConfig,
) -> tuple[dict[str, str], set[tuple[str, str]], float]:
    """Return globally unique suggestions, ambiguous selected edges, objective."""

    if not people:
        return {}, set(), 0.0
    player_columns = {item.external_player_id: index for index, item in enumerate(players)}
    column_count = len(players) + len(people)
    forbidden = -1_000_000.0
    utility = np.full((len(people), column_count), forbidden, dtype=np.float64)
    for row, person in enumerate(people):
        for player_id, candidate in candidates[person.canonical_person_id].items():
            column = player_columns[player_id]
            if (
                candidate.eligible
                and candidate.score >= config.min_candidate_score
                and candidate.identity_signal_score >= config.min_identity_signal_score
            ):
                utility[row, column] = candidate.score
        # ``min_candidate_score`` is the actual candidate threshold.  Put the
        # explicit unknown alternative one configured margin below it so we do
        # not accidentally apply the margin twice (once as a threshold and a
        # second time against the abstain column).
        utility[row, len(players) :] = max(
            0.0, config.min_candidate_score - config.assignment_margin
        )

    rows, columns = linear_sum_assignment(-utility)
    objective = float(sum(utility[row, column] for row, column in zip(rows, columns)))
    selected: dict[str, str] = {}
    selected_edges: list[tuple[int, int]] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= len(players) or utility[row, column] < config.min_candidate_score:
            continue
        person_id = people[row].canonical_person_id
        player_id = players[column].external_player_id
        selected[person_id] = player_id
        selected_edges.append((row, column))

    ambiguous: set[tuple[str, str]] = set()
    for selected_row, selected_column in selected_edges:
        alternative = utility.copy()
        alternative[selected_row, selected_column] = forbidden
        alt_rows, alt_columns = linear_sum_assignment(-alternative)
        alternative_objective = float(
            sum(alternative[row, column] for row, column in zip(alt_rows, alt_columns))
        )
        # Exact objective ties are ambiguous even when an experimental config
        # sets the requested margin to zero. Numerical epsilon keeps solver
        # tie-breaking from becoming an identity decision.
        if objective - alternative_objective <= max(
            config.assignment_margin, 1e-9
        ):
            ambiguous.add(
                (
                    people[selected_row].canonical_person_id,
                    players[selected_column].external_player_id,
                )
            )
    return selected, ambiguous, objective


def resolve_closed_set_roster(
    canonical_people: Iterable[CanonicalPersonEvidence],
    persisted_players: Iterable[PersistedRosterPlayer],
    config: RosterResolverConfig | None = None,
) -> ClosedSetRosterResolution:
    """Rank closed-set roster candidates without accepting any binding.

    ``suggested_external_player_id`` is a UI review hint only.  The caller must
    use the existing durable manual roster-binding endpoint after a person
    confirms it.  ``auto_bindings`` and every ``auto_bind_external_player_id``
    are intentionally empty/null.
    """

    config = config or RosterResolverConfig()
    people = tuple(sorted(tuple(canonical_people), key=lambda item: item.canonical_person_id))
    players = tuple(sorted(tuple(persisted_players), key=lambda item: item.external_player_id))
    person_ids = [item.canonical_person_id for item in people]
    player_ids = [item.external_player_id for item in players]
    if len(person_ids) != len(set(person_ids)):
        raise ValueError("canonical_person_id values must be unique")
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("persisted external_player_id values must be unique")

    player_by_id = {item.external_player_id: item for item in players}
    confirmed_owners: dict[str, list[CanonicalPersonEvidence]] = defaultdict(list)
    for person in people:
        if person.confirmed_external_player_id is not None:
            confirmed_owners[person.confirmed_external_player_id].append(person)
    reserved_player_ids = set(confirmed_owners)

    global_conflicts: list[RosterResolutionConflict] = []
    person_conflicts: dict[str, list[str]] = defaultdict(list)
    for external_id, owners in sorted(confirmed_owners.items()):
        if len(owners) <= 1:
            continue
        owner_ids = tuple(sorted(item.canonical_person_id for item in owners))
        global_conflicts.append(
            RosterResolutionConflict(
                code="duplicate-confirmed-player-binding",
                message=(
                    "The same real player is manually bound to multiple canonical identities; "
                    "the bindings were retained and require review."
                ),
                canonical_person_ids=owner_ids,
                external_player_id=external_id,
            )
        )
        for owner_id in owner_ids:
            person_conflicts[owner_id].append("duplicate-confirmed-player-binding")
        for left, right in combinations(owners, 2):
            if _intervals_overlap(left.visible_intervals, right.visible_intervals):
                simultaneous_ids = tuple(
                    sorted((left.canonical_person_id, right.canonical_person_id))
                )
                global_conflicts.append(
                    RosterResolutionConflict(
                        code="simultaneous-confirmed-player-duplicate",
                        message=(
                            "Two simultaneously visible identities carry the same confirmed "
                            "real player. Manual bindings remain authoritative but invalid."
                        ),
                        canonical_person_ids=simultaneous_ids,
                        external_player_id=external_id,
                    )
                )
                for owner_id in simultaneous_ids:
                    person_conflicts[owner_id].append(
                        "simultaneous-confirmed-player-duplicate"
                    )

    candidate_maps: dict[str, dict[str, RosterIdentityCandidate]] = {}
    for person in people:
        rows: dict[str, RosterIdentityCandidate] = {}
        for player in players:
            is_bound = person.confirmed_external_player_id == player.external_player_id
            candidate = _evaluate_candidate(
                person, player, config, confirmed_binding=is_bound
            )
            if (
                not is_bound
                and person.confirmed_external_player_id is not None
            ):
                candidate = replace(
                    candidate,
                    conflicts=tuple(
                        (*candidate.conflicts, "different-player-manually-confirmed")
                    ),
                    eligible=False,
                    proposal_status="blocked",
                )
            elif (
                not is_bound
                and player.external_player_id in reserved_player_ids
            ):
                candidate = replace(
                    candidate,
                    conflicts=tuple(
                        (*candidate.conflicts, "player-reserved-by-confirmed-binding")
                    ),
                    eligible=False,
                    proposal_status="blocked",
                )
            rows[player.external_player_id] = candidate
        candidate_maps[person.canonical_person_id] = rows

    unconfirmed_people = tuple(
        item for item in people if item.confirmed_external_player_id is None
    )
    unreserved_players = tuple(
        item for item in players if item.external_player_id not in reserved_player_ids
    )
    unreserved_maps = {
        person.canonical_person_id: {
            player.external_player_id: candidate_maps[person.canonical_person_id][
                player.external_player_id
            ]
            for player in unreserved_players
        }
        for person in unconfirmed_people
    }
    selected, ambiguous, assignment_objective = _assignment(
        unconfirmed_people,
        unreserved_players,
        unreserved_maps,
        config,
    )

    resolutions: list[PersonRosterResolution] = []
    for person in people:
        rows = candidate_maps[person.canonical_person_id]
        base_conflicts = list(dict.fromkeys(person_conflicts[person.canonical_person_id]))
        reasons: list[str] = []
        if person.confirmed_external_player_id is not None:
            external_id = person.confirmed_external_player_id
            if external_id not in player_by_id:
                base_conflicts.append("confirmed-player-missing-from-persisted-roster")
                missing = RosterIdentityCandidate(
                    external_player_id=external_id,
                    display_name=external_id,
                    team_id=None,
                    jersey_number=None,
                    role=None,
                    score=1.0,
                    identity_signal_score=1.0,
                    evidence=(
                        _candidate_evidence(
                            "manual-confirmed-binding",
                            1.0,
                            1.0,
                            "manual-roster-binding",
                        ),
                    ),
                    reasons=("manual-confirmed-binding-authoritative",),
                    conflicts=("confirmed-player-missing-from-persisted-roster",),
                    eligible=False,
                    proposal_status="confirmed",
                )
                rows = {external_id: missing, **rows}
            bound_candidate = rows.get(external_id)
            if bound_candidate is not None:
                base_conflicts.extend(bound_candidate.conflicts)
            reasons.append("manual-binding-retained")
            ranked = _rank_candidates(
                rows.values(), limit=config.candidate_limit, force_ids=(external_id,)
            )
            resolutions.append(
                PersonRosterResolution(
                    canonical_person_id=person.canonical_person_id,
                    status="confirmed",
                    confirmed_external_player_id=external_id,
                    suggested_external_player_id=None,
                    candidates=ranked,
                    reasons=tuple(reasons),
                    conflicts=tuple(dict.fromkeys(base_conflicts)),
                )
            )
            continue

        selected_id = selected.get(person.canonical_person_id)
        if selected_id is not None:
            candidate = rows[selected_id]
            if (person.canonical_person_id, selected_id) in ambiguous:
                rows[selected_id] = replace(
                    candidate,
                    proposal_status="ambiguous",
                    reasons=tuple(
                        (*candidate.reasons, "global-assignment-margin-too-small")
                    ),
                )
                selected_id = None
                reasons.append("global-assignment-ambiguous")
            else:
                rows[selected_id] = replace(candidate, proposal_status="selected")
                reasons.extend(
                    (
                        "globally-unique-roster-suggestion",
                        "manual-confirmation-required",
                    )
                )
        if selected_id is None and not reasons:
            eligible = [
                item
                for item in rows.values()
                if item.eligible
                and item.external_player_id not in reserved_player_ids
                and item.score >= config.min_candidate_score
            ]
            identity_eligible = [
                item
                for item in eligible
                if item.identity_signal_score >= config.min_identity_signal_score
            ]
            if not players:
                reasons.append("persisted-roster-empty")
            elif not identity_eligible:
                reasons.append("insufficient-identity-evidence")
            else:
                reasons.append("global-one-to-one-abstain")

        ranked = _rank_candidates(
            rows.values(),
            limit=config.candidate_limit,
            force_ids=(selected_id,) if selected_id is not None else (),
        )
        resolutions.append(
            PersonRosterResolution(
                canonical_person_id=person.canonical_person_id,
                status="suggested" if selected_id is not None else "abstain",
                confirmed_external_player_id=None,
                suggested_external_player_id=selected_id,
                candidates=ranked,
                reasons=tuple(dict.fromkeys(reasons)),
                conflicts=tuple(dict.fromkeys(base_conflicts)),
            )
        )

    resolutions.sort(key=lambda item: item.canonical_person_id)
    conflict_codes = Counter(
        conflict
        for resolution in resolutions
        for conflict in resolution.conflicts
    )
    candidate_conflict_codes = Counter(
        conflict
        for resolution in resolutions
        for candidate in resolution.candidates
        for conflict in candidate.conflicts
    )
    diagnostics: dict[str, object] = {
        "canonicalPersonCount": len(people),
        "persistedRosterPlayerCount": len(players),
        "confirmedBindingCount": sum(item.status == "confirmed" for item in resolutions),
        "suggestionCount": sum(item.status == "suggested" for item in resolutions),
        "abstainCount": sum(item.status == "abstain" for item in resolutions),
        "confirmedBindingConflictCount": sum(
            item.status == "confirmed" and bool(item.conflicts) for item in resolutions
        ),
        "globalConflictCount": len(global_conflicts),
        "conflictCounts": dict(sorted(conflict_codes.items())),
        "candidateConflictCounts": dict(sorted(candidate_conflict_codes.items())),
        "reservedExternalPlayerIds": sorted(reserved_player_ids),
        "assignmentObjective": round(assignment_objective, 6),
        "oneToOneSuggestions": len(
            {
                item.suggested_external_player_id
                for item in resolutions
                if item.suggested_external_player_id is not None
            }
        )
        == sum(item.status == "suggested" for item in resolutions),
        "automaticBindingCount": 0,
        "requiresManualConfirmation": True,
    }
    return ClosedSetRosterResolution(
        people=tuple(resolutions),
        conflicts=tuple(global_conflicts),
        diagnostics=diagnostics,
    )


__all__ = [
    "AttributeEvidence",
    "CandidateEvidence",
    "CanonicalPersonEvidence",
    "ClosedSetRosterResolution",
    "ParticipationEvidence",
    "PersistedRosterPlayer",
    "PlayerLikelihoodEvidence",
    "PersonRosterResolution",
    "RosterIdentityCandidate",
    "RosterResolutionConflict",
    "RosterResolverConfig",
    "TimeInterval",
    "resolve_closed_set_roster",
]
