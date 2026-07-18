"""Review-only roster candidate ranking from reliable jersey evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .jersey_ocr_contract import (
    JerseyEvidenceSummary,
    identifier,
    normalize_jersey_number,
)


@dataclass(frozen=True)
class RosterPlayer:
    external_player_id: str
    display_name: str
    jersey_number: str | int | None = None
    team_id: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_player_id",
            identifier(self.external_player_id, "RosterPlayer.external_player_id"),
        )
        object.__setattr__(
            self,
            "display_name",
            identifier(self.display_name, "RosterPlayer.display_name"),
        )
        team_id = str(self.team_id).strip() if self.team_id is not None else ""
        role = str(self.role).strip() if self.role is not None else ""
        object.__setattr__(self, "team_id", team_id or None)
        object.__setattr__(self, "role", role or None)
        object.__setattr__(
            self,
            "jersey_number",
            normalize_jersey_number(self.jersey_number),
        )


@dataclass(frozen=True)
class RosterCandidate:
    external_player_id: str
    display_name: str
    jersey_number: str
    team_id: str | None
    role: str | None
    score: float
    reasons: tuple[str, ...]
    requires_manual_confirmation: bool = field(default=True, init=False)

    def to_payload(self) -> dict:
        return {
            "externalPlayerId": self.external_player_id,
            "name": self.display_name,
            "number": self.jersey_number,
            "teamId": self.team_id,
            "position": self.role,
            "confidence": round(self.score, 6),
            "reasons": list(self.reasons),
            "requiresManualConfirmation": True,
        }


@dataclass(frozen=True)
class RosterCandidateSet:
    subject_id: str
    candidates: tuple[RosterCandidate, ...]
    reason: str
    requires_manual_confirmation: bool = field(default=True, init=False)

    def to_payload(self) -> list[dict]:
        return [item.to_payload() for item in self.candidates]


def generate_roster_candidates(
    evidence: JerseyEvidenceSummary,
    roster: Iterable[RosterPlayer],
    *,
    team_id: str | None = None,
    limit: int = 10,
) -> RosterCandidateSet:
    """Rank exact-number matches without selecting or binding a player."""

    if int(limit) < 1:
        raise ValueError("limit must be positive")
    rows = tuple(roster)
    ids = [item.external_player_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("roster external_player_id values must be unique")
    if evidence.jersey_number is None or evidence.status != "reliable":
        return RosterCandidateSet(
            subject_id=evidence.subject_id,
            candidates=(),
            reason="reliable-jersey-required",
        )

    normalized_team = str(team_id).strip() or None if team_id is not None else None
    candidates: list[RosterCandidate] = []
    for player in rows:
        if player.jersey_number != evidence.jersey_number:
            continue
        if normalized_team is not None and player.team_id == normalized_team:
            multiplier = 1.0
            team_reason = "team-match"
        elif normalized_team is None or player.team_id is None:
            multiplier = 0.90
            team_reason = "team-unavailable"
        else:
            multiplier = 0.50
            team_reason = "team-conflict"
        candidates.append(
            RosterCandidate(
                external_player_id=player.external_player_id,
                display_name=player.display_name,
                jersey_number=player.jersey_number,
                team_id=player.team_id,
                role=player.role,
                score=evidence.confidence * multiplier,
                reasons=("reliable-jersey-number-match", team_reason),
            )
        )
    candidates.sort(
        key=lambda item: (-item.score, item.display_name.casefold(), item.external_player_id)
    )
    return RosterCandidateSet(
        subject_id=evidence.subject_id,
        candidates=tuple(candidates[: int(limit)]),
        reason="manual-confirmation-required" if candidates else "no-number-match",
    )
