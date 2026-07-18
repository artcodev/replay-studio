from __future__ import annotations

"""Canonical match-roster lookup and team/role validation."""

from typing import Mapping

from .reconstruction_errors import ReconstructionError


def match_snapshot_player(
    match_snapshot: Mapping[str, object] | None,
    external_player_id: str,
) -> dict:
    snapshot = match_snapshot if isinstance(match_snapshot, Mapping) else {}
    players = [
        item
        for item in snapshot.get("roster") or []
        if isinstance(item, dict)
        and str(item.get("id") or "").strip() == external_player_id
    ]
    if len(players) != 1:
        if players:
            raise ReconstructionError(
                "The bound match roster contains a duplicate external player id"
            )
        raise ReconstructionError(
            "The selected player is not present in the bound match roster"
        )
    return players[0]


def validate_canonical_roster_team(
    match_snapshot: Mapping[str, object] | None,
    person: dict,
    player: dict,
) -> None:
    validate_canonical_roster_person(person)
    local_team_id = str(person.get("teamId") or "").strip()
    snapshot = match_snapshot if isinstance(match_snapshot, Mapping) else {}
    bound_team = snapshot.get(
        "homeTeam" if local_team_id == "home" else "awayTeam"
    )
    expected_team_id = (
        str(bound_team.get("id") or "").strip()
        if isinstance(bound_team, dict)
        else str(bound_team or "").strip()
    )
    roster_team_id = str(
        player.get("team_id") or player.get("teamId") or ""
    ).strip()
    if expected_team_id and roster_team_id and roster_team_id != expected_team_id:
        raise ReconstructionError(
            f"The selected roster player belongs to the other team ({local_team_id} expected)"
        )


def validate_canonical_roster_person(person: dict) -> None:
    local_team_id = str(person.get("teamId") or "").strip()
    role = str(person.get("role") or "").strip()
    if local_team_id not in {"home", "away"} or role in {"referee", "other"}:
        raise ReconstructionError(
            "Only a canonical home or away player can be bound to the match roster"
        )
