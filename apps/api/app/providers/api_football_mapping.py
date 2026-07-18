from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from ..match_contracts import (
    ExternalEvent,
    ExternalLineupEntry,
    ExternalPlayer,
    ExternalRosterQuality,
    ExternalSubstitution,
    ExternalTeam,
    ExternalTimelineEvent,
)


API_FOOTBALL_PROVIDER_ID = "api-football"


def response_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    response = data.get("response")
    if not isinstance(response, list):
        return []
    return [item for item in response if isinstance(item, dict)]


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed.lower() if character.isalnum()
    )


def map_fixture(item: dict[str, Any]) -> ExternalEvent:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    goals = item.get("goals") or {}
    status = fixture.get("status") or {}
    date_time = _text(fixture.get("date"))
    date_value = date_time[:10] if date_time else None
    time_value = None
    if date_time and "T" in date_time:
        time_value = date_time.split("T", 1)[1][:5]
    home_name = _text(home.get("name")) or "Home"
    away_name = _text(away.get("name")) or "Away"
    return ExternalEvent(
        id=str(fixture.get("id") or "unknown"),
        provider=API_FOOTBALL_PROVIDER_ID,
        name=f"{home_name} vs {away_name}",
        date=date_value,
        time=time_value,
        status=_text(status.get("long") or status.get("short")),
        league=_text(league.get("name")),
        season=_text(league.get("season")),
        home=ExternalTeam(
            id=str(home.get("id") or "home"),
            name=home_name,
            badge=_text(home.get("logo")),
        ),
        away=ExternalTeam(
            id=str(away.get("id") or "away"),
            name=away_name,
            badge=_text(away.get("logo")),
        ),
        home_score=_integer(goals.get("home")),
        away_score=_integer(goals.get("away")),
        thumbnail=None,
    )


def map_lineups(
    rows: list[dict[str, Any]],
    event: ExternalEvent,
) -> tuple[list[ExternalPlayer], list[ExternalLineupEntry]]:
    by_team = {
        event.home.id: ("home", event.home),
        event.away.id: ("away", event.away),
    }
    players: list[ExternalPlayer] = []
    entries: list[ExternalLineupEntry] = []
    seen_players: set[str] = set()
    order = 0
    for row in rows:
        team = row.get("team") or {}
        team_id = (
            str(team.get("id")) if team.get("id") not in (None, "") else None
        )
        side, canonical_team = by_team.get(team_id or "", ("unknown", None))
        team_name = (
            canonical_team.name
            if canonical_team is not None
            else _text(team.get("name"))
        )
        formation = _text(row.get("formation"))
        for role, key in (("starter", "startXI"), ("substitute", "substitutes")):
            source_players = row.get(key) or []
            if not isinstance(source_players, list):
                continue
            for source in source_players:
                player = source.get("player") if isinstance(source, dict) else None
                if not isinstance(player, dict):
                    continue
                raw_id = player.get("id")
                player_id = (
                    str(raw_id) if raw_id not in (None, "") else f"unknown-{order}"
                )
                player_name = _text(player.get("name")) or "Unknown player"
                number = _text(player.get("number"))
                position = _text(player.get("pos"))
                entries.append(
                    ExternalLineupEntry(
                        id=(
                            f"{event.id}:{team_id or side}:{role}:"
                            f"{player_id}:{order}"
                        ),
                        player_id=player_id,
                        player_name=player_name,
                        team_id=team_id,
                        team_name=team_name,
                        side=side,
                        position=position,
                        number=number,
                        role=role,
                        order=order,
                        formation=formation,
                        grid=_text(player.get("grid")),
                    )
                )
                order += 1
                if player_id in seen_players:
                    continue
                seen_players.add(player_id)
                players.append(
                    ExternalPlayer(
                        id=player_id,
                        name=player_name,
                        team_id=team_id,
                        team_name=team_name,
                        position=position,
                        number=number,
                        lineup_role=role,
                        lineup_order=order - 1,
                    )
                )
    return players, entries


def map_timeline(
    rows: list[dict[str, Any]],
    event: ExternalEvent,
) -> tuple[list[ExternalTimelineEvent], list[ExternalSubstitution]]:
    timeline: list[ExternalTimelineEvent] = []
    substitutions: list[ExternalSubstitution] = []
    signature_occurrences: dict[str, int] = {}
    for row in rows:
        time = row.get("time") or {}
        team = row.get("team") or {}
        player = row.get("player") or {}
        secondary = row.get("assist") or {}
        raw_type = _text(row.get("type")) or "event"
        detail = _text(row.get("detail") or row.get("comments"))
        normalized_type = (
            re.sub(r"[^a-z0-9]+", "-", raw_type.lower()).strip("-") or "event"
        )
        player_name = _text(player.get("name"))
        secondary_name = _text(secondary.get("name"))
        elapsed = _integer(time.get("elapsed"))
        extra = _integer(time.get("extra")) or 0
        minute = (elapsed + extra) if elapsed is not None else None
        clock = (
            f"{elapsed}+{extra}'"
            if elapsed is not None and extra
            else f"{elapsed}'"
            if elapsed is not None
            else None
        )
        signature = json.dumps(
            {
                "fixture": event.id,
                "elapsed": elapsed,
                "extra": extra,
                "team": team.get("id"),
                "player": player.get("id"),
                "assist": secondary.get("id"),
                "type": raw_type,
                "detail": detail,
                "comments": _text(row.get("comments")),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        occurrence = signature_occurrences.get(digest, 0)
        signature_occurrences[digest] = occurrence + 1
        timeline_id = f"{event.id}:event:{digest}:{occurrence}"
        label = " · ".join(
            value for value in (clock, raw_type, player_name) if value
        )
        item = ExternalTimelineEvent(
            id=timeline_id,
            minute=minute,
            type=normalized_type,
            label=label,
            player_id=_text(player.get("id")),
            player_name=player_name,
            team_id=_text(team.get("id")),
            team_name=_text(team.get("name")),
            secondary_player_id=_text(secondary.get("id")),
            secondary_player_name=secondary_name,
            detail=detail,
        )
        timeline.append(item)
        is_substitution = (
            normalized_type in {"subst", "substitution"}
            or "substitut" in normalized_type
        )
        if is_substitution:
            substitutions.append(
                ExternalSubstitution(
                    id=f"{event.id}:substitution:{digest}:{occurrence}",
                    minute=item.minute,
                    team_id=item.team_id,
                    team_name=item.team_name,
                    player_out_id=item.player_id,
                    player_out_name=item.player_name,
                    player_in_id=item.secondary_player_id,
                    player_in_name=item.secondary_player_name,
                    label=label,
                )
            )
    return timeline, substitutions


def assess_roster_quality(
    players: list[ExternalPlayer],
    lineup: list[ExternalLineupEntry],
    event: ExternalEvent,
    substitutions: list[ExternalSubstitution],
) -> ExternalRosterQuality:
    home_count = sum(player.team_id == event.home.id for player in players)
    away_count = sum(player.team_id == event.away.id for player in players)
    home_starters = sum(
        entry.side == "home" and entry.role == "starter" for entry in lineup
    )
    away_starters = sum(
        entry.side == "away" and entry.role == "starter" for entry in lineup
    )
    goalkeeper_positions = {"g", "gk", "goalkeeper"}
    home_goalkeepers = sum(
        entry.side == "home"
        and entry.role == "starter"
        and str(entry.position or "").strip().lower() in goalkeeper_positions
        for entry in lineup
    )
    away_goalkeepers = sum(
        entry.side == "away"
        and entry.role == "starter"
        and str(entry.position or "").strip().lower() in goalkeeper_positions
        for entry in lineup
    )
    reasons: list[str] = []
    if not players:
        reasons.append("roster-unavailable")
    if players and (home_count < 11 or away_count < 11):
        reasons.append("fewer-than-eleven-players-per-team")
    if players and (home_starters < 11 or away_starters < 11):
        reasons.append("fewer-than-eleven-starters-per-team")
    if players and (home_goalkeepers < 1 or away_goalkeepers < 1):
        reasons.append("starting-goalkeeper-missing")
    lineup_player_ids = [entry.player_id for entry in lineup]
    if len(lineup_player_ids) != len(set(lineup_player_ids)):
        reasons.append("duplicate-player-id-in-lineup")
    player_ids = {player.id for player in players}
    substitution_ids = {
        player_id
        for substitution in substitutions
        for player_id in (substitution.player_out_id, substitution.player_in_id)
        if player_id
    }
    if substitution_ids - player_ids:
        reasons.append("substitution-player-missing-from-lineup")
    automatic = bool(players) and not reasons
    return ExternalRosterQuality(
        status=(
            "automatic-ready"
            if automatic
            else "partial"
            if players
            else "unavailable"
        ),
        player_count=len(players),
        home_player_count=home_count,
        away_player_count=away_count,
        automatic_identity_eligible=automatic,
        manual_identity_eligible=bool(players),
        reasons=reasons,
    )


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value not in (None, "") else ""
    return text or None
