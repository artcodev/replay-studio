from __future__ import annotations

import re
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


THESPORTSDB_PROVIDER_ID = "thesportsdb"


def normalize_event_search_query(query: str) -> str:
    normalized = re.sub(
        r"\s+(?:vs\.?|v\.?|@)\s+",
        "_vs_",
        query.strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", "_", normalized)


def map_event(item: dict[str, Any]) -> ExternalEvent:
    return ExternalEvent(
        id=str(item.get("idEvent") or "unknown"),
        provider=THESPORTSDB_PROVIDER_ID,
        name=item.get("strEvent") or "Untitled match",
        date=item.get("dateEvent"),
        time=item.get("strTime"),
        status=item.get("strStatus"),
        league=item.get("strLeague"),
        season=item.get("strSeason"),
        home=ExternalTeam(
            id=str(item.get("idHomeTeam") or "home"),
            name=item.get("strHomeTeam") or "Home",
            badge=item.get("strHomeTeamBadge"),
        ),
        away=ExternalTeam(
            id=str(item.get("idAwayTeam") or "away"),
            name=item.get("strAwayTeam") or "Away",
            badge=item.get("strAwayTeamBadge"),
        ),
        home_score=_integer(item.get("intHomeScore")),
        away_score=_integer(item.get("intAwayScore")),
        thumbnail=item.get("strThumb") or item.get("strPoster"),
    )


def map_lineup(
    rows: list[dict[str, Any]],
    event: ExternalEvent,
) -> tuple[list[ExternalPlayer], list[ExternalLineupEntry]]:
    players: list[ExternalPlayer] = []
    entries: list[ExternalLineupEntry] = []
    seen_player_ids: set[str] = set()
    for index, item in enumerate(rows):
        entry = map_lineup_entry(item, event, index)
        entries.append(entry)
        player = _lineup_player(entry, item)
        if player.id in seen_player_ids:
            continue
        seen_player_ids.add(player.id)
        players.append(player)
    return players, entries


def map_lineup_entry(
    item: dict[str, Any],
    event: ExternalEvent,
    index: int,
) -> ExternalLineupEntry:
    side = _lineup_side(item)
    if side == "home":
        team_id, team_name = event.home.id, event.home.name
    elif side == "away":
        team_id, team_name = event.away.id, event.away.name
    else:
        team_id = str(item.get("idTeam")) if item.get("idTeam") else None
        team_name = item.get("strTeam")
    player_id = str(item.get("idPlayer") or f"lineup-player-{index}")
    number = item.get("intSquadNumber")
    return ExternalLineupEntry(
        id=str(item.get("idLineup") or f"lineup-{index}"),
        player_id=player_id,
        player_name=(
            item.get("strPlayer")
            or item.get("strPlayerName")
            or "Unknown player"
        ),
        team_id=team_id,
        team_name=team_name,
        side=side,
        position=item.get("strPosition"),
        number=str(number) if number not in (None, "") else None,
        role=_lineup_role(item),
        order=index,
    )


def map_timeline(
    rows: list[dict[str, Any]],
) -> tuple[list[ExternalTimelineEvent], list[ExternalSubstitution]]:
    timeline: list[ExternalTimelineEvent] = []
    substitutions: list[ExternalSubstitution] = []
    for index, item in enumerate(rows):
        minute = _integer(item.get("intTime") or item.get("strTime"))
        event_type = item.get("strTimeline") or item.get("strType") or "event"
        player_name = item.get("strPlayer") or item.get("strPlayerName")
        secondary_player_name = (
            item.get("strPlayer2") or item.get("strPlayerName2")
        )
        timeline_id = str(item.get("idTimeline") or f"timeline-{index}")
        team_id = str(item.get("idTeam")) if item.get("idTeam") else None
        team_name = item.get("strTeam")
        player_id = (
            str(item.get("idPlayer")) if item.get("idPlayer") else None
        )
        secondary_player_id = (
            str(item.get("idPlayer2")) if item.get("idPlayer2") else None
        )
        detail = item.get("strTimelineDetail") or item.get("strDetail")
        normalized_type = event_type.lower().replace(" ", "-")
        label = f"{event_type}{f' · {player_name}' if player_name else ''}"
        timeline.append(
            ExternalTimelineEvent(
                id=timeline_id,
                minute=minute,
                type=normalized_type,
                label=label,
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                team_name=team_name,
                secondary_player_id=secondary_player_id,
                secondary_player_name=secondary_player_name,
                detail=detail,
            )
        )
        if "substitut" in normalized_type:
            substitutions.append(
                ExternalSubstitution(
                    id=f"substitution-{timeline_id}",
                    minute=minute,
                    team_id=team_id,
                    team_name=team_name,
                    player_out_id=player_id,
                    player_out_name=player_name,
                    player_in_id=secondary_player_id,
                    player_in_name=secondary_player_name,
                    label=label,
                )
            )
    return timeline, substitutions


def assess_roster_quality(
    players: list[ExternalPlayer],
    event: ExternalEvent,
) -> ExternalRosterQuality:
    home_count = sum(player.team_id == event.home.id for player in players)
    away_count = sum(player.team_id == event.away.id for player in players)
    reasons: list[str] = []
    if not players:
        reasons.append("roster-unavailable")
    if len(players) == 5:
        reasons.append("provider-five-player-cap")
    if players and (home_count < 11 or away_count < 11):
        reasons.append("fewer-than-eleven-players-per-team")
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


def _lineup_player(
    entry: ExternalLineupEntry,
    item: dict[str, Any],
) -> ExternalPlayer:
    return ExternalPlayer(
        id=entry.player_id,
        name=entry.player_name,
        team_id=entry.team_id,
        team_name=entry.team_name,
        position=entry.position,
        number=entry.number,
        thumbnail=item.get("strThumb") or item.get("strCutout"),
        lineup_role=entry.role,
        lineup_order=entry.order,
    )


def _lineup_side(item: dict[str, Any]) -> str:
    side = str(item.get("strHome") or "").lower()
    if side == "yes":
        return "home"
    if side == "no":
        return "away"
    return "unknown"


def _lineup_role(item: dict[str, Any]) -> str:
    substitute = str(item.get("strSubstitute") or "").strip().lower()
    if substitute in {"yes", "true", "1", "substitute", "sub"}:
        return "substitute"
    if substitute in {"no", "false", "0", "starter", "start"}:
        return "starter"
    return "unknown"


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
