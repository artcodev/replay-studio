from __future__ import annotations

"""Provider-neutral football match normalization.

Provider adapters are allowed to speak in upstream identifiers.  Everything
past this module uses opaque Replay Studio identifiers; upstream provenance is
persisted only through ``MatchSnapshotRow`` and ``ExternalReferenceRow``.
"""

from typing import Any

from .project_identifiers import stable_identifier
from .match_contracts import EventBundle


CANONICAL_MATCH_SCHEMA_VERSION = 1


def _text(value: object) -> str | None:
    result = str(value or "").strip()
    return result or None


def _canonical_id(prefix: str, provider: str, external_id: object, *fallback: object) -> str:
    identity = _text(external_id)
    if identity is not None:
        return stable_identifier(prefix, provider, identity, length=24)
    return stable_identifier(prefix, provider, *fallback, length=24)


def _event_type(value: object, detail: object = None) -> str:
    combined = f"{value or ''} {detail or ''}".strip().lower().replace("_", "-")
    if "own goal" in combined or "own-goal" in combined:
        return "own-goal"
    if "penalty" in combined and "goal" in combined:
        return "penalty-goal"
    if "goal" in combined:
        return "goal"
    if "yellow" in combined:
        return "yellow-card"
    if "red" in combined:
        return "red-card"
    if "substitut" in combined or combined in {"subst", "sub"}:
        return "substitution"
    if "var" in combined:
        return "var"
    return "other"


def canonical_match_id(bundle: EventBundle) -> str:
    return _canonical_id(
        "match",
        bundle.source,
        bundle.event.id,
        bundle.event.date,
        bundle.event.home.name,
        bundle.event.away.name,
    )


def canonicalize_event_bundle(
    bundle: EventBundle,
    *,
    match_id: str | None = None,
) -> dict[str, Any]:
    """Map one provider bundle into the stable public match contract."""

    internal_match_id = match_id or canonical_match_id(bundle)
    provider = bundle.source
    home_id = _canonical_id(
        "team", provider, bundle.event.home.id, bundle.event.home.name
    )
    away_id = _canonical_id(
        "team", provider, bundle.event.away.id, bundle.event.away.name
    )
    team_ids = {
        str(bundle.event.home.id): home_id,
        str(bundle.event.away.id): away_id,
    }

    player_ids: dict[str, str] = {}
    roster: list[dict[str, Any]] = []
    for order, player in enumerate(bundle.players):
        provider_player_id = str(player.id)
        internal_player_id = _canonical_id(
            "player",
            provider,
            player.id,
            internal_match_id,
            player.team_id,
            player.name,
            player.number,
            order,
        )
        player_ids[provider_player_id] = internal_player_id
        roster.append(
            {
                "id": internal_player_id,
                "name": player.name,
                "teamId": team_ids.get(str(player.team_id or "")),
                "teamName": player.team_name,
                "position": player.position,
                "number": player.number,
                "thumbnail": player.thumbnail,
                "lineupRole": player.lineup_role,
                "lineupOrder": player.lineup_order,
            }
        )

    lineup: list[dict[str, Any]] = []
    for order, entry in enumerate(bundle.lineup):
        internal_player_id = player_ids.get(str(entry.player_id)) or _canonical_id(
            "player",
            provider,
            entry.player_id,
            internal_match_id,
            entry.player_name,
            entry.number,
        )
        lineup.append(
            {
                "id": stable_identifier(
                    "lineup", internal_match_id, internal_player_id, order, length=24
                ),
                "playerId": internal_player_id,
                "playerName": entry.player_name,
                "teamId": team_ids.get(str(entry.team_id or "")),
                "teamName": entry.team_name,
                "side": entry.side,
                "position": entry.position,
                "number": entry.number,
                "role": entry.role,
                "order": entry.order,
                "formation": entry.formation,
                "grid": entry.grid,
            }
        )

    events: list[dict[str, Any]] = []
    for order, item in enumerate(bundle.timeline):
        events.append(
            {
                "id": _canonical_id(
                    "event",
                    provider,
                    item.id,
                    internal_match_id,
                    item.minute,
                    item.type,
                    item.player_name,
                    order,
                ),
                "minute": item.minute,
                "type": _event_type(item.type, item.detail),
                "label": item.label,
                "playerId": player_ids.get(str(item.player_id or "")),
                "playerName": item.player_name,
                "secondaryPlayerId": player_ids.get(
                    str(item.secondary_player_id or "")
                ),
                "secondaryPlayerName": item.secondary_player_name,
                "teamId": team_ids.get(str(item.team_id or "")),
                "teamName": item.team_name,
                "detail": item.detail,
            }
        )

    substitutions: list[dict[str, Any]] = []
    for order, item in enumerate(bundle.substitutions):
        substitutions.append(
            {
                "id": _canonical_id(
                    "substitution",
                    provider,
                    item.id,
                    internal_match_id,
                    item.minute,
                    item.player_out_name,
                    item.player_in_name,
                    order,
                ),
                "minute": item.minute,
                "teamId": team_ids.get(str(item.team_id or "")),
                "teamName": item.team_name,
                "playerOutId": player_ids.get(str(item.player_out_id or "")),
                "playerOutName": item.player_out_name,
                "playerInId": player_ids.get(str(item.player_in_id or "")),
                "playerInName": item.player_in_name,
                "label": item.label,
            }
        )

    quality = bundle.roster_quality
    if quality is None:
        home_count = sum(item.get("teamId") == home_id for item in roster)
        away_count = sum(item.get("teamId") == away_id for item in roster)
        automatic = bool(roster) and home_count >= 11 and away_count >= 11
        quality_payload = {
            "status": "automatic-ready" if automatic else "partial" if roster else "unavailable",
            "playerCount": len(roster),
            "homePlayerCount": home_count,
            "awayPlayerCount": away_count,
            "automaticIdentityEligible": automatic,
            "manualIdentityEligible": bool(roster),
            "reasons": [] if automatic else ["canonical-roster-incomplete"],
        }
    else:
        quality_payload = {
            "status": quality.status,
            "playerCount": quality.player_count,
            "homePlayerCount": quality.home_player_count,
            "awayPlayerCount": quality.away_player_count,
            "automaticIdentityEligible": quality.automatic_identity_eligible,
            "manualIdentityEligible": quality.manual_identity_eligible,
            "reasons": list(quality.reasons),
        }

    return {
        "schemaVersion": CANONICAL_MATCH_SCHEMA_VERSION,
        "id": internal_match_id,
        "name": bundle.event.name,
        "competition": bundle.event.league,
        "season": bundle.event.season,
        "date": bundle.event.date,
        "time": bundle.event.time,
        "status": bundle.event.status,
        "score": {
            "home": bundle.event.home_score,
            "away": bundle.event.away_score,
        },
        "homeTeam": {
            "id": home_id,
            "name": bundle.event.home.name,
            "badge": bundle.event.home.badge,
        },
        "awayTeam": {
            "id": away_id,
            "name": bundle.event.away.name,
            "badge": bundle.event.away.badge,
        },
        "roster": roster,
        "lineup": lineup,
        "events": events,
        "substitutions": substitutions,
        "rosterQuality": quality_payload,
        "sync": {
            "state": (
                "ready"
                if quality_payload["status"] == "automatic-ready"
                else "partial"
                if roster or events
                else "unavailable"
            ),
            "syncedAt": bundle.fetched_at,
            "stale": False,
            "warnings": list(bundle.warnings),
        },
    }
