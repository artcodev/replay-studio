from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .match_contracts import (
    EventBundle,
    ExternalEvent,
    ExternalRosterQuality,
    ManualMatchImportRequest,
)


def roster_quality_payload(bundle: EventBundle) -> dict[str, Any]:
    """Return the canonical roster-quality contract for a normalized bundle."""

    quality = bundle.roster_quality
    if quality is not None:
        return {
            "status": quality.status,
            "playerCount": quality.player_count,
            "homePlayerCount": quality.home_player_count,
            "awayPlayerCount": quality.away_player_count,
            "automaticIdentityEligible": quality.automatic_identity_eligible,
            "manualIdentityEligible": quality.manual_identity_eligible,
            "reasons": list(quality.reasons),
        }

    home_count = sum(
        player.team_id == bundle.event.home.id for player in bundle.players
    )
    away_count = sum(
        player.team_id == bundle.event.away.id for player in bundle.players
    )
    reasons: list[str] = []
    if not bundle.players:
        reasons.append("roster-unavailable")
    if bundle.players and (home_count < 11 or away_count < 11):
        reasons.append("fewer-than-eleven-players-per-team")
    automatic = bool(bundle.players) and not reasons
    return {
        "status": (
            "automatic-ready"
            if automatic
            else "partial"
            if bundle.players
            else "unavailable"
        ),
        "playerCount": len(bundle.players),
        "homePlayerCount": home_count,
        "awayPlayerCount": away_count,
        "automaticIdentityEligible": automatic,
        "manualIdentityEligible": bool(bundle.players),
        "reasons": reasons,
    }


def _identifier(value: object, label: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{label} must not be empty")
    if len(identifier) > 160:
        raise ValueError(f"{label} is longer than 160 characters")
    return identifier


@dataclass
class ManualMatchNormalizer:
    """Validate references and normalize one manually supplied match graph.

    The normalizer owns only import-time graph validation. Persistence and the
    provider-neutral canonical-ID mapping remain separate application steps.
    """

    request: ManualMatchImportRequest
    teams: dict[str, tuple[str, dict[str, Any]]] = field(init=False)
    players: list[dict[str, Any]] = field(default_factory=list, init=False)
    players_by_id: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def normalize(self) -> tuple[EventBundle, dict[str, Any]]:
        event_id, home, away = self._normalize_teams()
        self._normalize_players()
        lineup = self._normalize_lineup()
        timeline = self._normalize_timeline()
        substitutions = self._normalize_substitutions()
        imported_at = datetime.now(UTC).isoformat()

        event = ExternalEvent.model_validate(
            {
                **self.request.event.model_dump(),
                "id": event_id,
                "home": home,
                "away": away,
            }
        )
        bundle = EventBundle.model_validate(
            {
                "source": "manual",
                "event": event,
                "players": self.players,
                "lineup": lineup,
                "timeline": timeline,
                "substitutions": substitutions,
                "fetched_at": imported_at,
                "warnings": [],
            }
        )
        self._attach_roster_quality(bundle)
        return bundle, self._provenance(imported_at)

    def _normalize_teams(self) -> tuple[str, dict[str, Any], dict[str, Any]]:
        event_id = _identifier(self.request.event.id, "event.id")
        home = self.request.teams.home.model_dump()
        away = self.request.teams.away.model_dump()
        home["id"] = _identifier(home["id"], "teams.home.id")
        away["id"] = _identifier(away["id"], "teams.away.id")
        if home["id"] == away["id"]:
            raise ValueError("Home and away teams must have different ids")
        if not str(home.get("name") or "").strip() or not str(
            away.get("name") or ""
        ).strip():
            raise ValueError("Both teams must have a name")
        self.teams = {
            home["id"]: ("home", home),
            away["id"]: ("away", away),
        }
        return event_id, home, away

    def _normalize_players(self) -> None:
        jersey_owners: set[tuple[str, str]] = set()
        for index, source in enumerate(self.request.players):
            player = source.model_dump()
            player_id = _identifier(player["id"], f"players[{index}].id")
            if player_id in self.players_by_id:
                raise ValueError(f"Duplicate player id: {player_id}")
            if not str(player.get("name") or "").strip():
                raise ValueError(f"players[{index}].name must not be empty")
            team_id = _identifier(
                player.get("team_id"),
                f"players[{index}].team_id",
            )
            if team_id not in self.teams:
                raise ValueError(f"Player {player_id} references an unknown team")
            team_name = self.teams[team_id][1]["name"]
            supplied_team_name = str(player.get("team_name") or "").strip()
            if supplied_team_name and supplied_team_name != team_name:
                raise ValueError(
                    f"Player {player_id} has a team_name that disagrees with teams"
                )
            number = str(player.get("number") or "").strip() or None
            if number is not None:
                owner = (team_id, number)
                if owner in jersey_owners:
                    raise ValueError(
                        f"Team {team_id} has duplicate jersey number {number}"
                    )
                jersey_owners.add(owner)
            player.update(
                {
                    "id": player_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "number": number,
                    "lineup_order": (
                        int(player["lineup_order"])
                        if player.get("lineup_order") is not None
                        else index
                    ),
                }
            )
            self.players.append(player)
            self.players_by_id[player_id] = player

    def _normalize_lineup(self) -> list[dict[str, Any]]:
        if not self.request.lineup:
            return [self._default_lineup_entry(player) for player in self.players]

        lineup: list[dict[str, Any]] = []
        ids: set[str] = set()
        player_ids: set[str] = set()
        orders: set[int] = set()
        for index, source in enumerate(self.request.lineup):
            entry = source.model_dump()
            entry_id = _identifier(entry["id"], f"lineup[{index}].id")
            player_id = _identifier(entry["player_id"], f"lineup[{index}].player_id")
            if entry_id in ids:
                raise ValueError(f"Duplicate lineup id: {entry_id}")
            if player_id in player_ids:
                raise ValueError(f"Player {player_id} appears twice in the lineup")
            player = self.players_by_id.get(player_id)
            if player is None:
                raise ValueError(f"Lineup references unknown player {player_id}")
            order = int(entry["order"])
            if order in orders:
                raise ValueError(f"Duplicate lineup order: {order}")
            team_id = str(entry.get("team_id") or player["team_id"])
            if team_id != player["team_id"]:
                raise ValueError(f"Lineup team disagrees for player {player_id}")
            expected_side, team = self.teams[team_id]
            if entry.get("side") not in {"unknown", expected_side}:
                raise ValueError(f"Lineup side disagrees for player {player_id}")
            entry.update(
                {
                    "id": entry_id,
                    "player_id": player_id,
                    "player_name": player["name"],
                    "team_id": team_id,
                    "team_name": team["name"],
                    "side": expected_side,
                    "number": player.get("number"),
                    "order": order,
                }
            )
            player["lineup_role"] = entry["role"]
            player["lineup_order"] = order
            ids.add(entry_id)
            player_ids.add(player_id)
            orders.add(order)
            lineup.append(entry)
        return lineup

    def _default_lineup_entry(self, player: dict[str, Any]) -> dict[str, Any]:
        side, team = self.teams[player["team_id"]]
        return {
            "id": f"manual-lineup-{player['id']}",
            "player_id": player["id"],
            "player_name": player["name"],
            "team_id": player["team_id"],
            "team_name": team["name"],
            "side": side,
            "position": player.get("position"),
            "number": player.get("number"),
            "role": player.get("lineup_role") or "unknown",
            "order": int(player["lineup_order"]),
        }

    def _normalize_timeline(self) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        ids: set[str] = set()
        for index, source in enumerate(self.request.timeline):
            item = source.model_dump()
            item_id = _identifier(item["id"], f"timeline[{index}].id")
            if item_id in ids:
                raise ValueError(f"Duplicate timeline id: {item_id}")
            primary = self.players_by_id.get(str(item.get("player_id") or ""))
            secondary = self.players_by_id.get(
                str(item.get("secondary_player_id") or "")
            )
            if item.get("player_id") and primary is None:
                raise ValueError(
                    f"Timeline references unknown player {item['player_id']}"
                )
            if item.get("secondary_player_id") and secondary is None:
                raise ValueError(
                    "Timeline references unknown player "
                    f"{item['secondary_player_id']}"
                )
            inferred_team_id = (
                primary.get("team_id")
                if primary
                else secondary.get("team_id")
                if secondary
                else None
            )
            team_id = str(item.get("team_id") or inferred_team_id or "") or None
            if team_id is not None and team_id not in self.teams:
                raise ValueError(
                    f"Timeline event {item_id} references an unknown team"
                )
            item.update(
                {
                    "id": item_id,
                    "player_name": (
                        primary["name"] if primary else item.get("player_name")
                    ),
                    "secondary_player_name": (
                        secondary["name"]
                        if secondary
                        else item.get("secondary_player_name")
                    ),
                    "team_id": team_id,
                    "team_name": self.teams[team_id][1]["name"] if team_id else None,
                }
            )
            ids.add(item_id)
            timeline.append(item)
        return timeline

    def _normalize_substitutions(self) -> list[dict[str, Any]]:
        substitutions: list[dict[str, Any]] = []
        ids: set[str] = set()
        for index, source in enumerate(self.request.substitutions):
            item = source.model_dump()
            item_id = _identifier(item["id"], f"substitutions[{index}].id")
            if item_id in ids:
                raise ValueError(f"Duplicate substitution id: {item_id}")
            out_id = _identifier(
                item.get("player_out_id"),
                f"substitutions[{index}].player_out_id",
            )
            in_id = _identifier(
                item.get("player_in_id"),
                f"substitutions[{index}].player_in_id",
            )
            outgoing = self.players_by_id.get(out_id)
            incoming = self.players_by_id.get(in_id)
            if outgoing is None or incoming is None:
                raise ValueError(
                    f"Substitution {item_id} references an unknown player"
                )
            if out_id == in_id or outgoing["team_id"] != incoming["team_id"]:
                raise ValueError(
                    f"Substitution {item_id} must exchange two players on one team"
                )
            team_id = str(item.get("team_id") or outgoing["team_id"])
            if team_id != outgoing["team_id"]:
                raise ValueError(f"Substitution {item_id} has a conflicting team")
            item.update(
                {
                    "id": item_id,
                    "team_id": team_id,
                    "team_name": self.teams[team_id][1]["name"],
                    "player_out_id": out_id,
                    "player_out_name": outgoing["name"],
                    "player_in_id": in_id,
                    "player_in_name": incoming["name"],
                }
            )
            ids.add(item_id)
            substitutions.append(item)
        return substitutions

    def _attach_roster_quality(self, bundle: EventBundle) -> None:
        quality = roster_quality_payload(bundle)
        if not quality["automaticIdentityEligible"]:
            bundle.warnings.append(
                "The manually imported roster is incomplete for automatic identity; "
                "available players can still be bound manually."
            )
        bundle.roster_quality = ExternalRosterQuality(
            status=quality["status"],
            player_count=quality["playerCount"],
            home_player_count=quality["homePlayerCount"],
            away_player_count=quality["awayPlayerCount"],
            automatic_identity_eligible=quality["automaticIdentityEligible"],
            manual_identity_eligible=quality["manualIdentityEligible"],
            reasons=quality["reasons"],
        )

    def _provenance(self, imported_at: str) -> dict[str, Any]:
        supplied = self.request.provenance
        captured_at = supplied.captured_at if supplied else None
        if captured_at is not None and captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=UTC)
        return {
            "kind": "manual-json",
            "importedAt": imported_at,
            "capturedAt": (
                captured_at.astimezone(UTC).isoformat() if captured_at else None
            ),
            "label": supplied.label if supplied else None,
            "reference": supplied.reference if supplied else None,
            "notes": supplied.notes if supplied else None,
        }


def build_manual_match_bundle(
    request: ManualMatchImportRequest,
) -> tuple[EventBundle, dict[str, Any]]:
    return ManualMatchNormalizer(request).normalize()
