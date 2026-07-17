from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..config import Settings, get_settings
from ..schemas import (
    EventBundle,
    ExternalEvent,
    ExternalLineupEntry,
    ExternalPlayer,
    ExternalRosterQuality,
    ExternalSubstitution,
    ExternalTeam,
    ExternalTimelineEvent,
)
from .base import MatchDataError, MatchDataEventNotFound, MatchDataProviderNotConfigured


class SportsDbError(MatchDataError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream-error",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=TheSportsDbProvider.id,
            code=code,
            retryable=retryable,
        )


class TheSportsDbProvider:
    id = "thesportsdb"
    name = "TheSportsDB"
    capabilities = ("fixtures", "search", "lineups", "events")

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.base_url = f"{settings.sportsdb_base_url.rstrip('/')}/{settings.sportsdb_api_key}"
        self._api_key = settings.sportsdb_api_key.strip()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._redis = Redis.from_url(settings.redis_url, decode_responses=True) if settings.redis_url else None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "SPORTSDB_API_KEY is not configured"

    async def _get(self, endpoint: str, params: dict[str, Any], ttl: int = 300) -> dict:
        if not self.configured:
            raise MatchDataProviderNotConfigured(self.id, self.name)
        key = f"{endpoint}:{sorted(params.items())}"
        now = asyncio.get_running_loop().time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < ttl:
            return cached[1]
        if self._redis:
            try:
                remote_cached = await self._redis.get(f"sportsdb:{key}")
                if remote_cached:
                    data = json.loads(remote_cached)
                    self._cache[key] = (now, data)
                    return data
            except (RedisError, ValueError):
                pass
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                response = await client.get(f"{self.base_url}/{endpoint}", params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise SportsDbError(
                f"TheSportsDB returned HTTP {status}",
                code="provider-rate-limit" if status == 429 else "upstream-error",
                retryable=status == 429 or status >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise SportsDbError(
                "TheSportsDB could not be reached",
                code="provider-unreachable",
                retryable=True,
            ) from exc
        except ValueError as exc:
            raise SportsDbError(
                "TheSportsDB returned an invalid response",
                code="invalid-provider-response",
            ) from exc
        self._cache[key] = (now, data)
        if self._redis:
            try:
                await self._redis.setex(f"sportsdb:{key}", ttl, json.dumps(data))
            except RedisError:
                pass
        return data

    @staticmethod
    def _score(value: Any) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    def _map_event(self, item: dict) -> ExternalEvent:
        return ExternalEvent(
            id=str(item.get("idEvent") or "unknown"),
            provider=self.id,
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
            home_score=self._score(item.get("intHomeScore")),
            away_score=self._score(item.get("intAwayScore")),
            thumbnail=item.get("strThumb") or item.get("strPoster"),
        )

    async def events_by_date(self, date: str) -> list[ExternalEvent]:
        data = await self._get("eventsday.php", {"d": date, "s": "Soccer"}, ttl=180)
        return [self._map_event(item) for item in data.get("events") or []]

    async def search_events(self, query: str) -> list[ExternalEvent]:
        normalized = re.sub(r"\s+(?:vs\.?|v\.?|@)\s+", "_vs_", query.strip(), flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", "_", normalized)
        data = await self._get("searchevents.php", {"e": normalized}, ttl=600)
        events = data.get("event") or data.get("events") or []
        return [
            self._map_event(item)
            for item in events
            if str(item.get("strSport") or "Soccer").lower() == "soccer"
        ]

    @staticmethod
    def _lineup_side(item: dict) -> str:
        side = str(item.get("strHome") or "").lower()
        if side == "yes":
            return "home"
        if side == "no":
            return "away"
        return "unknown"

    @staticmethod
    def _lineup_role(item: dict) -> str:
        substitute = str(item.get("strSubstitute") or "").strip().lower()
        if substitute in {"yes", "true", "1", "substitute", "sub"}:
            return "substitute"
        if substitute in {"no", "false", "0", "starter", "start"}:
            return "starter"
        return "unknown"

    @classmethod
    def _map_lineup_entry(
        cls,
        item: dict,
        event: ExternalEvent,
        index: int,
    ) -> ExternalLineupEntry:
        side = cls._lineup_side(item)
        if side == "home":
            team_id, team_name = event.home.id, event.home.name
        elif side == "away":
            team_id, team_name = event.away.id, event.away.name
        else:
            team_id = str(item.get("idTeam")) if item.get("idTeam") else None
            team_name = item.get("strTeam")
        player_id = str(item.get("idPlayer") or f"lineup-player-{index}")
        return ExternalLineupEntry(
            id=str(item.get("idLineup") or f"lineup-{index}"),
            player_id=player_id,
            player_name=item.get("strPlayer") or item.get("strPlayerName") or "Unknown player",
            team_id=team_id,
            team_name=team_name,
            side=side,
            position=item.get("strPosition"),
            number=str(item.get("intSquadNumber")) if item.get("intSquadNumber") not in (None, "") else None,
            role=cls._lineup_role(item),
            order=index,
        )

    @staticmethod
    def _lineup_player(entry: ExternalLineupEntry, item: dict) -> ExternalPlayer:
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

    @classmethod
    def _map_lineup_player(
        cls,
        item: dict,
        event: ExternalEvent,
        index: int,
    ) -> ExternalPlayer:
        """Backward-compatible single-row mapper used by provider callers/tests."""

        entry = cls._map_lineup_entry(item, event, index)
        return cls._lineup_player(entry, item)

    @staticmethod
    def _roster_quality(
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

    async def event_bundle(self, event_id: str) -> EventBundle:
        event_data, lineup_data, timeline_data = await asyncio.gather(
            self._get("lookupevent.php", {"id": event_id}, ttl=600),
            self._get("lookuplineup.php", {"id": event_id}, ttl=900),
            self._get("lookuptimeline.php", {"id": event_id}, ttl=300),
        )
        events = event_data.get("events") or []
        if not events:
            raise MatchDataEventNotFound(self.id, event_id)
        event = self._map_event(events[0])

        lineup = lineup_data.get("lineup") or lineup_data.get("players") or []
        players: list[ExternalPlayer] = []
        lineup_entries: list[ExternalLineupEntry] = []
        seen: set[str] = set()
        for index, item in enumerate(lineup):
            entry = self._map_lineup_entry(item, event, index)
            lineup_entries.append(entry)
            player = self._lineup_player(entry, item)
            if player.id in seen:
                continue
            seen.add(player.id)
            players.append(player)

        raw_timeline = timeline_data.get("timeline") or timeline_data.get("events") or []
        timeline: list[ExternalTimelineEvent] = []
        substitutions: list[ExternalSubstitution] = []
        for index, item in enumerate(raw_timeline):
            minute = self._score(item.get("intTime") or item.get("strTime"))
            event_type = item.get("strTimeline") or item.get("strType") or "event"
            player_name = item.get("strPlayer") or item.get("strPlayerName")
            secondary_player_name = item.get("strPlayer2") or item.get("strPlayerName2")
            timeline_id = str(item.get("idTimeline") or f"timeline-{index}")
            team_id = str(item.get("idTeam")) if item.get("idTeam") else None
            team_name = item.get("strTeam")
            player_id = str(item.get("idPlayer")) if item.get("idPlayer") else None
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

        roster_quality = self._roster_quality(players, event)
        warnings = []
        if not players:
            warnings.append("The free source returned no lineup for this event; bindings can be entered manually.")
        elif len(players) == 5:
            warnings.append(
                "The free source returned only five lineup players; automatic identity is disabled, but manual binding remains available."
            )
        elif not roster_quality.automatic_identity_eligible:
            warnings.append(
                "The lineup is incomplete for automatic identity; available players can still be bound manually."
            )
        if not timeline:
            warnings.append("The free source returned no event timeline for this match.")
        elif len(timeline) == 5:
            warnings.append("The free source returned only the first five timeline events.")
        return EventBundle(
            event=event,
            players=players,
            lineup=lineup_entries,
            timeline=timeline,
            substitutions=substitutions,
            roster_quality=roster_quality,
            fetched_at=datetime.now(UTC).isoformat(),
            warnings=warnings,
        )
