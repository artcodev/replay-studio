from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
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
from .base import (
    MatchDataError,
    MatchDataEventNotFound,
    MatchDataProviderNotConfigured,
)


class ApiFootballError(MatchDataError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream-error",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=ApiFootballProvider.id,
            code=code,
            retryable=retryable,
        )


class ApiFootballProvider:
    """API-Football adapter exposing the application's normalized contract."""

    id = "api-football"
    name = "API-Football"
    capabilities = ("fixtures", "search", "lineups", "events")

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.base_url = settings.api_football_base_url.rstrip("/")
        self._api_key = (settings.api_football_api_key or "").strip()
        self._credential_scope = (
            hashlib.sha256(self._api_key.encode("utf-8")).hexdigest()[:12]
            if self._api_key
            else "unconfigured"
        )
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._redis = (
            Redis.from_url(settings.redis_url, decode_responses=True)
            if settings.redis_url
            else None
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "API_FOOTBALL_API_KEY is not configured"

    @staticmethod
    def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
        return f"{endpoint}:{json.dumps(params, sort_keys=True, separators=(',', ':'))}"

    @staticmethod
    def _upstream_errors(data: dict[str, Any]) -> list[str]:
        errors = data.get("errors")
        if isinstance(errors, dict):
            return [str(value) for value in errors.values() if value]
        if isinstance(errors, list):
            return [str(value) for value in errors if value]
        if errors:
            return [str(errors)]
        return []

    async def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
        ttl: int = 300,
    ) -> dict[str, Any]:
        if not self.configured:
            raise MatchDataProviderNotConfigured(self.id, self.name)
        key = self._cache_key(endpoint, params)
        now = asyncio.get_running_loop().time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < ttl:
            return cached[1]
        if self._redis:
            try:
                remote_cached = await self._redis.get(
                    f"match-data:{self.id}:{self._credential_scope}:{key}"
                )
                if remote_cached:
                    data = json.loads(remote_cached)
                    self._cache[key] = (now, data)
                    return data
            except (RedisError, ValueError, TypeError):
                pass

        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint.lstrip('/')}",
                    params=params,
                    headers={"x-apisports-key": self._api_key},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                raise ApiFootballError(
                    "API-Football rejected the server credential or plan coverage",
                    code="provider-auth-or-coverage",
                ) from exc
            if status == 429:
                raise ApiFootballError(
                    "API-Football request quota was exceeded",
                    code="provider-rate-limit",
                    retryable=True,
                ) from exc
            raise ApiFootballError(
                f"API-Football returned HTTP {status}",
                retryable=status >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiFootballError(
                "API-Football could not be reached",
                code="provider-unreachable",
                retryable=True,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise ApiFootballError(
                "API-Football returned an invalid response",
                code="invalid-provider-response",
            ) from exc

        if not isinstance(data, dict):
            raise ApiFootballError(
                "API-Football returned an invalid response",
                code="invalid-provider-response",
            )
        errors = self._upstream_errors(data)
        if errors:
            # API-Sports often returns validation, coverage and quota errors in
            # a successful HTTP response. Keep their text out of the public
            # message because it may echo account-specific information.
            raise ApiFootballError(
                "API-Football rejected the request",
                code="provider-request-rejected",
                retryable=any("limit" in item.lower() for item in errors),
            )

        self._cache[key] = (now, data)
        if self._redis:
            try:
                await self._redis.setex(
                    f"match-data:{self.id}:{self._credential_scope}:{key}",
                    ttl,
                    json.dumps(data),
                )
            except RedisError:
                pass
        return data

    @staticmethod
    def _response_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
        response = data.get("response")
        return [item for item in response if isinstance(item, dict)] if isinstance(response, list) else []

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _text(value: Any) -> str | None:
        text = str(value).strip() if value not in (None, "") else ""
        return text or None

    @classmethod
    def _map_fixture(cls, item: dict[str, Any]) -> ExternalEvent:
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        goals = item.get("goals") or {}
        status = fixture.get("status") or {}
        date_time = cls._text(fixture.get("date"))
        date_value = date_time[:10] if date_time else None
        time_value = None
        if date_time and "T" in date_time:
            time_value = date_time.split("T", 1)[1][:5]
        home_name = cls._text(home.get("name")) or "Home"
        away_name = cls._text(away.get("name")) or "Away"
        return ExternalEvent(
            id=str(fixture.get("id") or "unknown"),
            provider=cls.id,
            name=f"{home_name} vs {away_name}",
            date=date_value,
            time=time_value,
            status=cls._text(status.get("long") or status.get("short")),
            league=cls._text(league.get("name")),
            season=cls._text(league.get("season")),
            home=ExternalTeam(
                id=str(home.get("id") or "home"),
                name=home_name,
                badge=cls._text(home.get("logo")),
            ),
            away=ExternalTeam(
                id=str(away.get("id") or "away"),
                name=away_name,
                badge=cls._text(away.get("logo")),
            ),
            home_score=cls._integer(goals.get("home")),
            away_score=cls._integer(goals.get("away")),
            thumbnail=None,
        )

    async def events_by_date(self, date: str) -> list[ExternalEvent]:
        data = await self._get("fixtures", {"date": date}, ttl=180)
        return [self._map_fixture(item) for item in self._response_rows(data)]

    @staticmethod
    def _normalize_name(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        return "".join(character for character in decomposed.lower() if character.isalnum())

    async def _team_id(self, name: str) -> str | None:
        data = await self._get("teams", {"search": name.strip()}, ttl=3600)
        rows = self._response_rows(data)
        if not rows:
            return None
        expected = self._normalize_name(name)
        exact = next(
            (
                row
                for row in rows
                if self._normalize_name(str((row.get("team") or {}).get("name") or ""))
                == expected
            ),
            rows[0],
        )
        team_id = (exact.get("team") or {}).get("id")
        return str(team_id) if team_id not in (None, "") else None

    async def search_events(self, query: str) -> list[ExternalEvent]:
        pair = re.split(r"\s+(?:vs\.?|v\.?|@|[-–—])\s+", query.strip(), maxsplit=1, flags=re.IGNORECASE)
        if len(pair) != 2 or not all(part.strip() for part in pair):
            raise ApiFootballError(
                "API-Football match search requires two teams, for example 'Spain vs Belgium'",
                code="team-pair-required",
            )
        home_id, away_id = await asyncio.gather(
            self._team_id(pair[0]),
            self._team_id(pair[1]),
        )
        if not home_id or not away_id:
            return []
        data = await self._get(
            "fixtures/headtohead",
            {"h2h": f"{home_id}-{away_id}", "last": 40},
            ttl=600,
        )
        expected_team_ids = {home_id, away_id}
        events = [
            self._map_fixture(item)
            for item in self._response_rows(data)
            if {
                str(((item.get("teams") or {}).get("home") or {}).get("id") or ""),
                str(((item.get("teams") or {}).get("away") or {}).get("id") or ""),
            }
            == expected_team_ids
        ]
        return sorted(events, key=lambda event: (event.date or "", event.time or ""), reverse=True)

    @classmethod
    def _map_lineups(
        cls,
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
            team_id = str(team.get("id")) if team.get("id") not in (None, "") else None
            side, canonical_team = by_team.get(team_id or "", ("unknown", None))
            team_name = (
                canonical_team.name
                if canonical_team is not None
                else cls._text(team.get("name"))
            )
            formation = cls._text(row.get("formation"))
            for role, key in (("starter", "startXI"), ("substitute", "substitutes")):
                source_players = row.get(key) or []
                if not isinstance(source_players, list):
                    continue
                for source in source_players:
                    player = source.get("player") if isinstance(source, dict) else None
                    if not isinstance(player, dict):
                        continue
                    raw_id = player.get("id")
                    player_id = str(raw_id) if raw_id not in (None, "") else f"unknown-{order}"
                    player_name = cls._text(player.get("name")) or "Unknown player"
                    number = cls._text(player.get("number"))
                    position = cls._text(player.get("pos"))
                    entries.append(
                        ExternalLineupEntry(
                            id=f"{event.id}:{team_id or side}:{role}:{player_id}:{order}",
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
                            grid=cls._text(player.get("grid")),
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

    @classmethod
    def _map_timeline(
        cls,
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
            raw_type = cls._text(row.get("type")) or "event"
            detail = cls._text(row.get("detail") or row.get("comments"))
            normalized_type = re.sub(r"[^a-z0-9]+", "-", raw_type.lower()).strip("-") or "event"
            player_name = cls._text(player.get("name"))
            secondary_name = cls._text(secondary.get("name"))
            elapsed = cls._integer(time.get("elapsed"))
            extra = cls._integer(time.get("extra")) or 0
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
                    "comments": cls._text(row.get("comments")),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
            occurrence = signature_occurrences.get(digest, 0)
            signature_occurrences[digest] = occurrence + 1
            timeline_id = f"{event.id}:event:{digest}:{occurrence}"
            label = " · ".join(
                value
                for value in (clock, raw_type, player_name)
                if value
            )
            item = ExternalTimelineEvent(
                id=timeline_id,
                minute=minute,
                type=normalized_type,
                label=label,
                player_id=cls._text(player.get("id")),
                player_name=player_name,
                team_id=cls._text(team.get("id")),
                team_name=cls._text(team.get("name")),
                secondary_player_id=cls._text(secondary.get("id")),
                secondary_player_name=secondary_name,
                detail=detail,
            )
            timeline.append(item)
            if normalized_type in {"subst", "substitution"} or "substitut" in normalized_type:
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

    @staticmethod
    def _roster_quality(
        players: list[ExternalPlayer],
        lineup: list[ExternalLineupEntry],
        event: ExternalEvent,
        substitutions: list[ExternalSubstitution],
    ) -> ExternalRosterQuality:
        home_count = sum(player.team_id == event.home.id for player in players)
        away_count = sum(player.team_id == event.away.id for player in players)
        home_starters = sum(entry.side == "home" and entry.role == "starter" for entry in lineup)
        away_starters = sum(entry.side == "away" and entry.role == "starter" for entry in lineup)
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
            status="automatic-ready" if automatic else "partial" if players else "unavailable",
            player_count=len(players),
            home_player_count=home_count,
            away_player_count=away_count,
            automatic_identity_eligible=automatic,
            manual_identity_eligible=bool(players),
            reasons=reasons,
        )

    async def event_bundle(self, event_id: str) -> EventBundle:
        fixture_data = await self._get("fixtures", {"id": event_id}, ttl=600)
        fixtures = self._response_rows(fixture_data)
        if not fixtures:
            raise MatchDataEventNotFound(self.id, event_id)
        fixture = fixtures[0]
        event = self._map_fixture(fixture)

        # A fixture-id response can embed both blocks. This keeps a complete
        # bundle to one request on API-Football's small free quota. Dedicated
        # endpoints are only fallbacks when the keys are absent, not when a
        # provider explicitly reports an empty block.
        embedded_lineups = fixture.get("lineups") if "lineups" in fixture else None
        embedded_events = fixture.get("events") if "events" in fixture else None
        pending: list[tuple[str, Any]] = []
        if embedded_lineups is None:
            pending.append(
                (
                    "lineups",
                    self._get("fixtures/lineups", {"fixture": event_id}, ttl=900),
                )
            )
        if embedded_events is None:
            pending.append(
                (
                    "events",
                    self._get("fixtures/events", {"fixture": event_id}, ttl=300),
                )
            )
        fetched: dict[str, Any] = {}
        if pending:
            results = await asyncio.gather(
                *(request for _, request in pending),
                return_exceptions=True,
            )
            fetched = {
                name: result for (name, _), result in zip(pending, results, strict=True)
            }
        lineup_result: Any = (
            {"response": embedded_lineups}
            if embedded_lineups is not None
            else fetched.get("lineups")
        )
        events_result: Any = (
            {"response": embedded_events}
            if embedded_events is not None
            else fetched.get("events")
        )
        warnings: list[str] = []
        lineup_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        if isinstance(lineup_result, MatchDataError):
            warnings.append("API-Football lineup is unavailable for this match or subscription; manual roster binding remains available.")
        elif isinstance(lineup_result, Exception):
            raise lineup_result
        else:
            lineup_rows = self._response_rows(lineup_result)
            if not lineup_rows:
                warnings.append("API-Football returned no lineup for this match; it may not be published or covered yet.")
        if isinstance(events_result, MatchDataError):
            warnings.append("API-Football match events are unavailable for this match or subscription.")
        elif isinstance(events_result, Exception):
            raise events_result
        else:
            event_rows = self._response_rows(events_result)
            if not event_rows:
                warnings.append("API-Football returned no event timeline for this match.")

        players, lineup = self._map_lineups(lineup_rows, event)
        timeline, substitutions = self._map_timeline(event_rows, event)
        quality = self._roster_quality(players, lineup, event, substitutions)
        if players and not quality.automatic_identity_eligible:
            warnings.append("The API-Football lineup is partial for automatic identity; available players can still be bound manually.")
        return EventBundle(
            source=self.id,
            event=event,
            players=players,
            lineup=lineup,
            timeline=timeline,
            substitutions=substitutions,
            roster_quality=quality,
            fetched_at=datetime.now(UTC).isoformat(),
            warnings=warnings,
        )
