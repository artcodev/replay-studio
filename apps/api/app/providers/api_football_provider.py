from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from ..config import Settings
from ..match_contracts import EventBundle, ExternalEvent
from .api_football_errors import ApiFootballError
from .api_football_mapping import (
    assess_roster_quality,
    map_fixture,
    map_lineups,
    map_timeline,
    normalize_name,
    response_rows,
)
from .api_football_transport import ApiFootballClient
from .base import MatchDataError, MatchDataEventNotFound


class ApiFootballGateway(Protocol):
    @property
    def configured(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any],
        ttl: int = 300,
    ) -> dict[str, Any]: ...


class ApiFootballProvider:
    """Coordinates API-Football requests into the provider-neutral contract."""

    id = "api-football"
    name = "API-Football"
    capabilities = ("fixtures", "search", "lineups", "events")

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: ApiFootballGateway | None = None,
    ) -> None:
        self._client = client or ApiFootballClient(settings)

    @property
    def configured(self) -> bool:
        return self._client.configured

    @property
    def unavailable_reason(self) -> str | None:
        return self._client.unavailable_reason

    async def events_by_date(self, date: str) -> list[ExternalEvent]:
        data = await self._client.get("fixtures", {"date": date}, ttl=180)
        return [map_fixture(item) for item in response_rows(data)]

    async def search_events(self, query: str) -> list[ExternalEvent]:
        pair = re.split(
            r"\s+(?:vs\.?|v\.?|@|[-–—])\s+",
            query.strip(),
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(pair) != 2 or not all(part.strip() for part in pair):
            raise ApiFootballError(
                "API-Football match search requires two teams, "
                "for example 'Spain vs Belgium'",
                code="team-pair-required",
            )
        home_id, away_id = await asyncio.gather(
            self._team_id(pair[0]),
            self._team_id(pair[1]),
        )
        if not home_id or not away_id:
            return []
        data = await self._client.get(
            "fixtures/headtohead",
            # ``h2h`` is available on the Free plan, while ``last`` is often
            # plan-gated. Filter and sort the complete response locally.
            {"h2h": f"{home_id}-{away_id}"},
            ttl=600,
        )
        expected_team_ids = {home_id, away_id}
        events = [
            map_fixture(item)
            for item in response_rows(data)
            if _fixture_team_ids(item) == expected_team_ids
        ]
        return sorted(
            events,
            key=lambda event: (event.date or "", event.time or ""),
            reverse=True,
        )

    async def event_bundle(self, event_id: str) -> EventBundle:
        fixture_data = await self._client.get(
            "fixtures",
            {"id": event_id},
            ttl=600,
        )
        fixtures = response_rows(fixture_data)
        if not fixtures:
            raise MatchDataEventNotFound(self.id, event_id)
        fixture = fixtures[0]
        event = map_fixture(fixture)

        lineup_result, events_result = await self._fixture_details(
            fixture,
            event_id,
        )
        warnings: list[str] = []
        lineup_rows = _optional_rows(
            lineup_result,
            unavailable_warning=(
                "API-Football lineup is unavailable for this match or "
                "subscription; manual roster binding remains available."
            ),
            empty_warning=(
                "API-Football returned no lineup for this match; it may not "
                "be published or covered yet."
            ),
            warnings=warnings,
        )
        event_rows = _optional_rows(
            events_result,
            unavailable_warning=(
                "API-Football match events are unavailable for this match "
                "or subscription."
            ),
            empty_warning=(
                "API-Football returned no event timeline for this match."
            ),
            warnings=warnings,
        )

        players, lineup = map_lineups(lineup_rows, event)
        timeline, substitutions = map_timeline(event_rows, event)
        roster_quality = assess_roster_quality(
            players,
            lineup,
            event,
            substitutions,
        )
        if players and not roster_quality.automatic_identity_eligible:
            warnings.append(
                "The API-Football lineup is partial for automatic identity; "
                "available players can still be bound manually."
            )
        return EventBundle(
            source=self.id,
            event=event,
            players=players,
            lineup=lineup,
            timeline=timeline,
            substitutions=substitutions,
            roster_quality=roster_quality,
            fetched_at=datetime.now(UTC).isoformat(),
            warnings=warnings,
        )

    async def _team_id(self, name: str) -> str | None:
        data = await self._client.get(
            "teams",
            {"search": name.strip()},
            ttl=3600,
        )
        rows = response_rows(data)
        if not rows:
            return None
        expected = normalize_name(name)
        exact = next(
            (
                row
                for row in rows
                if normalize_name(
                    str((row.get("team") or {}).get("name") or "")
                )
                == expected
            ),
            rows[0],
        )
        team_id = (exact.get("team") or {}).get("id")
        return str(team_id) if team_id not in (None, "") else None

    async def _fixture_details(
        self,
        fixture: dict[str, Any],
        event_id: str,
    ) -> tuple[Any, Any]:
        # A fixture-id response can embed both blocks. Dedicated endpoints are
        # fallbacks only when a key is absent, not when the provider reports an
        # explicitly empty block.
        embedded_lineups = fixture.get("lineups") if "lineups" in fixture else None
        embedded_events = fixture.get("events") if "events" in fixture else None
        pending: list[tuple[str, Any]] = []
        if embedded_lineups is None:
            pending.append(
                (
                    "lineups",
                    self._client.get(
                        "fixtures/lineups",
                        {"fixture": event_id},
                        ttl=900,
                    ),
                )
            )
        if embedded_events is None:
            pending.append(
                (
                    "events",
                    self._client.get(
                        "fixtures/events",
                        {"fixture": event_id},
                        ttl=300,
                    ),
                )
            )
        fetched: dict[str, Any] = {}
        if pending:
            results = await asyncio.gather(
                *(request for _, request in pending),
                return_exceptions=True,
            )
            fetched = {
                name: result
                for (name, _), result in zip(pending, results, strict=True)
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
        return lineup_result, events_result


def _fixture_team_ids(item: dict[str, Any]) -> set[str]:
    teams = item.get("teams") or {}
    return {
        str((teams.get("home") or {}).get("id") or ""),
        str((teams.get("away") or {}).get("id") or ""),
    }


def _optional_rows(
    result: Any,
    *,
    unavailable_warning: str,
    empty_warning: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if isinstance(result, MatchDataError):
        warnings.append(unavailable_warning)
        return []
    if isinstance(result, Exception):
        raise result
    rows = response_rows(result)
    if not rows:
        warnings.append(empty_warning)
    return rows
