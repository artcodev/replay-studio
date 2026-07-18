from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Protocol

from ..config import Settings
from ..match_contracts import EventBundle, ExternalEvent, ExternalRosterQuality
from .base import MatchDataEventNotFound
from .thesportsdb_mapping import (
    assess_roster_quality,
    map_event,
    map_lineup,
    map_timeline,
    normalize_event_search_query,
)
from .thesportsdb_transport import TheSportsDbClient


class TheSportsDbGateway(Protocol):
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


class TheSportsDbProvider:
    """Coordinates TheSportsDB requests into the provider-neutral contract."""

    id = "thesportsdb"
    name = "TheSportsDB"
    capabilities = ("fixtures", "search", "lineups", "events")

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: TheSportsDbGateway | None = None,
    ) -> None:
        self._client = client or TheSportsDbClient(settings)

    @property
    def configured(self) -> bool:
        return self._client.configured

    @property
    def unavailable_reason(self) -> str | None:
        return self._client.unavailable_reason

    async def events_by_date(self, date: str) -> list[ExternalEvent]:
        data = await self._client.get(
            "eventsday.php",
            {"d": date, "s": "Soccer"},
            ttl=180,
        )
        return [map_event(item) for item in _rows(data, "events")]

    async def search_events(self, query: str) -> list[ExternalEvent]:
        data = await self._client.get(
            "searchevents.php",
            {"e": normalize_event_search_query(query)},
            ttl=600,
        )
        events = _rows(data, "event", "events")
        return [
            map_event(item)
            for item in events
            if str(item.get("strSport") or "Soccer").lower() == "soccer"
        ]

    async def event_bundle(self, event_id: str) -> EventBundle:
        event_data, lineup_data, timeline_data = await asyncio.gather(
            self._client.get("lookupevent.php", {"id": event_id}, ttl=600),
            self._client.get("lookuplineup.php", {"id": event_id}, ttl=900),
            self._client.get(
                "lookuptimeline.php",
                {"id": event_id},
                ttl=300,
            ),
        )
        events = _rows(event_data, "events")
        if not events:
            raise MatchDataEventNotFound(self.id, event_id)
        event = map_event(events[0])
        players, lineup = map_lineup(
            _rows(lineup_data, "lineup", "players"),
            event,
        )
        timeline, substitutions = map_timeline(
            _rows(timeline_data, "timeline", "events")
        )
        roster_quality = assess_roster_quality(players, event)
        return EventBundle(
            source=self.id,
            event=event,
            players=players,
            lineup=lineup,
            timeline=timeline,
            substitutions=substitutions,
            roster_quality=roster_quality,
            fetched_at=datetime.now(UTC).isoformat(),
            warnings=_bundle_warnings(
                player_count=len(players),
                timeline_count=len(timeline),
                roster_quality=roster_quality,
            ),
        )


def _rows(data: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _bundle_warnings(
    *,
    player_count: int,
    timeline_count: int,
    roster_quality: ExternalRosterQuality,
) -> list[str]:
    warnings: list[str] = []
    if player_count == 0:
        warnings.append(
            "The free source returned no lineup for this event; bindings can "
            "be entered manually."
        )
    elif player_count == 5:
        warnings.append(
            "The free source returned only five lineup players; automatic "
            "identity is disabled, but manual binding remains available."
        )
    elif not roster_quality.automatic_identity_eligible:
        warnings.append(
            "The lineup is incomplete for automatic identity; available "
            "players can still be bound manually."
        )
    if timeline_count == 0:
        warnings.append(
            "The free source returned no event timeline for this match."
        )
    elif timeline_count == 5:
        warnings.append(
            "The free source returned only the first five timeline events."
        )
    return warnings
