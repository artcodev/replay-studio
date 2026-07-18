from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from app.providers.thesportsdb_provider import TheSportsDbProvider


class FakeTheSportsDbClient:
    configured = True
    unavailable_reason = None

    def __init__(
        self,
        get: Callable[[str, dict[str, Any], int], Awaitable[dict[str, Any]]],
    ) -> None:
        self._get = get

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any],
        ttl: int = 300,
    ) -> dict[str, Any]:
        return await self._get(endpoint, params, ttl)


def _event_payload() -> dict:
    return {
        "events": [
            {
                "idEvent": "event-1",
                "strEvent": "Home vs Away",
                "idHomeTeam": "team-home",
                "strHomeTeam": "Home",
                "idAwayTeam": "team-away",
                "strAwayTeam": "Away",
                "strSport": "Soccer",
            }
        ]
    }


def _lineup_player(index: int) -> dict:
    home = index < 11
    team_index = index if home else index - 11
    return {
        "idLineup": f"lineup-{index}",
        "idPlayer": f"player-{index}",
        "strPlayer": f"Player {index}",
        "strHome": "Yes" if home else "No",
        "strSubstitute": "No" if team_index < 8 else "Yes",
        "strPosition": "Midfielder",
        "intSquadNumber": index + 1,
    }


def test_search_events_accepts_singular_event_root() -> None:
    async def fake_get(endpoint: str, params: dict, ttl: int = 300) -> dict:
        del ttl
        assert endpoint == "searchevents.php"
        assert params == {"e": "Spain_vs_Belgium"}
        return {
            "event": [
                {
                    "idEvent": "2519345",
                    "strEvent": "Spain vs Belgium",
                    "strSport": "Soccer",
                    "idHomeTeam": "133909",
                    "strHomeTeam": "Spain",
                    "idAwayTeam": "134515",
                    "strAwayTeam": "Belgium",
                }
            ]
        }

    provider = TheSportsDbProvider(client=FakeTheSportsDbClient(fake_get))

    events = asyncio.run(provider.search_events("Spain vs Belgium"))

    assert [event.id for event in events] == ["2519345"]


def test_event_bundle_preserves_lineup_timeline_and_substitution() -> None:
    async def fake_get(endpoint: str, _params: dict, ttl: int = 300) -> dict:
        del ttl
        if endpoint == "lookupevent.php":
            return _event_payload()
        if endpoint == "lookuplineup.php":
            return {"lineup": [_lineup_player(index) for index in range(22)]}
        assert endpoint == "lookuptimeline.php"
        return {
            "timeline": [
                {
                    "idTimeline": "timeline-61",
                    "intTime": "61",
                    "strTimeline": "Substitution",
                    "strTimelineDetail": "Tactical change",
                    "idTeam": "team-home",
                    "strTeam": "Home",
                    "idPlayer": "player-7",
                    "strPlayer": "Player 7",
                    "idPlayer2": "player-8",
                    "strPlayer2": "Player 8",
                }
            ]
        }

    provider = TheSportsDbProvider(client=FakeTheSportsDbClient(fake_get))

    bundle = asyncio.run(provider.event_bundle("event-1"))

    assert len(bundle.players) == 22
    assert len(bundle.lineup) == 22
    assert bundle.players[10].team_id == "team-home"
    assert bundle.players[11].team_id == "team-away"
    assert bundle.lineup[8].role == "substitute"
    assert bundle.timeline[0].secondary_player_id == "player-8"
    assert bundle.timeline[0].detail == "Tactical change"
    assert bundle.substitutions[0].player_out_id == "player-7"
    assert bundle.substitutions[0].player_in_id == "player-8"
    assert bundle.roster_quality is not None
    assert bundle.roster_quality.status == "automatic-ready"
    assert bundle.roster_quality.automatic_identity_eligible is True
    assert bundle.warnings == []


def test_event_bundle_flags_the_free_five_player_cap_as_manual_only() -> None:
    async def fake_get(endpoint: str, _params: dict, ttl: int = 300) -> dict:
        del ttl
        if endpoint == "lookupevent.php":
            return _event_payload()
        if endpoint == "lookuplineup.php":
            return {"lineup": [_lineup_player(index) for index in range(5)]}
        return {"timeline": []}

    provider = TheSportsDbProvider(client=FakeTheSportsDbClient(fake_get))

    bundle = asyncio.run(provider.event_bundle("event-1"))

    assert bundle.roster_quality is not None
    assert bundle.roster_quality.status == "partial"
    assert bundle.roster_quality.automatic_identity_eligible is False
    assert bundle.roster_quality.manual_identity_eligible is True
    assert "provider-five-player-cap" in bundle.roster_quality.reasons
    assert "automatic identity is disabled" in bundle.warnings[0]
