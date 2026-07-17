from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.providers.api_football import ApiFootballError, ApiFootballProvider


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        redis_url=None,
        api_football_api_key="server-secret",
    )


def _fixture(*, embedded: bool = True) -> dict:
    fixture = {
        "fixture": {
            "id": 12345,
            "date": "2026-07-17T19:00:00+00:00",
            "status": {"long": "Match Finished", "short": "FT"},
        },
        "league": {"name": "World Cup", "season": 2026},
        "teams": {
            "home": {"id": 1, "name": "Spain", "logo": "https://img/spain.png"},
            "away": {"id": 2, "name": "Belgium", "logo": "https://img/belgium.png"},
        },
        "goals": {"home": 2, "away": 1},
    }
    if embedded:
        fixture["lineups"] = _lineups()
        fixture["events"] = _events()
    return fixture


def _players(team: int, prefix: str) -> list[dict]:
    return [
        {
            "player": {
                "id": team * 100 + index,
                "name": f"{prefix} Player {index}",
                "number": index + 1,
                "pos": "G" if index == 0 else "M",
                "grid": "1:1" if index == 0 else f"2:{index}",
            }
        }
        for index in range(11)
    ]


def _lineups() -> list[dict]:
    return [
        {
            "team": {"id": 1, "name": "Spain"},
            "formation": "4-3-3",
            "startXI": _players(1, "Spain"),
            "substitutes": [
                {
                    "player": {
                        "id": 199,
                        "name": "Spain Substitute",
                        "number": 19,
                        "pos": "F",
                        "grid": None,
                    }
                }
            ],
        },
        {
            "team": {"id": 2, "name": "Belgium"},
            "formation": "3-4-2-1",
            "startXI": _players(2, "Belgium"),
            "substitutes": [
                {
                    "player": {
                        "id": 299,
                        "name": "Belgium Substitute",
                        "number": 19,
                        "pos": "F",
                        "grid": None,
                    }
                }
            ],
        },
    ]


def _events() -> list[dict]:
    return [
        {
            "time": {"elapsed": 45, "extra": 2},
            "team": {"id": 1, "name": "Spain"},
            "player": {"id": 110, "name": "Spain Player 10"},
            "assist": {"id": 199, "name": "Spain Substitute"},
            "type": "subst",
            "detail": "Tactical",
            "comments": None,
        }
    ]


def test_bundle_uses_embedded_fixture_data_and_normalizes_complete_lineup(
    monkeypatch,
) -> None:
    provider = ApiFootballProvider(_settings())
    calls: list[tuple[str, dict]] = []

    async def fake_get(endpoint: str, params: dict, ttl: int = 300) -> dict:
        del ttl
        calls.append((endpoint, params))
        assert endpoint == "fixtures"
        return {"response": [_fixture()]}

    monkeypatch.setattr(provider, "_get", fake_get)
    bundle = asyncio.run(provider.event_bundle("12345"))

    assert calls == [("fixtures", {"id": "12345"})]
    assert bundle.source == "api-football"
    assert bundle.event.provider == "api-football"
    assert bundle.event.home.name == "Spain"
    assert bundle.event.away_score == 1
    assert len(bundle.players) == 24
    assert len(bundle.lineup) == 24
    assert bundle.lineup[0].formation == "4-3-3"
    assert bundle.lineup[0].grid == "1:1"
    assert bundle.roster_quality is not None
    assert bundle.roster_quality.status == "automatic-ready"
    assert bundle.timeline[0].minute == 47
    assert bundle.timeline[0].label.startswith("45+2'")
    assert bundle.substitutions[0].player_out_id == "110"
    assert bundle.substitutions[0].player_in_id == "199"
    assert bundle.warnings == []


def test_timeline_ids_are_signature_based_and_stable_when_rows_reorder() -> None:
    provider = ApiFootballProvider(_settings())
    event = provider._map_fixture(_fixture())
    first = _events()[0]
    second = {
        "time": {"elapsed": 51, "extra": None},
        "team": {"id": 2, "name": "Belgium"},
        "player": {"id": 201, "name": "Belgium Player 1"},
        "assist": {"id": None, "name": None},
        "type": "Card",
        "detail": "Yellow Card",
    }

    original, _ = provider._map_timeline([first, second], event)
    reordered, _ = provider._map_timeline([second, first], event)

    original_by_player = {item.player_id: item.id for item in original}
    reordered_by_player = {item.player_id: item.id for item in reordered}
    assert reordered_by_player == original_by_player


def test_roster_quality_requires_a_starting_goalkeeper_for_each_team() -> None:
    provider = ApiFootballProvider(_settings())
    fixture = _fixture()
    for team_lineup in fixture["lineups"]:
        team_lineup["startXI"][0]["player"]["pos"] = "M"
    event = provider._map_fixture(fixture)
    players, lineup = provider._map_lineups(fixture["lineups"], event)
    timeline, substitutions = provider._map_timeline(fixture["events"], event)

    quality = provider._roster_quality(players, lineup, event, substitutions)

    assert timeline
    assert quality.status == "partial"
    assert quality.automatic_identity_eligible is False
    assert "starting-goalkeeper-missing" in quality.reasons


def test_bundle_retains_fixture_when_optional_lineup_request_fails(monkeypatch) -> None:
    provider = ApiFootballProvider(_settings())

    async def fake_get(endpoint: str, _params: dict, ttl: int = 300) -> dict:
        del ttl
        if endpoint == "fixtures":
            return {"response": [_fixture(embedded=False)]}
        if endpoint == "fixtures/lineups":
            raise ApiFootballError(
                "coverage unavailable", code="provider-auth-or-coverage"
            )
        assert endpoint == "fixtures/events"
        return {"response": _events()}

    monkeypatch.setattr(provider, "_get", fake_get)
    bundle = asyncio.run(provider.event_bundle("12345"))

    assert bundle.event.id == "12345"
    assert bundle.players == []
    assert bundle.timeline
    assert bundle.roster_quality is not None
    assert bundle.roster_quality.status == "unavailable"
    assert any("lineup is unavailable" in warning for warning in bundle.warnings)


def test_search_resolves_team_pair_then_uses_head_to_head(monkeypatch) -> None:
    provider = ApiFootballProvider(_settings())
    calls: list[tuple[str, dict]] = []

    async def fake_get(endpoint: str, params: dict, ttl: int = 300) -> dict:
        del ttl
        calls.append((endpoint, params))
        if endpoint == "teams":
            team_id = 1 if params["search"] == "Spain" else 2
            return {
                "response": [
                    {"team": {"id": team_id, "name": params["search"]}}
                ]
            }
        assert endpoint == "fixtures/headtohead"
        return {"response": [_fixture(embedded=False)]}

    monkeypatch.setattr(provider, "_get", fake_get)
    events = asyncio.run(provider.search_events("Spain vs Belgium"))

    assert events[0].id == "12345"
    assert events[0].provider == "api-football"
    assert ("fixtures/headtohead", {"h2h": "1-2"}) in calls
    assert all("last" not in params for endpoint, params in calls if endpoint == "fixtures/headtohead")


def test_search_requires_an_unambiguous_team_pair() -> None:
    provider = ApiFootballProvider(_settings())
    with pytest.raises(ApiFootballError) as error:
        asyncio.run(provider.search_events("Spain"))
    assert error.value.code == "team-pair-required"


def test_upstream_error_is_sanitized_while_key_stays_in_server_header(
    monkeypatch,
) -> None:
    provider = ApiFootballProvider(_settings())
    captured_headers: dict[str, str] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, *, params: dict, headers: dict):
            del params
            assert url.endswith("/fixtures")
            captured_headers.update(headers)
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                request=request,
                json={
                    "errors": {
                        "token": "invalid server-secret account credential"
                    },
                    "response": [],
                },
            )

    monkeypatch.setattr(
        "app.providers.api_football.httpx.AsyncClient",
        lambda **_kwargs: FakeClient(),
    )

    with pytest.raises(ApiFootballError) as error:
        asyncio.run(provider.events_by_date("2026-07-17"))

    assert captured_headers == {"x-apisports-key": "server-secret"}
    assert str(error.value) == "API-Football rejected the request"
    assert "server-secret" not in str(error.value)
