from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.config import Settings
from app.providers.base import MatchDataProviderNotConfigured
from app.providers.registry import MatchDataProviderRegistry


@dataclass
class FakeProvider:
    id: str
    configured: bool
    name: str = "Fake"
    capabilities: tuple[str, ...] = ("fixtures",)

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "missing server key"

    async def events_by_date(self, date: str):
        return [(self.id, date)]

    async def search_events(self, query: str):
        return [(self.id, query)]

    async def event_bundle(self, event_id: str):
        return (self.id, event_id)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        match_data_provider="api-football",
        api_football_api_key=None,
        sportsdb_api_key="123",
        redis_url=None,
    )


def test_omitted_provider_resolves_before_lookup_but_explicit_never_falls_back() -> None:
    api = FakeProvider("api-football", configured=False, name="API-Football")
    legacy = FakeProvider("thesportsdb", configured=True, name="TheSportsDB")
    registry = MatchDataProviderRegistry(_settings(), [api, legacy])

    assert registry.preferred_provider == "api-football"
    assert registry.default_provider == "thesportsdb"
    assert asyncio.run(registry.event_bundle("same-raw-id")) == (
        "thesportsdb",
        "same-raw-id",
    )
    with pytest.raises(MatchDataProviderNotConfigured):
        registry.get("api-football")


def test_descriptors_are_safe_and_report_resolved_and_preferred_defaults() -> None:
    registry = MatchDataProviderRegistry(
        _settings(),
        [
            FakeProvider("api-football", configured=False, name="API-Football"),
            FakeProvider("thesportsdb", configured=True, name="TheSportsDB"),
        ],
    )

    descriptors = registry.descriptors()

    assert descriptors["defaultProvider"] == "thesportsdb"
    assert descriptors["preferredProvider"] == "api-football"
    assert descriptors["providers"][0] == {
        "id": "api-football",
        "name": "API-Football",
        "configured": False,
        "available": False,
        "reason": "missing server key",
        "capabilities": ["fixtures"],
    }
    assert "server-secret" not in str(descriptors)


def test_explicit_same_raw_event_id_is_dispatched_to_its_selected_provider() -> None:
    registry = MatchDataProviderRegistry(
        _settings(),
        [
            FakeProvider("api-football", configured=True, name="API-Football"),
            FakeProvider("thesportsdb", configured=True, name="TheSportsDB"),
        ],
    )

    api_result = asyncio.run(
        registry.event_bundle_for("api-football", "collision-42")
    )
    sportsdb_result = asyncio.run(
        registry.event_bundle_for("thesportsdb", "collision-42")
    )

    assert api_result == ("api-football", "collision-42")
    assert sportsdb_result == ("thesportsdb", "collision-42")
