from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.config import Settings
from app.providers.base import MatchDataError, MatchDataProviderNotConfigured
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


def test_omitted_or_explicit_unconfigured_provider_fails_without_substitution() -> None:
    api = FakeProvider("api-football", configured=False, name="API-Football")
    alternative = FakeProvider("thesportsdb", configured=True, name="TheSportsDB")
    registry = MatchDataProviderRegistry(_settings(), [api, alternative])

    assert registry.preferred_provider == "api-football"
    assert registry.default_provider == "api-football"
    with pytest.raises(MatchDataProviderNotConfigured):
        asyncio.run(registry.event_bundle("same-raw-id"))
    with pytest.raises(MatchDataProviderNotConfigured):
        registry.get("api-football")
    assert asyncio.run(
        registry.event_bundle_for("thesportsdb", "same-raw-id")
    ) == ("thesportsdb", "same-raw-id")


def test_descriptors_are_safe_and_report_the_strict_configured_default() -> None:
    registry = MatchDataProviderRegistry(
        _settings(),
        [
            FakeProvider("api-football", configured=False, name="API-Football"),
            FakeProvider("thesportsdb", configured=True, name="TheSportsDB"),
        ],
    )

    descriptors = registry.descriptors()

    assert descriptors["defaultProvider"] == "api-football"
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


def test_implicit_date_lookup_falls_back_and_returns_actual_provider() -> None:
    class PlanLimitedProvider(FakeProvider):
        async def events_by_date(self, _date: str):
            raise MatchDataError(
                "plan does not cover date fixtures",
                provider=self.id,
                code="provider-request-rejected",
            )

    registry = MatchDataProviderRegistry(
        _settings(),
        [
            PlanLimitedProvider("api-football", configured=True),
            FakeProvider("thesportsdb", configured=True),
        ],
    )

    provider_id, events = asyncio.run(
        registry.events_by_date_with_fallback("2026-07-10")
    )

    assert provider_id == "thesportsdb"
    assert events == [("thesportsdb", "2026-07-10")]


def test_date_lookup_does_not_substitute_for_an_unconfigured_default() -> None:
    registry = MatchDataProviderRegistry(
        _settings(),
        [
            FakeProvider("api-football", configured=False, name="API-Football"),
            FakeProvider("thesportsdb", configured=True, name="TheSportsDB"),
        ],
    )

    with pytest.raises(MatchDataProviderNotConfigured) as error:
        asyncio.run(registry.events_by_date_with_fallback("2026-07-10"))

    assert error.value.provider == "api-football"
    assert error.value.code == "provider-not-configured"


def test_implicit_date_lookup_does_not_mask_non_provider_error() -> None:
    class InvalidRequestProvider(FakeProvider):
        async def events_by_date(self, _date: str):
            raise MatchDataError(
                "invalid date",
                provider=self.id,
                code="invalid-date",
            )

    registry = MatchDataProviderRegistry(
        _settings(),
        [
            InvalidRequestProvider("api-football", configured=True),
            FakeProvider("thesportsdb", configured=True),
        ],
    )

    with pytest.raises(MatchDataError) as error:
        asyncio.run(registry.events_by_date_with_fallback("not-a-date"))

    assert error.value.code == "invalid-date"
