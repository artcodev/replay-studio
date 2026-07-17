from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from app.main import app, sports_provider
from app.schemas import ExternalEvent, ExternalTeam


@dataclass
class RouteProvider:
    id: str
    configured: bool = True
    name: str = "Route Provider"
    capabilities: tuple[str, ...] = ("fixtures", "search")

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.configured else "not configured"

    @staticmethod
    def _event(provider: str) -> ExternalEvent:
        return ExternalEvent(
            id="raw-42",
            provider=provider,
            name="Spain vs Belgium",
            home=ExternalTeam(id="1", name="Spain"),
            away=ExternalTeam(id="2", name="Belgium"),
        )

    async def events_by_date(self, _date: str):
        return [self._event(self.id)]

    async def search_events(self, _query: str):
        return [self._event(self.id)]

    async def event_bundle(self, _event_id: str):
        raise AssertionError("not used by this route test")


async def _async_get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def _get(path: str) -> httpx.Response:
    return asyncio.run(_async_get(path))


def test_catalog_dispatches_explicit_provider_and_returns_provider_identity(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sports_provider._providers,
        "api-football",
        RouteProvider("api-football"),
    )

    response = _get(
        "/api/catalog/events?date=2026-07-17&provider=api-football"
    )

    assert response.status_code == 200
    assert response.json()[0]["id"] == "raw-42"
    assert response.json()[0]["provider"] == "api-football"


def test_explicit_unconfigured_provider_fails_without_cross_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sports_provider._providers,
        "api-football",
        RouteProvider("api-football", configured=False),
    )

    response = _get(
        "/api/catalog/events?date=2026-07-17&provider=api-football"
    )

    assert response.status_code == 503
    assert response.headers["x-match-data-provider"] == "api-football"
    assert response.headers["x-match-data-error"] == "provider-not-configured"
    assert "not configured" in response.json()["detail"]


def test_unknown_provider_is_a_client_error() -> None:
    response = _get(
        "/api/catalog/events?date=2026-07-17&provider=unknown-provider"
    )

    assert response.status_code == 422
    assert response.headers["x-match-data-error"] == "unknown-provider"
