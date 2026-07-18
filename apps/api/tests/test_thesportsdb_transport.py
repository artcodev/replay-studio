from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.providers.base import MatchDataProviderNotConfigured
from app.providers.thesportsdb_errors import SportsDbError
from app.providers.thesportsdb_transport import TheSportsDbClient


def _settings(*, api_key: str = "server-key") -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        redis_url=None,
        sportsdb_api_key=api_key,
    )


def test_transport_rejects_null_json_before_it_can_poison_cache(
    monkeypatch,
) -> None:
    client = TheSportsDbClient(_settings())
    requests = 0

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, *, params: dict):
            nonlocal requests
            requests += 1
            request = httpx.Request("GET", url, params=params)
            if requests == 1:
                return httpx.Response(200, request=request, content=b"null")
            return httpx.Response(
                200,
                request=request,
                json={"events": []},
            )

    monkeypatch.setattr(
        "app.providers.thesportsdb_transport.httpx.AsyncClient",
        lambda **_kwargs: FakeHttpClient(),
    )

    with pytest.raises(SportsDbError) as error:
        asyncio.run(client.get("eventsday.php", {"d": "2026-07-17"}))
    recovered = asyncio.run(
        client.get("eventsday.php", {"d": "2026-07-17"})
    )

    assert error.value.code == "invalid-provider-response"
    assert recovered == {"events": []}
    assert requests == 2


def test_unconfigured_client_fails_before_network_access() -> None:
    client = TheSportsDbClient(_settings(api_key=""))

    with pytest.raises(MatchDataProviderNotConfigured):
        asyncio.run(client.get("eventsday.php", {"d": "2026-07-17"}))
