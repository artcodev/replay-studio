from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.providers.api_football_errors import ApiFootballError
from app.providers.api_football_transport import ApiFootballClient
from app.providers.base import MatchDataProviderNotConfigured


def _settings(*, api_key: str | None = "server-secret") -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        redis_url=None,
        api_football_api_key=api_key,
    )


def test_upstream_error_is_sanitized_while_key_stays_in_server_header(
    monkeypatch,
) -> None:
    client = ApiFootballClient(_settings())
    captured_headers: dict[str, str] = {}

    class FakeHttpClient:
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
        "app.providers.api_football_transport.httpx.AsyncClient",
        lambda **_kwargs: FakeHttpClient(),
    )

    with pytest.raises(ApiFootballError) as error:
        asyncio.run(client.get("fixtures", {"date": "2026-07-17"}))

    assert captured_headers == {"x-apisports-key": "server-secret"}
    assert str(error.value) == "API-Football rejected the request"
    assert "server-secret" not in str(error.value)


def test_memory_cache_avoids_repeating_identical_request(monkeypatch) -> None:
    client = ApiFootballClient(_settings())
    requests = 0

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str, *, params: dict, headers: dict):
            nonlocal requests
            del params, headers
            requests += 1
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                request=request,
                json={"response": [{"fixture": {"id": 42}}]},
            )

    monkeypatch.setattr(
        "app.providers.api_football_transport.httpx.AsyncClient",
        lambda **_kwargs: FakeHttpClient(),
    )

    first = asyncio.run(client.get("fixtures", {"id": "42"}))
    second = asyncio.run(client.get("fixtures", {"id": "42"}))

    assert second == first
    assert requests == 1


def test_unconfigured_client_fails_before_network_access() -> None:
    client = ApiFootballClient(_settings(api_key=None))

    with pytest.raises(MatchDataProviderNotConfigured):
        asyncio.run(client.get("fixtures", {"date": "2026-07-17"}))
