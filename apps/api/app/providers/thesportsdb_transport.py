from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..config import Settings, get_settings
from .base import MatchDataProviderNotConfigured
from .thesportsdb_errors import SportsDbError


class TheSportsDbClient:
    """Authenticated TheSportsDB HTTP transport with bounded response caching."""

    provider_id = "thesportsdb"
    provider_name = "TheSportsDB"

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._api_key = settings.sportsdb_api_key.strip()
        self.base_url = (
            f"{settings.sportsdb_base_url.rstrip('/')}/{self._api_key}"
        )
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
        if self.configured:
            return None
        return "SPORTSDB_API_KEY is not configured"

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any],
        ttl: int = 300,
    ) -> dict[str, Any]:
        if not self.configured:
            raise MatchDataProviderNotConfigured(self.provider_id, self.provider_name)
        key = self._cache_key(endpoint, params)
        now = asyncio.get_running_loop().time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < ttl:
            return cached[1]

        remote_cached = await self._read_remote_cache(key)
        if remote_cached is not None:
            self._cache[key] = (now, remote_cached)
            return remote_cached

        data = await self._request(endpoint, params)
        self._cache[key] = (now, data)
        await self._write_remote_cache(key, data, ttl)
        return data

    @staticmethod
    def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
        encoded_params = json.dumps(
            params,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{endpoint}:{encoded_params}"

    def _remote_cache_key(self, key: str) -> str:
        return f"sportsdb:{self._credential_scope}:{key}"

    async def _read_remote_cache(self, key: str) -> dict[str, Any] | None:
        if self._redis is None:
            return None
        try:
            cached = await self._redis.get(self._remote_cache_key(key))
            if not cached:
                return None
            data = json.loads(cached)
            return data if isinstance(data, dict) else None
        except (RedisError, ValueError, TypeError):
            return None

    async def _write_remote_cache(
        self,
        key: str,
        data: dict[str, Any],
        ttl: int,
    ) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.setex(
                self._remote_cache_key(key),
                ttl,
                json.dumps(data),
            )
        except RedisError:
            return

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=12,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise SportsDbError(
                f"TheSportsDB returned HTTP {status}",
                code=(
                    "provider-rate-limit"
                    if status == 429
                    else "upstream-error"
                ),
                retryable=status == 429 or status >= 500,
            ) from exc
        except httpx.HTTPError as exc:
            raise SportsDbError(
                "TheSportsDB could not be reached",
                code="provider-unreachable",
                retryable=True,
            ) from exc
        except ValueError as exc:
            raise SportsDbError(
                "TheSportsDB returned an invalid response",
                code="invalid-provider-response",
            ) from exc
        if not isinstance(data, dict):
            raise SportsDbError(
                "TheSportsDB returned an invalid response",
                code="invalid-provider-response",
            )
        return data
