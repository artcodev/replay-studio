from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..config import Settings, get_settings
from .api_football_errors import ApiFootballError
from .base import MatchDataProviderNotConfigured


class ApiFootballClient:
    """Authenticated API-Football HTTP transport with bounded response caching."""

    provider_id = "api-football"
    provider_name = "API-Football"

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.base_url = settings.api_football_base_url.rstrip("/")
        self._api_key = (settings.api_football_api_key or "").strip()
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
        return "API_FOOTBALL_API_KEY is not configured"

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
        return (
            f"match-data:{self.provider_id}:{self._credential_scope}:{key}"
        )

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
                timeout=15,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint.lstrip('/')}",
                    params=params,
                    headers={"x-apisports-key": self._api_key},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._status_error(exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise ApiFootballError(
                "API-Football could not be reached",
                code="provider-unreachable",
                retryable=True,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise ApiFootballError(
                "API-Football returned an invalid response",
                code="invalid-provider-response",
            ) from exc

        if not isinstance(data, dict):
            raise ApiFootballError(
                "API-Football returned an invalid response",
                code="invalid-provider-response",
            )
        upstream_errors = self._upstream_errors(data)
        if upstream_errors:
            # Provider error bodies may echo account-specific data. Preserve
            # only stable classification and retryability in the public error.
            raise ApiFootballError(
                "API-Football rejected the request",
                code="provider-request-rejected",
                retryable=any(
                    "limit" in error.lower() for error in upstream_errors
                ),
            )
        return data

    @staticmethod
    def _status_error(status: int) -> ApiFootballError:
        if status in {401, 403}:
            return ApiFootballError(
                "API-Football rejected the server credential or plan coverage",
                code="provider-auth-or-coverage",
            )
        if status == 429:
            return ApiFootballError(
                "API-Football request quota was exceeded",
                code="provider-rate-limit",
                retryable=True,
            )
        return ApiFootballError(
            f"API-Football returned HTTP {status}",
            retryable=status >= 500,
        )

    @staticmethod
    def _upstream_errors(data: dict[str, Any]) -> list[str]:
        errors = data.get("errors")
        if isinstance(errors, dict):
            return [str(value) for value in errors.values() if value]
        if isinstance(errors, list):
            return [str(value) for value in errors if value]
        if errors:
            return [str(errors)]
        return []
