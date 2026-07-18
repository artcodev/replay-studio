from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from ..match_contracts import EventBundle, ExternalEvent
from .api_football_provider import ApiFootballProvider
from .base import (
    MatchDataError,
    MatchDataProvider,
    MatchDataProviderNotConfigured,
    UnknownMatchDataProvider,
)
from .thesportsdb_provider import TheSportsDbProvider


class MatchDataProviderRegistry:
    """Strict registry for all server-side match-data providers."""

    _implicit_fallback_error_codes = frozenset(
        {
            "invalid-provider-response",
            "provider-auth-or-coverage",
            "provider-rate-limit",
            "provider-request-rejected",
            "provider-unreachable",
            "upstream-error",
        }
    )

    def __init__(
        self,
        settings: Settings | None = None,
        providers: list[MatchDataProvider] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        provider_list = providers or [
            ApiFootballProvider(self.settings),
            TheSportsDbProvider(self.settings),
        ]
        self._providers = {provider.id: provider for provider in provider_list}
        if self.settings.match_data_provider not in self._providers:
            raise UnknownMatchDataProvider(self.settings.match_data_provider)

    @property
    def default_provider(self) -> str:
        return self.settings.match_data_provider

    @property
    def preferred_provider(self) -> str:
        return self.settings.match_data_provider

    def get(self, provider_id: str | None = None) -> MatchDataProvider:
        selected = provider_id or self.default_provider
        provider = self._providers.get(selected)
        if provider is None:
            raise UnknownMatchDataProvider(selected)
        if not provider.configured:
            raise MatchDataProviderNotConfigured(provider.id, provider.name)
        return provider

    def has(self, provider_id: str) -> bool:
        return provider_id in self._providers

    def descriptors(self) -> dict[str, Any]:
        return {
            "providers": [
                {
                    "id": provider.id,
                    "name": provider.name,
                    "configured": provider.configured,
                    "available": provider.configured,
                    "reason": provider.unavailable_reason,
                    "capabilities": list(provider.capabilities),
                }
                for provider in self._providers.values()
            ],
            "defaultProvider": self.default_provider,
            "preferredProvider": self.preferred_provider,
        }

    async def events_by_date(self, date: str) -> list[ExternalEvent]:
        return await self.get().events_by_date(date)

    async def events_by_date_with_fallback(
        self,
        date: str,
    ) -> tuple[str, list[ExternalEvent]]:
        """Resolve an omitted provider while retaining honest provenance.

        Fixture-by-date coverage is plan-dependent. A configured preferred
        provider can therefore reject this single capability while its team
        search and fixture-detail APIs remain healthy. Provider-neutral callers
        may try another configured adapter, but explicit provider routes must
        continue to use :meth:`events_by_date_for` and surface its error.
        """

        default_id = self.default_provider
        # Provider omission means "use the configured default", never "pick
        # any provider that happens to have credentials". The bounded date
        # fallback below is available only after that explicit default has
        # passed configuration validation and then rejects/fails the request.
        primary = self.get(default_id)
        provider_ids = [
            provider_id
            for provider_id in (primary.id, *self._providers)
            if self._providers[provider_id].configured
        ]
        provider_ids = list(dict.fromkeys(provider_ids))

        for index, provider_id in enumerate(provider_ids):
            provider = self.get(provider_id)
            try:
                return provider.id, await provider.events_by_date(date)
            except MatchDataError as exc:
                can_fallback = (
                    exc.code in self._implicit_fallback_error_codes
                    and index + 1 < len(provider_ids)
                )
                if not can_fallback:
                    raise
        raise AssertionError("unreachable")

    async def search_events(self, query: str) -> list[ExternalEvent]:
        return await self.get().search_events(query)

    async def event_bundle(self, event_id: str) -> EventBundle:
        return await self.get().event_bundle(event_id)

    async def events_by_date_for(
        self, provider_id: str | None, date: str
    ) -> list[ExternalEvent]:
        return await self.get(provider_id).events_by_date(date)

    async def search_events_for(
        self, provider_id: str | None, query: str
    ) -> list[ExternalEvent]:
        return await self.get(provider_id).search_events(query)

    async def event_bundle_for(
        self, provider_id: str | None, event_id: str
    ) -> EventBundle:
        return await self.get(provider_id).event_bundle(event_id)


sports_provider = MatchDataProviderRegistry()
