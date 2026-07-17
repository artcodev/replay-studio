from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from ..config import Settings, get_settings
from ..schemas import EventBundle, ExternalEvent
from .api_football import ApiFootballProvider
from .base import (
    MatchDataProvider,
    MatchDataProviderNotConfigured,
    UnknownMatchDataProvider,
)
from .thesportsdb import TheSportsDbProvider


class MatchDataProviderRegistry:
    """Registry and compatibility facade for all server-side providers."""

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
        self._override: ContextVar[str | None] = ContextVar(
            "match_data_provider_override", default=None
        )
        if self.settings.match_data_provider not in self._providers:
            raise UnknownMatchDataProvider(self.settings.match_data_provider)

    @property
    def default_provider(self) -> str:
        preferred = self._providers[self.settings.match_data_provider]
        if preferred.configured:
            return preferred.id
        # Omitted provider is the legacy contract. Resolve its default before
        # issuing any request so old installations keep working while an
        # explicit provider (including a saved event source) never falls back.
        for provider in self._providers.values():
            if provider.configured:
                return provider.id
        return preferred.id

    @property
    def preferred_provider(self) -> str:
        return self.settings.match_data_provider

    def get(self, provider_id: str | None = None) -> MatchDataProvider:
        selected = provider_id or self._override.get() or self.default_provider
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

    async def search_events(self, query: str) -> list[ExternalEvent]:
        return await self.get().search_events(query)

    async def event_bundle(self, event_id: str) -> EventBundle:
        return await self.get().event_bundle(event_id)

    async def events_by_date_for(
        self, provider_id: str | None, date: str
    ) -> list[ExternalEvent]:
        token = self._override.set(provider_id)
        try:
            return await self.events_by_date(date)
        finally:
            self._override.reset(token)

    async def search_events_for(
        self, provider_id: str | None, query: str
    ) -> list[ExternalEvent]:
        token = self._override.set(provider_id)
        try:
            return await self.search_events(query)
        finally:
            self._override.reset(token)

    async def event_bundle_for(
        self, provider_id: str | None, event_id: str
    ) -> EventBundle:
        # The indirection through event_bundle deliberately preserves the
        # long-standing ``app.main.sports_provider.event_bundle`` test/plugin
        # seam while selecting a provider safely via a task-local ContextVar.
        token = self._override.set(provider_id)
        try:
            return await self.event_bundle(event_id)
        finally:
            self._override.reset(token)


sports_provider = MatchDataProviderRegistry()
