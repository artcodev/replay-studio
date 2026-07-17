from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import EventBundle, ExternalEvent


class MatchDataError(RuntimeError):
    """A sanitized, provider-neutral failure safe to return from the API.

    Provider implementations must never include credentials, request headers,
    or full upstream URLs in this message.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        code: str = "upstream-error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.retryable = retryable


class UnknownMatchDataProvider(MatchDataError):
    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Unknown match-data provider: {provider}",
            provider=provider,
            code="unknown-provider",
        )


class MatchDataProviderNotConfigured(MatchDataError):
    def __init__(self, provider: str, label: str) -> None:
        super().__init__(
            f"{label} is not configured on the API server",
            provider=provider,
            code="provider-not-configured",
        )


class MatchDataEventNotFound(MatchDataError):
    def __init__(self, provider: str, event_id: str) -> None:
        super().__init__(
            f"Match {event_id} was not found by {provider}",
            provider=provider,
            code="event-not-found",
        )


@runtime_checkable
class MatchDataProvider(Protocol):
    id: str
    name: str
    capabilities: tuple[str, ...]

    @property
    def configured(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    async def events_by_date(self, date: str) -> list[ExternalEvent]: ...

    async def search_events(self, query: str) -> list[ExternalEvent]: ...

    async def event_bundle(self, event_id: str) -> EventBundle: ...
