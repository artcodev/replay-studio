from __future__ import annotations

from .base import MatchDataError


class SportsDbError(MatchDataError):
    """Sanitized TheSportsDB failure shared by transport and orchestration."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream-error",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider="thesportsdb",
            code=code,
            retryable=retryable,
        )
