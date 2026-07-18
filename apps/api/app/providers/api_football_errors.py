from __future__ import annotations

from .base import MatchDataError


class ApiFootballError(MatchDataError):
    """Sanitized API-Football failure shared by transport and orchestration."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream-error",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider="api-football",
            code=code,
            retryable=retryable,
        )
