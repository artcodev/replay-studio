from __future__ import annotations

from .provider_contract import BallDetectionProvider
from .wasb_configuration import WasbConfiguration
from .wasb_provider import WasbSoccerProvider


def provider_from_environment() -> BallDetectionProvider:
    """Build the configured ball provider at the HTTP composition boundary."""

    return WasbSoccerProvider(WasbConfiguration.from_environment())
