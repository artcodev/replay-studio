from __future__ import annotations

"""Select exactly one person detector provider for a reconstruction run."""

from .config import get_settings
from .local_person_detection_provider import (
    LocalUltralyticsPersonDetectionProvider,
)
from .person_detection_provider_contract import PersonDetectionProvider
from .remote_person_detection_provider import RemotePersonDetectionProvider


def build_person_detection_provider(
    model_name: str,
    model: object,
) -> PersonDetectionProvider:
    settings = get_settings()
    if settings.person_detection_worker_url:
        return RemotePersonDetectionProvider(
            settings.person_detection_worker_url,
            timeout=float(settings.person_detection_worker_timeout),
            expected_checkpoint=model_name,
        )
    return LocalUltralyticsPersonDetectionProvider(model_name, model)


__all__ = ("build_person_detection_provider",)
