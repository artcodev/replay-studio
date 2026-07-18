from __future__ import annotations

import os
from typing import Any, Sequence

from .easyocr_provider import EasyOCRProvider
from .mmocr_provider import MMOCRProvider
from .provider_contract import JerseyOcrProvider, OcrResult, OcrSample, ProviderUnavailable


class UnavailableProvider:
    backend = "unavailable"
    inference_scope = "unavailable"

    def __init__(self, detail: str) -> None:
        self.detail = detail

    @property
    def loaded(self) -> bool:
        return False

    def load(self) -> None:
        raise ProviderUnavailable(self.detail)

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "providerVersion": None,
            "modelVersion": None,
            "device": None,
            "batchSize": None,
            "modelLoadSeconds": None,
            "inferenceScope": self.inference_scope,
        }

    def recognize(self, samples: Sequence[OcrSample]) -> list[OcrResult]:
        del samples
        raise ProviderUnavailable(self.detail)


def provider_from_environment() -> JerseyOcrProvider:
    name = os.environ.get("JERSEY_OCR_PROVIDER", "mmocr").strip().lower()
    try:
        if name == "mmocr":
            return MMOCRProvider()
        if name == "easyocr":
            return EasyOCRProvider()
    except (TypeError, ValueError) as exc:
        return UnavailableProvider(f"Invalid OCR provider configuration: {exc}")
    return UnavailableProvider(
        f"Unsupported JERSEY_OCR_PROVIDER={name!r}; expected 'mmocr' or 'easyocr'"
    )
