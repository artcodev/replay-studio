from __future__ import annotations

import os
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .provider_contract import (
    OcrResult,
    OcrSample,
    ProviderUnavailable,
    RawTextCandidate,
)
from .provider_values import confidence, package_version, polygon


class EasyOCRProvider:
    """Lighter SoccerNet alternative using EasyOCR's batched reader."""

    backend = "easyocr-english-digits"
    inference_scope = "crop"

    def __init__(self) -> None:
        self.device_name = os.environ.get("JERSEY_OCR_DEVICE", "cpu")
        self.batch_size = max(1, int(os.environ.get("JERSEY_OCR_MODEL_BATCH_SIZE", "64")))
        self.model_directory = os.environ.get("EASYOCR_MODULE_PATH")
        self._loaded = False
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._reader = None
        self._load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            started = perf_counter()
            try:
                import easyocr

                kwargs: dict[str, Any] = {
                    "gpu": self.device_name not in {"cpu", "mps"},
                    "download_enabled": os.environ.get(
                        "JERSEY_OCR_ALLOW_RUNTIME_DOWNLOADS", "0"
                    )
                    in {"1", "true", "True"},
                }
                if self.model_directory:
                    kwargs["model_storage_directory"] = self.model_directory
                reader = easyocr.Reader(["en"], **kwargs)
            except Exception as exc:  # pragma: no cover - exercised in Docker
                raise ProviderUnavailable(f"EasyOCR model failed to load: {exc}") from exc
            self._reader = reader
            self._load_seconds = perf_counter() - started
            self._loaded = True

    def info(self) -> dict[str, Any]:
        version = package_version("easyocr", "unavailable")
        return {
            "backend": self.backend,
            "providerVersion": version,
            "modelVersion": f"easyocr-{package_version('easyocr', 'unknown')}/english-v1",
            "device": self.device_name,
            "batchSize": self.batch_size,
            "modelLoadSeconds": (
                round(self._load_seconds, 4) if self._load_seconds is not None else None
            ),
            "inferenceScope": self.inference_scope,
        }

    def recognize(self, samples: Sequence[OcrSample]) -> list[OcrResult]:
        if not samples:
            return []
        if not self._loaded:
            self.load()
        if self._reader is None:
            raise ProviderUnavailable("EasyOCR provider is not initialized")
        images = [np.ascontiguousarray(sample.image_rgb[:, :, ::-1]) for sample in samples]
        with self._inference_lock:
            try:
                raw_results = self._reader.readtext_batched(
                    images,
                    n_width=64,
                    n_height=128,
                    batch_size=self.batch_size,
                    workers=0,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",
                    text_threshold=0.5,
                    link_threshold=0.3,
                )
            except Exception as exc:  # pragma: no cover - model-runtime dependent
                raise ProviderUnavailable(f"EasyOCR inference failed: {exc}") from exc
        if len(raw_results) != len(samples):
            raise ProviderUnavailable("EasyOCR returned an incomplete batch")
        results: list[OcrResult] = []
        for sample, raw_items in zip(samples, raw_results):
            items = tuple(
                RawTextCandidate(
                    text=str(raw_item[1]),
                    confidence=confidence(raw_item[2]),
                    polygon=polygon(raw_item[0]),
                )
                for raw_item in raw_items
                if isinstance(raw_item, (list, tuple)) and len(raw_item) >= 3
            )
            results.append(OcrResult(sample.crop_id, items))
        return results
