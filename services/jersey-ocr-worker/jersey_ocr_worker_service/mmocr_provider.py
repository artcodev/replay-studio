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


DEFAULT_DETECTOR = "dbnet_resnet18_fpnc_1200e_icdar2015"
DEFAULT_RECOGNIZER = "SAR"


class MMOCRProvider:
    """SoccerNet baseline: DBNet text detection followed by SAR recognition."""

    backend = "mmocr-dbnet18-sar"
    inference_scope = "crop"

    def __init__(self) -> None:
        self.device_name = os.environ.get("JERSEY_OCR_DEVICE", "cpu")
        self.batch_size = max(1, int(os.environ.get("JERSEY_OCR_MODEL_BATCH_SIZE", "32")))
        self.detector_name = os.environ.get("JERSEY_OCR_MMOCR_DETECTOR", DEFAULT_DETECTOR)
        self.recognizer_name = os.environ.get(
            "JERSEY_OCR_MMOCR_RECOGNIZER", DEFAULT_RECOGNIZER
        )
        self._loaded = False
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._detector = None
        self._recognizer = None
        self._bbox2poly = None
        self._poly2bbox = None
        self._crop_img = None
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
                from mmocr.apis import TextDetInferencer, TextRecInferencer
                from mmocr.utils import bbox2poly, crop_img, poly2bbox

                detector = TextDetInferencer(self.detector_name, device=self.device_name)
                recognizer = TextRecInferencer(self.recognizer_name, device=self.device_name)
            except Exception as exc:  # pragma: no cover - exercised in Docker
                raise ProviderUnavailable(f"MMOCR model failed to load: {exc}") from exc
            self._detector = detector
            self._recognizer = recognizer
            self._bbox2poly = bbox2poly
            self._poly2bbox = poly2bbox
            self._crop_img = crop_img
            self._load_seconds = perf_counter() - started
            self._loaded = True

    def info(self) -> dict[str, Any]:
        version = package_version("mmocr", "unavailable")
        return {
            "backend": self.backend,
            "providerVersion": version,
            "modelVersion": (
                f"mmocr-{package_version('mmocr', 'unknown')}/"
                f"{self.detector_name}/{self.recognizer_name}"
            ),
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
        if (
            self._detector is None
            or self._recognizer is None
            or self._bbox2poly is None
            or self._poly2bbox is None
            or self._crop_img is None
        ):
            raise ProviderUnavailable("MMOCR provider is not initialized")
        images = [np.ascontiguousarray(sample.image_rgb[:, :, ::-1]) for sample in samples]
        with self._inference_lock:
            try:
                detected = self._detector(
                    images,
                    return_datasamples=True,
                    batch_size=self.batch_size,
                    progress_bar=False,
                ).get("predictions", [])
                if len(detected) != len(samples):
                    raise ProviderUnavailable("MMOCR detector returned an incomplete batch")

                recognition_inputs: list[np.ndarray] = []
                recognition_sources: list[tuple[int, list[list[float]] | None]] = []
                for sample_index, (image, data_sample) in enumerate(zip(images, detected)):
                    for raw_polygon in data_sample.pred_instances.get("polygons", []):
                        quad = self._bbox2poly(self._poly2bbox(raw_polygon)).tolist()
                        text_crop = self._crop_img(image, quad)
                        if text_crop.size == 0:
                            continue
                        recognition_inputs.append(text_crop)
                        recognition_sources.append((sample_index, polygon(quad)))

                candidates: list[list[RawTextCandidate]] = [[] for _ in samples]
                if recognition_inputs:
                    recognized = self._recognizer(
                        recognition_inputs,
                        return_datasamples=True,
                        batch_size=self.batch_size,
                        progress_bar=False,
                    ).get("predictions", [])
                    if len(recognized) != len(recognition_inputs):
                        raise ProviderUnavailable("MMOCR recognizer returned an incomplete batch")
                    for (sample_index, candidate_polygon), prediction in zip(
                        recognition_sources, recognized
                    ):
                        value = self._recognizer.pred2dict(prediction)
                        candidates[sample_index].append(
                            RawTextCandidate(
                                text=str(value.get("text") or ""),
                                confidence=confidence(value.get("scores")),
                                polygon=candidate_polygon,
                            )
                        )
            except ProviderUnavailable:
                raise
            except Exception as exc:  # pragma: no cover - model-runtime dependent
                raise ProviderUnavailable(f"MMOCR inference failed: {exc}") from exc
        return [
            OcrResult(sample.crop_id, tuple(items))
            for sample, items in zip(samples, candidates)
        ]
