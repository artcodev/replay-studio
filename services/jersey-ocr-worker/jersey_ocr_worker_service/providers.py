from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from threading import Lock
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np


MMOCR_DETECTOR = "dbnet_resnet18_fpnc_1200e_icdar2015"
MMOCR_RECOGNIZER = "SAR"


class ProviderUnavailable(RuntimeError):
    """The configured OCR runtime or its model assets are unavailable."""


@dataclass(frozen=True, slots=True)
class OcrSample:
    crop_id: str
    image_rgb: np.ndarray
    tracklet_id: str | None = None
    observation_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class RawTextCandidate:
    text: str
    confidence: float
    polygon: list[list[float]] | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    crop_id: str
    candidates: tuple[RawTextCandidate, ...]


class JerseyOcrProvider(Protocol):
    backend: str

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> dict[str, Any]: ...

    def recognize(self, samples: Sequence[OcrSample]) -> list[OcrResult]: ...


def _package_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _confidence(value: Any) -> float:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    array = array[np.isfinite(array)]
    if not array.size:
        return 0.0
    return float(np.clip(array.mean(), 0.0, 1.0))


def _polygon(value: Any) -> list[list[float]] | None:
    try:
        points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.shape[0] < 2 or not np.isfinite(points).all():
        return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


class MMOCRProvider:
    """SoccerNet baseline: DBNet text detection followed by SAR recognition."""

    backend = "mmocr-dbnet18-sar"
    inference_scope = "crop"

    def __init__(self) -> None:
        self.device_name = os.environ.get("JERSEY_OCR_DEVICE", "cpu")
        self.batch_size = max(1, int(os.environ.get("JERSEY_OCR_MODEL_BATCH_SIZE", "32")))
        self.detector_name = os.environ.get("JERSEY_OCR_MMOCR_DETECTOR", MMOCR_DETECTOR)
        self.recognizer_name = os.environ.get("JERSEY_OCR_MMOCR_RECOGNIZER", MMOCR_RECOGNIZER)
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
            except Exception as exc:  # pragma: no cover - exercised in the Docker image
                raise ProviderUnavailable(f"MMOCR model failed to load: {exc}") from exc
            self._detector = detector
            self._recognizer = recognizer
            self._bbox2poly = bbox2poly
            self._poly2bbox = poly2bbox
            self._crop_img = crop_img
            self._load_seconds = perf_counter() - started
            self._loaded = True

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "providerVersion": _package_version("mmocr", "unavailable"),
            "modelVersion": (
                f"mmocr-{_package_version('mmocr', 'unknown')}/"
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
        # TrackLab/SoccerNet feeds crops loaded by cv2 into both OCR baselines.
        # The HTTP boundary is explicitly RGB, so restore the reference BGR
        # convention before calling MMOCR.
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
                    instances = data_sample.pred_instances
                    for polygon in instances.get("polygons", []):
                        quad = self._bbox2poly(self._poly2bbox(polygon)).tolist()
                        text_crop = self._crop_img(image, quad)
                        if text_crop.size == 0:
                            continue
                        recognition_inputs.append(text_crop)
                        recognition_sources.append((sample_index, _polygon(quad)))

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
                    for (sample_index, polygon), prediction in zip(
                        recognition_sources, recognized
                    ):
                        value = self._recognizer.pred2dict(prediction)
                        text = str(value.get("text") or "")
                        candidates[sample_index].append(
                            RawTextCandidate(
                                text=text,
                                confidence=_confidence(value.get("scores")),
                                polygon=polygon,
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
            except Exception as exc:  # pragma: no cover - exercised in the Docker image
                raise ProviderUnavailable(f"EasyOCR model failed to load: {exc}") from exc
            self._reader = reader
            self._load_seconds = perf_counter() - started
            self._loaded = True

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "providerVersion": _package_version("easyocr", "unavailable"),
            "modelVersion": f"easyocr-{_package_version('easyocr', 'unknown')}/english-v1",
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
        # EasyOCR's SoccerNet adapter also receives cv2/BGR crops.
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
            items: list[RawTextCandidate] = []
            for raw_item in raw_items:
                if not isinstance(raw_item, (list, tuple)) or len(raw_item) < 3:
                    continue
                items.append(
                    RawTextCandidate(
                        text=str(raw_item[1]),
                        confidence=_confidence(raw_item[2]),
                        polygon=_polygon(raw_item[0]),
                    )
                )
            results.append(OcrResult(sample.crop_id, tuple(items)))
        return results


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
