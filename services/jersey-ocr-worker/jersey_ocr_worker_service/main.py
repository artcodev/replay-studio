from __future__ import annotations

from contextlib import asynccontextmanager
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
import re
from typing import Any, Sequence
import unicodedata
from threading import Lock
from time import monotonic

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .providers import (
    JerseyOcrProvider,
    OcrResult,
    OcrSample,
    ProviderUnavailable,
    RawTextCandidate,
    provider_from_environment,
)


SERVICE_NAME = "replay-studio-jersey-ocr-worker"
CONTRACT_VERSION = "jersey-ocr.v1"
MAX_JERSEY_DIGITS = 2
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"


def _capabilities() -> dict[str, Any]:
    return {
        "digitsOnly": True,
        "maxDigits": MAX_JERSEY_DIGITS,
        "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
        # Inputs are independently addressable crops grouped by tracklet. A
        # future PARSeq provider may consume the whole group jointly while
        # preserving this response contract.
        "inputScopes": ["crop", "tracklet"],
    }


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    minimum_width: int = 20
    minimum_height: int = 36
    minimum_sharpness: float = 5.0
    minimum_contrast: float = 4.0
    minimum_confidence: float = 0.25
    ambiguity_margin: float = 0.05
    max_crop_bytes: int = 8_000_000
    max_crop_pixels: int = 4_000_000
    max_batch_size: int = 128

    @classmethod
    def from_environment(cls) -> "QualityPolicy":
        return cls(
            minimum_width=max(1, int(os.environ.get("JERSEY_OCR_MIN_CROP_WIDTH", "20"))),
            minimum_height=max(1, int(os.environ.get("JERSEY_OCR_MIN_CROP_HEIGHT", "36"))),
            minimum_sharpness=max(
                0.0, float(os.environ.get("JERSEY_OCR_MIN_SHARPNESS", "5"))
            ),
            minimum_contrast=max(
                0.0, float(os.environ.get("JERSEY_OCR_MIN_CONTRAST", "4"))
            ),
            minimum_confidence=float(
                os.environ.get("JERSEY_OCR_MIN_CONFIDENCE", "0.25")
            ),
            ambiguity_margin=max(
                0.0, float(os.environ.get("JERSEY_OCR_AMBIGUITY_MARGIN", "0.05"))
            ),
            max_crop_bytes=max(
                1024, int(os.environ.get("JERSEY_OCR_MAX_CROP_BYTES", "8000000"))
            ),
            max_crop_pixels=max(
                1, int(os.environ.get("JERSEY_OCR_MAX_CROP_PIXELS", "4000000"))
            ),
            max_batch_size=max(
                1, int(os.environ.get("JERSEY_OCR_MAX_BATCH_SIZE", "128"))
            ),
        )

    def validate(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ProviderUnavailable("JERSEY_OCR_MIN_CONFIDENCE must be between 0 and 1")
        if self.ambiguity_margin > 1.0:
            raise ProviderUnavailable("JERSEY_OCR_AMBIGUITY_MARGIN must not exceed 1")


class OcrResultCache:
    """Bounded in-memory cache for deterministic crop-level providers."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._items: OrderedDict[
            str, tuple[float, tuple[RawTextCandidate, ...]]
        ] = OrderedDict()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: str) -> tuple[RawTextCandidate, ...] | None:
        if not self.enabled:
            return None
        now = monotonic()
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            created_at, candidates = value
            if now - created_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return candidates

    def put(self, key: str, candidates: Sequence[RawTextCandidate]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items[key] = (monotonic(), tuple(candidates))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


def _cache_key(image: np.ndarray, info: dict[str, Any], policy: QualityPolicy) -> str:
    digest = sha256()
    digest.update(str(info["modelVersion"]).encode("utf-8"))
    digest.update(
        (
            f"|{policy.minimum_width}|{policy.minimum_height}|"
            f"{policy.minimum_sharpness}|{policy.minimum_contrast}|"
            f"{policy.minimum_confidence}|{policy.ambiguity_margin}|"
        ).encode("ascii")
    )
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _evidence_fingerprint(image: np.ndarray) -> str:
    """Fingerprint decoded pixels independently from crop/request identity."""

    digest = sha256()
    digest.update(EVIDENCE_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(image.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes())
    return f"{EVIDENCE_FINGERPRINT_VERSION}:{digest.hexdigest()}"


def _decode_image(data: bytes, file_index: int, policy: QualityPolicy) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=422, detail=f"Crop fileIndex={file_index} is empty")
    if len(data) > policy.max_crop_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Crop fileIndex={file_index} exceeds the byte limit",
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > policy.max_crop_pixels:
                raise HTTPException(
                    status_code=413,
                    detail=f"Crop fileIndex={file_index} exceeds the pixel limit",
                )
            return np.asarray(source.convert("RGB")).copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Crop fileIndex={file_index} is not a readable image",
        ) from exc


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{label} must be a non-empty string")
    return value.strip()


def _parse_manifest(
    raw: str,
    crop_count: int,
    policy: QualityPolicy,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="manifest must be an object")
    if value.get("contractVersion", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise HTTPException(status_code=422, detail="Unsupported manifest contractVersion")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=422, detail="manifest.items must be a non-empty array")
    if len(items) > policy.max_batch_size:
        raise HTTPException(status_code=413, detail="OCR batch exceeds the configured item limit")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for offset, item in enumerate(items):
        label = f"manifest.items[{offset}]"
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail=f"{label} must be an object")
        crop_id = _optional_string(item.get("cropId"), f"{label}.cropId")
        assert crop_id is not None
        if crop_id in seen:
            raise HTTPException(status_code=422, detail=f"Duplicate cropId: {crop_id}")
        seen.add(crop_id)
        file_index = item.get("fileIndex")
        if (
            isinstance(file_index, bool)
            or not isinstance(file_index, int)
            or not 0 <= file_index < crop_count
        ):
            raise HTTPException(status_code=422, detail=f"{label}.fileIndex is invalid")
        frame_index = item.get("frameIndex")
        if frame_index is not None and (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise HTTPException(status_code=422, detail=f"{label}.frameIndex is invalid")
        timestamp = item.get("timestamp")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            raise HTTPException(status_code=422, detail=f"{label}.timestamp is invalid")
        parsed.append(
            {
                "cropId": crop_id,
                "fileIndex": file_index,
                "observationId": _optional_string(
                    item.get("observationId"), f"{label}.observationId"
                ),
                "trackletId": _optional_string(item.get("trackletId"), f"{label}.trackletId"),
                "frameIndex": frame_index,
                "timestamp": float(timestamp) if timestamp is not None else None,
            }
        )
    return parsed


def _quality(image: np.ndarray, policy: QualityPolicy) -> tuple[dict[str, Any], list[str]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    reasons: list[str] = []
    if width < policy.minimum_width or height < policy.minimum_height:
        reasons.append("crop-too-small")
    if sharpness < policy.minimum_sharpness:
        reasons.append("crop-too-blurry")
    if contrast < policy.minimum_contrast:
        reasons.append("crop-too-low-contrast")
    return (
        {
            "cropWidth": width,
            "cropHeight": height,
            "sharpness": round(sharpness, 4),
            "contrast": round(contrast, 4),
        },
        reasons,
    )


def _ascii_digits(value: str) -> str:
    converted: list[str] = []
    for character in value:
        if not character.isdigit():
            converted.append(character)
            continue
        try:
            converted.append(str(unicodedata.digit(character)))
        except (TypeError, ValueError):
            converted.append(character)
    return "".join(converted)


def _number_candidates(raw: Sequence[RawTextCandidate]) -> list[dict[str, Any]]:
    by_number: dict[str, dict[str, Any]] = {}
    for candidate in raw:
        text = str(candidate.text)
        confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
        for match in re.finditer(r"(?<!\d)(\d{1,2})(?!\d)", _ascii_digits(text)):
            number = match.group(1)
            current = by_number.get(number)
            value = {
                "number": number,
                "confidence": round(confidence, 6),
                "rawText": text,
                "polygon": candidate.polygon,
            }
            if current is None or confidence > float(current["confidence"]):
                by_number[number] = value
    return sorted(
        by_number.values(),
        key=lambda item: (-float(item["confidence"]), item["number"]),
    )[:5]


def _decision(
    raw: Sequence[RawTextCandidate],
    policy: QualityPolicy,
) -> tuple[str, str | None, float | None, list[dict[str, Any]], list[str]]:
    candidates = _number_candidates(raw)
    if not candidates:
        return "no-number", None, None, [], ["no-numeric-text"]
    best = candidates[0]
    confidence = float(best["confidence"])
    if confidence < policy.minimum_confidence:
        return "low-confidence", None, None, candidates, ["confidence-below-threshold"]
    if (
        len(candidates) > 1
        and candidates[1]["number"] != best["number"]
        and confidence - float(candidates[1]["confidence"]) <= policy.ambiguity_margin
    ):
        return "ambiguous", None, None, candidates, ["competing-numbers"]
    return "recognized", str(best["number"]), confidence, candidates, []


def _provider_info(provider: JerseyOcrProvider) -> dict[str, Any]:
    info = provider.info()
    if not isinstance(info.get("backend"), str) or not info["backend"]:
        raise ProviderUnavailable("OCR provider did not report a backend")
    if not isinstance(info.get("modelVersion"), str) or not info["modelVersion"]:
        raise ProviderUnavailable("OCR provider did not report modelVersion")
    return info


def create_app(
    provider: JerseyOcrProvider | None = None,
    *,
    quality_policy: QualityPolicy | None = None,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or provider_from_environment()
    policy_error: str | None = None
    try:
        policy = quality_policy or QualityPolicy.from_environment()
        policy.validate()
    except (TypeError, ValueError, ProviderUnavailable) as exc:
        # A bad deployment setting must remain visible through readiness; it
        # must not kill the process and make liveness indistinguishable from a
        # crash loop.
        policy = QualityPolicy()
        policy_error = f"Invalid jersey OCR policy: {exc}"
    should_preload = (
        os.environ.get("JERSEY_OCR_PRELOAD", "1") not in {"0", "false", "False"}
        if preload is None
        else preload
    )
    try:
        result_cache = OcrResultCache(
            int(os.environ.get("JERSEY_OCR_CACHE_MAX_ENTRIES", "4096")),
            float(os.environ.get("JERSEY_OCR_CACHE_TTL_SECONDS", "86400")),
        )
    except (TypeError, ValueError) as exc:
        result_cache = OcrResultCache(0, 0)
        policy_error = policy_error or f"Invalid jersey OCR cache policy: {exc}"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.provider_error = None
        if should_preload:
            try:
                await run_in_threadpool(configured_provider.load)
            except Exception as exc:
                application.state.provider_error = str(exc)
        yield

    application = FastAPI(
        title="Replay Studio Jersey OCR Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
    application.state.provider_error = None

    async def ensure_loaded() -> dict[str, Any]:
        if policy_error is not None:
            raise HTTPException(status_code=503, detail=policy_error)
        if not configured_provider.loaded:
            try:
                await run_in_threadpool(configured_provider.load)
                application.state.provider_error = None
            except Exception as exc:
                application.state.provider_error = str(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            return _provider_info(configured_provider)
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/health/live")
    async def health_live() -> dict[str, Any]:
        return {"status": "ok", "service": SERVICE_NAME}

    @application.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        info = await ensure_loaded()
        return {
            "status": "ready",
            "service": SERVICE_NAME,
            "contractVersion": CONTRACT_VERSION,
            "capabilities": _capabilities(),
            **info,
        }

    @application.post("/v1/analyze")
    async def analyze(
        crops: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict[str, Any]:
        info = await ensure_loaded()
        if not crops:
            raise HTTPException(status_code=422, detail="At least one crop is required")
        if len(crops) > policy.max_batch_size:
            raise HTTPException(status_code=413, detail="OCR batch exceeds the file limit")
        crop_bytes = [await crop.read() for crop in crops]
        images = [
            _decode_image(data, file_index, policy)
            for file_index, data in enumerate(crop_bytes)
        ]
        items = _parse_manifest(manifest, len(images), policy)
        responses: list[dict[str, Any]] = []
        samples: list[OcrSample] = []
        sample_cache_keys: list[str | None] = []
        sample_response_indices: list[list[int]] = []
        pending_by_cache_key: dict[str, int] = {}
        cache_hits = 0
        request_deduplicated = 0
        use_crop_cache = info.get("inferenceScope", "crop") == "crop"
        for item in items:
            image = images[int(item["fileIndex"])]
            quality, rejection_reasons = _quality(image, policy)
            response = {
                "cropId": item["cropId"],
                "observationId": item["observationId"],
                "trackletId": item["trackletId"],
                "frameIndex": item["frameIndex"],
                "timestamp": item["timestamp"],
                "evidenceFingerprint": _evidence_fingerprint(image),
                "usable": not rejection_reasons,
                "status": "rejected" if rejection_reasons else "pending",
                "number": None,
                "confidence": None,
                "candidates": [],
                "quality": quality,
                "rejectionReasons": rejection_reasons,
                "decisionReasons": [],
            }
            responses.append(response)
            if not rejection_reasons:
                cache_key = _cache_key(image, info, policy) if use_crop_cache else None
                cached = result_cache.get(cache_key) if cache_key is not None else None
                if cached is not None:
                    status, number, confidence, candidates, reasons = _decision(cached, policy)
                    responses[-1].update(
                        {
                            "status": status,
                            "number": number,
                            "confidence": confidence,
                            "candidates": candidates,
                            "decisionReasons": reasons,
                            "cacheHit": True,
                        }
                    )
                    cache_hits += 1
                    continue
                if cache_key is not None and cache_key in pending_by_cache_key:
                    sample_response_indices[pending_by_cache_key[cache_key]].append(
                        len(responses) - 1
                    )
                    request_deduplicated += 1
                    continue
                samples.append(
                    OcrSample(
                        crop_id=item["cropId"], image_rgb=image,
                        tracklet_id=item["trackletId"], observation_id=item["observationId"],
                        frame_index=item["frameIndex"], timestamp=item["timestamp"],
                    )
                )
                sample_cache_keys.append(cache_key)
                sample_response_indices.append([len(responses) - 1])
                if cache_key is not None:
                    pending_by_cache_key[cache_key] = len(samples) - 1
        try:
            recognized: list[OcrResult] = await run_in_threadpool(
                configured_provider.recognize, samples
            )
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR provider failed: {exc}") from exc
        if len(recognized) != len(samples):
            raise HTTPException(status_code=503, detail="OCR provider returned an incomplete batch")
        for response_indices, cache_key, sample, result in zip(
            sample_response_indices, sample_cache_keys, samples, recognized
        ):
            if result.crop_id != sample.crop_id:
                raise HTTPException(status_code=503, detail="OCR provider changed crop order")
            status, number, confidence, candidates, reasons = _decision(
                result.candidates, policy
            )
            if cache_key is not None:
                result_cache.put(cache_key, result.candidates)
            for response_index in response_indices:
                responses[response_index].update(
                    {
                        "status": status,
                        "number": number,
                        "confidence": confidence,
                        "candidates": candidates,
                        "decisionReasons": reasons,
                        "cacheHit": False,
                    }
                )
        usable_fingerprints = [
            str(item["evidenceFingerprint"])
            for item in responses
            if item["usable"] is True
        ]
        unique_usable_fingerprints = set(usable_fingerprints)
        return {
            "contractVersion": CONTRACT_VERSION,
            "capabilities": _capabilities(),
            **info,
            "items": responses,
            "diagnostics": {
                "requestedCropCount": len(responses),
                "usableCropCount": sum(item["usable"] for item in responses),
                "recognizedCropCount": sum(
                    item["status"] == "recognized" for item in responses
                ),
                "ambiguousCropCount": sum(
                    item["status"] == "ambiguous" for item in responses
                ),
                "rejectedCropCount": sum(not item["usable"] for item in responses),
                "providerInferenceCropCount": len(samples),
                "cacheHitCount": cache_hits,
                "requestDeduplicatedCount": request_deduplicated,
                "uniqueEvidenceFingerprintCount": len(unique_usable_fingerprints),
                "duplicateEvidenceFingerprintCount": (
                    len(usable_fingerprints) - len(unique_usable_fingerprints)
                ),
                "cacheEnabled": result_cache.enabled and use_crop_cache,
            },
        }

    return application


app = create_app()
