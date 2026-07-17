from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .cache import (
    CACHE_SCHEMA_VERSION,
    IdentityCacheEntry,
    IdentityEmbeddingCache,
)
from .providers import (
    EMBEDDING_DIMENSION,
    EmbeddingSample,
    IdentityEmbeddingProvider,
    PRTReIDProvider,
    ProviderEmbedding,
    ProviderUnavailable,
)


EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
KNOWN_IDENTITY_ROLES = {"ball", "goalkeeper", "other", "player", "referee"}


@dataclass(frozen=True)
class QualityPolicy:
    minimum_width: int = 16
    minimum_height: int = 30
    minimum_sharpness: float = 12.0
    padding_ratio: float = 0.08

    @classmethod
    def from_environment(cls) -> "QualityPolicy":
        return cls(
            minimum_width=max(1, int(os.environ.get("REID_MIN_CROP_WIDTH", "16"))),
            minimum_height=max(1, int(os.environ.get("REID_MIN_CROP_HEIGHT", "30"))),
            minimum_sharpness=max(0.0, float(os.environ.get("REID_MIN_SHARPNESS", "12"))),
            padding_ratio=max(0.0, float(os.environ.get("REID_CROP_PADDING", "0.08"))),
        )


def _decode_image(data: bytes, file_index: int) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(image)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Frame fileIndex={file_index} is not a readable image",
        ) from exc


def _parse_manifest(raw: str, frame_count: int) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="manifest is not valid JSON") from exc
    frame_items = value.get("frames") if isinstance(value, dict) else None
    if not isinstance(frame_items, list):
        raise HTTPException(status_code=422, detail="manifest.frames must be an array")
    seen: set[str] = set()
    for frame in frame_items:
        if not isinstance(frame, dict):
            raise HTTPException(status_code=422, detail="Each manifest frame must be an object")
        file_index = frame.get("fileIndex")
        if not isinstance(file_index, int) or not 0 <= file_index < frame_count:
            raise HTTPException(status_code=422, detail="manifest fileIndex is out of range")
        observations = frame.get("observations")
        if not isinstance(observations, list):
            raise HTTPException(status_code=422, detail="frame.observations must be an array")
        for observation in observations:
            observation_id = observation.get("observationId") if isinstance(observation, dict) else None
            bbox = observation.get("bbox") if isinstance(observation, dict) else None
            if not isinstance(observation_id, str) or not observation_id.strip():
                raise HTTPException(status_code=422, detail="observationId is required")
            if observation_id in seen:
                raise HTTPException(status_code=422, detail=f"Duplicate observationId: {observation_id}")
            seen.add(observation_id)
            if not isinstance(bbox, dict) or any(
                not isinstance(bbox.get(key), (int, float))
                for key in ("x", "y", "width", "height")
            ):
                raise HTTPException(status_code=422, detail=f"Invalid bbox for {observation_id}")
            if float(bbox["width"]) <= 0 or float(bbox["height"]) <= 0:
                raise HTTPException(status_code=422, detail=f"Empty bbox for {observation_id}")
    return frame_items


def _crop(
    image: np.ndarray,
    bbox: dict,
    policy: QualityPolicy,
) -> tuple[np.ndarray, dict, list[str]]:
    height, width = image.shape[:2]
    x = float(bbox["x"])
    y = float(bbox["y"])
    box_width = float(bbox["width"])
    box_height = float(bbox["height"])
    padding_x = box_width * policy.padding_ratio
    padding_y = box_height * policy.padding_ratio
    requested = (
        int(np.floor(x - padding_x)),
        int(np.floor(y - padding_y)),
        int(np.ceil(x + box_width + padding_x)),
        int(np.ceil(y + box_height + padding_y)),
    )
    x1 = min(width, max(0, requested[0]))
    y1 = min(height, max(0, requested[1]))
    x2 = min(width, max(0, requested[2]))
    y2 = min(height, max(0, requested[3]))
    crop = image[y1:y2, x1:x2]
    border_clipped = requested != (x1, y1, x2, y2)
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if box_width < policy.minimum_width or box_height < policy.minimum_height:
        reasons.append("crop-too-small")
    if not crop.size:
        reasons.append("crop-outside-frame")
    elif sharpness < policy.minimum_sharpness:
        reasons.append("crop-too-blurry")
    quality = {
        "cropWidth": int(max(0, x2 - x1)),
        "cropHeight": int(max(0, y2 - y1)),
        "sourceBoxWidth": round(box_width, 3),
        "sourceBoxHeight": round(box_height, 3),
        "borderClipped": border_clipped,
        "sharpness": round(sharpness, 4),
    }
    return crop, quality, reasons


def _validated_embedding(item: ProviderEmbedding) -> np.ndarray:
    vector = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
        raise ProviderUnavailable("Provider returned an invalid embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ProviderUnavailable("Provider returned a zero embedding")
    return vector / norm


def _cache_key(
    crop: np.ndarray,
    bbox: dict,
    policy: QualityPolicy,
    provider_info: dict[str, Any],
) -> str:
    """Fingerprint exact crop content and every semantic cache dependency."""

    namespace = {
        "schema": CACHE_SCHEMA_VERSION,
        "backend": provider_info.get("backend"),
        "dimension": provider_info.get("dimension"),
        "normalized": provider_info.get("normalized"),
        "modelVersion": provider_info.get("modelVersion"),
        "checkpointSha256": provider_info.get("checkpointSha256")
        or provider_info.get("modelVersion"),
        "hrnetCheckpointSha256": provider_info.get("hrnetCheckpointSha256"),
        "soccerNetCommit": provider_info.get("soccerNetCommit"),
        "bbox": {
            key: float(bbox[key]).hex()
            for key in ("x", "y", "width", "height")
        },
        "cropPolicy": {
            "minimumWidth": policy.minimum_width,
            "minimumHeight": policy.minimum_height,
            "minimumSharpness": float(policy.minimum_sharpness).hex(),
            "paddingRatio": float(policy.padding_ratio).hex(),
        },
        "cropShape": list(crop.shape),
        "cropDtype": str(crop.dtype),
    }
    digest = sha256(
        json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(crop).tobytes())
    return digest.hexdigest()


def _evidence_fingerprint(crop: np.ndarray) -> str:
    """Return an observation-independent fingerprint of decoded crop pixels."""

    digest = sha256()
    digest.update(EVIDENCE_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(crop.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(crop.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(crop).tobytes())
    return f"{EVIDENCE_FINGERPRINT_VERSION}:{digest.hexdigest()}"


def _rejected_cache_entry(quality: dict, reasons: list[str]) -> IdentityCacheEntry:
    return IdentityCacheEntry(
        usable=False,
        quality=dict(quality),
        rejection_reasons=tuple(str(reason) for reason in reasons),
    )


def _provider_cache_entry(
    item: ProviderEmbedding,
    quality: dict,
) -> IdentityCacheEntry:
    vector = _validated_embedding(item)
    visibility_scores: tuple[float, ...] | None = None
    if item.visibility_scores is not None:
        raw_visibility = np.asarray(item.visibility_scores, dtype=np.float32).reshape(-1)
        if not np.isfinite(raw_visibility).all():
            raise ProviderUnavailable("Provider returned non-finite visibility scores")
        visibility_scores = tuple(float(value) for value in raw_visibility)
    role = item.role
    if role is not None and role not in KNOWN_IDENTITY_ROLES:
        raise ProviderUnavailable("Provider returned an invalid role")
    role_confidence = item.role_confidence
    if role is None and role_confidence is not None:
        raise ProviderUnavailable("Provider returned role confidence without a role")
    if role is not None and role_confidence is None:
        raise ProviderUnavailable("Provider returned a role without confidence")
    if role_confidence is not None:
        role_confidence = float(role_confidence)
        if not np.isfinite(role_confidence) or not 0.0 <= role_confidence <= 1.0:
            raise ProviderUnavailable("Provider returned an invalid role confidence")
    return IdentityCacheEntry(
        usable=True,
        quality=dict(quality),
        rejection_reasons=(),
        embedding=tuple(float(value) for value in vector),
        visibility_scores=visibility_scores,
        role=role,
        role_confidence=role_confidence,
    )


def _apply_cache_entry(
    response: dict,
    entry: IdentityCacheEntry,
    *,
    cache_source: str,
) -> None:
    response.update(
        {
            "usable": entry.usable,
            "quality": dict(entry.quality),
            "rejectionReasons": list(entry.rejection_reasons),
            "embedding": list(entry.embedding) if entry.embedding is not None else None,
            "visibilityScores": (
                list(entry.visibility_scores)
                if entry.visibility_scores is not None
                else None
            ),
            "role": entry.role,
            "roleConfidence": entry.role_confidence,
            "cacheHit": cache_source == "cache-hit",
            "cacheSource": cache_source,
        }
    )


@dataclass(slots=True)
class _RequestGroup:
    key: str
    crop: np.ndarray
    quality: dict
    reasons: list[str]
    representative_observation_id: str
    response_indices: list[int]


def create_app(
    provider: IdentityEmbeddingProvider | None = None,
    *,
    quality_policy: QualityPolicy | None = None,
    embedding_cache: IdentityEmbeddingCache | None = None,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or PRTReIDProvider()
    policy = quality_policy or QualityPolicy.from_environment()
    cache = embedding_cache or IdentityEmbeddingCache.from_environment(
        dimension=EMBEDDING_DIMENSION,
        environment=os.environ,
    )
    should_preload = (
        os.environ.get("REID_PRELOAD", "1") not in {"0", "false", "False"}
        if preload is None
        else preload
    )

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
        title="Replay Studio Identity Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
    application.state.quality_policy = policy
    application.state.embedding_cache = cache
    application.state.provider_error = None

    async def ensure_loaded() -> None:
        if configured_provider.loaded:
            application.state.provider_error = None
            return
        try:
            await run_in_threadpool(configured_provider.load)
            application.state.provider_error = None
        except Exception as exc:
            application.state.provider_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/health/live")
    async def health_live() -> dict:
        return {
            "status": "ok",
            "service": "replay-studio-identity-worker",
        }

    @application.get("/health/ready")
    async def health_ready() -> dict:
        await ensure_loaded()
        if not configured_provider.loaded:
            raise HTTPException(status_code=503, detail="Identity provider is not loaded")
        return {
            "status": "ready",
            **configured_provider.info(),
            "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
            "cache": cache.stats(),
        }

    @application.post("/v1/embeddings")
    async def embeddings(
        frames: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict:
        await ensure_loaded()
        if not frames:
            raise HTTPException(status_code=422, detail="At least one frame is required")
        frame_bytes = [await frame.read() for frame in frames]
        decoded = [_decode_image(data, index) for index, data in enumerate(frame_bytes)]
        frame_items = _parse_manifest(manifest, len(decoded))
        provider_info = configured_provider.info()
        responses: list[dict] = []
        groups: dict[str, _RequestGroup] = {}
        in_request_deduplicated = 0
        for frame in frame_items:
            image = decoded[int(frame["fileIndex"])]
            frame_index = int(frame.get("frameIndex") or 0)
            for observation in frame["observations"]:
                crop, quality, reasons = _crop(image, observation["bbox"], policy)
                response = {
                    "observationId": observation["observationId"],
                    "frameIndex": frame_index,
                    "usable": not reasons,
                    "quality": quality,
                    "rejectionReasons": reasons,
                    "embedding": None,
                    "visibilityScores": None,
                    "role": None,
                    "roleConfidence": None,
                    "evidenceFingerprint": _evidence_fingerprint(crop),
                    "cacheHit": False,
                    "cacheSource": "pending",
                }
                responses.append(response)
                key = _cache_key(crop, observation["bbox"], policy, provider_info)
                group = groups.get(key)
                if group is not None:
                    group.response_indices.append(len(responses) - 1)
                    in_request_deduplicated += 1
                    continue
                groups[key] = _RequestGroup(
                    key=key,
                    crop=crop,
                    quality=quality,
                    reasons=reasons,
                    representative_observation_id=str(observation["observationId"]),
                    response_indices=[len(responses) - 1],
                )

        cache.note_in_request_deduplicated(in_request_deduplicated)
        unresolved = list(groups)
        request_hits = 0
        request_misses = 0
        concurrent_deduplicated = 0
        corrupt_misses = 0
        expired_misses = 0
        provider_inference_count = 0
        resolution_round = 0
        while unresolved:
            resolution_round += 1
            if resolution_round > 3:
                raise HTTPException(
                    status_code=503,
                    detail="Identity cache single-flight could not resolve a concurrent miss",
                )
            reservation = cache.reserve_many(unresolved)
            corrupt_misses += reservation.corrupt_misses
            expired_misses += reservation.expired_misses
            next_unresolved: list[str] = []

            for key, entry in reservation.hits.items():
                request_hits += 1
                for response_index in groups[key].response_indices:
                    _apply_cache_entry(
                        responses[response_index], entry, cache_source="cache-hit"
                    )

            request_misses += len(reservation.owners)
            rejected_entries: dict[str, IdentityCacheEntry] = {}
            provider_keys: list[str] = []
            samples: list[EmbeddingSample] = []
            for key in reservation.owners:
                group = groups[key]
                if group.reasons:
                    rejected_entries[key] = _rejected_cache_entry(
                        group.quality, group.reasons
                    )
                    continue
                provider_keys.append(key)
                samples.append(
                    EmbeddingSample(
                        observation_id=group.representative_observation_id,
                        image_rgb=group.crop,
                    )
                )

            if rejected_entries:
                cache.publish(rejected_entries)
                for key, entry in rejected_entries.items():
                    for response_index in groups[key].response_indices:
                        _apply_cache_entry(
                            responses[response_index], entry, cache_source="qa-computed"
                        )

            if samples:
                provider_inference_count += len(samples)
                try:
                    embedded = await run_in_threadpool(configured_provider.embed, samples)
                    if len(embedded) != len(samples):
                        raise ProviderUnavailable(
                            "Provider returned an incomplete batch"
                        )
                    provider_entries: dict[str, IdentityCacheEntry] = {}
                    for key, sample, item in zip(provider_keys, samples, embedded):
                        if item.observation_id != sample.observation_id:
                            raise ProviderUnavailable(
                                "Provider changed observation order"
                            )
                        provider_entries[key] = _provider_cache_entry(
                            item, groups[key].quality
                        )
                    cache.publish(provider_entries)
                except ProviderUnavailable as exc:
                    cache.fail(provider_keys)
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    cache.fail(provider_keys)
                    raise HTTPException(
                        status_code=503,
                        detail=f"Identity provider failed: {exc}",
                    ) from exc
                for key, entry in provider_entries.items():
                    for response_index in groups[key].response_indices:
                        _apply_cache_entry(
                            responses[response_index], entry, cache_source="provider"
                        )

            if reservation.waiters:
                concurrent_deduplicated += len(reservation.waiters)
                waited = await run_in_threadpool(
                    cache.wait_many, reservation.waiters
                )
                for key, entry in waited.items():
                    for response_index in groups[key].response_indices:
                        _apply_cache_entry(
                            responses[response_index],
                            entry,
                            cache_source="concurrent-deduplicated",
                        )
                next_unresolved.extend(
                    key for key in reservation.waiters if key not in waited
                )
            unresolved = next_unresolved

        usable_count = sum(item["usable"] is True for item in responses)
        rejected_count = len(responses) - usable_count
        usable_fingerprints = [
            str(item["evidenceFingerprint"])
            for item in responses
            if item["usable"] is True
        ]
        unique_usable_fingerprints = set(usable_fingerprints)
        return {
            **provider_info,
            "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
            "items": responses,
            "diagnostics": {
                "requestedObservationCount": len(responses),
                "usableObservationCount": usable_count,
                "rejectedObservationCount": rejected_count,
                "cacheHitCount": request_hits,
                "cacheMissCount": request_misses,
                "deduplicatedObservationCount": in_request_deduplicated,
                "concurrentDeduplicatedCount": concurrent_deduplicated,
                "providerInferenceCount": provider_inference_count,
                "corruptCacheMissCount": corrupt_misses,
                "expiredCacheMissCount": expired_misses,
                "uniqueEvidenceFingerprintCount": len(unique_usable_fingerprints),
                "duplicateEvidenceFingerprintCount": (
                    len(usable_fingerprints) - len(unique_usable_fingerprints)
                ),
                "cache": cache.stats(),
            },
        }

    return application


app = create_app()
