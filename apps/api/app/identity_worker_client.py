from __future__ import annotations

"""Batch orchestration for the optional provider-neutral identity worker.

Contract v2: the detection pass already cut and QA-gated every person crop
into the person crop store, so batches upload crop bytes — never full
frames — and the disk cache is keyed by the exact crop digest. Crops whose
bytes are missing from the store degrade to explicit rejected items instead
of failing the batch.
"""

import json
from pathlib import Path
from time import sleep
from typing import Callable, Sequence

from .config import get_settings
from .identity_embedding_cache import (
    IdentityEmbeddingCacheError,
    lookup_identity_embedding_cache,
    store_identity_embedding_cache,
)
from .identity_worker_batch_validation import validate_embedding_payload
from .identity_worker_contract import (
    IDENTITY_REQUEST_CONTRACT_VERSION,
    IdentityWorkerBatchResult,
    IdentityWorkerError,
)
from .identity_worker_model_contract import (
    project_model_contract,
    validate_readiness_payload,
)
from .identity_worker_transport import (
    IdentityWorkerTransportError,
    fetch_identity_readiness,
    post_identity_batch,
)
from .person_crop_store import lookup_person_crop_envelope, person_crop_store_runtime


# One sampled frame's sendable crops: (frame_index, frame_sha256,
# observations[{observationId, cropSha256, quality}]).
IdentityFrameRequest = tuple[int, str, list[dict]]


def identity_worker_readiness(*, timeout: float = 2.0) -> dict:
    """Report the optional ReID dependency without making the API unhealthy."""

    settings = get_settings()
    if not settings.identity_worker_url:
        return {"configured": False, "status": "disabled", "backend": None}
    try:
        payload = fetch_identity_readiness(
            settings.identity_worker_url,
            timeout=timeout,
        )
    except IdentityWorkerTransportError as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": None,
            "detail": str(exc),
        }
    try:
        value = validate_readiness_payload(payload)
    except IdentityWorkerError:
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": payload.get("backend") if isinstance(payload, dict) else None,
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": value["backend"],
        "device": value.get("device"),
        "batchSize": value.get("batchSize"),
        "dimension": value["dimension"],
        "normalized": True,
        "modelVersion": value["modelVersion"],
        "evidenceFingerprintVersion": value["evidenceFingerprintVersion"],
        "modelLoadSeconds": value.get("modelLoadSeconds"),
        "soccerNetCommit": value.get("soccerNetCommit"),
    }


def _validate_frame_requests(frames: Sequence[IdentityFrameRequest]) -> None:
    requested_ids: set[str] = set()
    for _frame_index, frame_sha256, observations in frames:
        digest = str(frame_sha256 or "").strip().lower()
        if len(digest) != 64:
            raise IdentityWorkerError(
                "Every identity frame request requires the frame content digest"
            )
        for observation in observations:
            observation_id = observation.get("observationId")
            if not isinstance(observation_id, str) or not observation_id:
                raise IdentityWorkerError(
                    "Every identity observation requires observationId"
                )
            if observation_id in requested_ids:
                raise IdentityWorkerError(
                    f"Duplicate identity observation: {observation_id}"
                )
            crop_digest = observation.get("cropSha256")
            if not isinstance(crop_digest, str) or len(crop_digest) != 64:
                raise IdentityWorkerError(
                    f"Observation {observation_id} has no crop digest"
                )
            if not isinstance(observation.get("quality"), dict):
                raise IdentityWorkerError(
                    f"Observation {observation_id} has no crop quality evidence"
                )
            requested_ids.add(observation_id)


def crop_store_rejected_item(
    observation: dict,
    frame_index: int,
    *,
    reason: str = "crop-store-unavailable",
) -> dict:
    """A locally rejected item for a crop whose bytes never reached a worker."""

    return {
        "observationId": str(observation["observationId"]),
        "frameIndex": int(frame_index),
        "usable": False,
        "quality": dict(observation.get("quality") or {}),
        "rejectionReasons": [reason],
        "embedding": None,
        "visibilityScores": None,
        "role": None,
        "roleConfidence": None,
        "evidenceFingerprint": None,
        "cacheHit": False,
        "cacheSource": "crop-store",
    }


def _batch_payload(
    batch: Sequence[IdentityFrameRequest],
    store_directory: Path,
    policy,
) -> tuple[
    list[tuple[str, tuple[str, bytes, str]]],
    str,
    set[str],
    list[tuple[dict, int]],
]:
    """Load crop bytes for one batch; missing crops become local rejections."""

    files: list[tuple[str, tuple[str, bytes, str]]] = []
    manifest_crops: list[dict] = []
    missing: list[tuple[dict, int]] = []
    sent_ids: set[str] = set()
    for frame_index, frame_sha256, observations in batch:
        records = lookup_person_crop_envelope(
            store_directory,
            frame_sha256=frame_sha256,
            policy=policy,
        )
        for observation in observations:
            observation_id = str(observation["observationId"])
            record = (records or {}).get(observation_id)
            if (
                record is None
                or not record.crop_jpeg
                or record.crop_sha256 != observation["cropSha256"]
            ):
                missing.append((observation, int(frame_index)))
                continue
            manifest_crops.append(
                {
                    "observationId": observation_id,
                    "frameIndex": int(frame_index),
                    "fileIndex": len(files),
                    "quality": dict(observation.get("quality") or {}),
                }
            )
            files.append(
                (
                    "crops",
                    (
                        f"crop-{len(files):04d}.jpg",
                        record.crop_jpeg,
                        "image/jpeg",
                    ),
                )
            )
            sent_ids.add(observation_id)
    manifest = json.dumps(
        {
            "contractVersion": IDENTITY_REQUEST_CONTRACT_VERSION,
            "crops": manifest_crops,
        },
        separators=(",", ":"),
    )
    return files, manifest, sent_ids, missing


def _merge_diagnostics(target: dict, values: dict) -> None:
    for field, value in values.items():
        if field == "cache":
            target[field] = value
        else:
            target[field] = int(target.get(field, 0)) + int(value)


def _post_batch_with_retry(
    worker_url: str,
    *,
    files: list[tuple[str, tuple[str, bytes, str]]],
    manifest: str,
    timeout: float,
    retry_count: int,
) -> tuple[object, int]:
    """Retry only transient transport failures; contract errors stay fatal."""

    attempts = 0
    while True:
        attempts += 1
        try:
            payload = post_identity_batch(
                worker_url,
                files=files,
                manifest=manifest,
                timeout=timeout,
            )
        except IdentityWorkerTransportError:
            if attempts > retry_count:
                raise
            sleep(min(2.0, 0.5 * attempts))
        else:
            return payload, attempts


def _disk_cache_model_contract(settings) -> dict[str, object] | None:
    """Resolve the worker's model contract; no identity means no caching."""

    try:
        payload = fetch_identity_readiness(
            settings.identity_worker_url,
            timeout=2.0,
        )
        return project_model_contract(validate_readiness_payload(payload))
    except (IdentityWorkerError, KeyError, TypeError, ValueError):
        return None


def embed_identity_frames(
    frames: Sequence[IdentityFrameRequest],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> IdentityWorkerBatchResult:
    """Embed stored person crops, batching whole sampled frames over HTTP."""

    settings = get_settings()
    if not settings.identity_worker_url or not frames:
        return IdentityWorkerBatchResult()
    _validate_frame_requests(frames)
    store_directory, crop_policy = person_crop_store_runtime()
    batch_size = max(1, int(settings.identity_worker_batch_size))
    effective_timeout = max(
        0.1,
        min(float(timeout), float(settings.identity_worker_timeout))
        if timeout is not None
        else float(settings.identity_worker_timeout),
    )
    retry_count = max(0, int(settings.identity_worker_batch_retry_count))
    result = IdentityWorkerBatchResult()
    usable_count = 0
    retried_batch_count = 0
    crop_store_miss_count = 0
    accepted_model_contract: dict[str, object] | None = None

    cache_directory = (
        Path(settings.media_root) / "identity-embeddings"
        if settings.identity_embedding_cache_enabled
        else None
    )
    cache_contract = (
        _disk_cache_model_contract(settings)
        if cache_directory is not None
        else None
    )
    disk_hits = 0
    disk_misses = 0
    disk_write_errors = 0
    # Rebuild each frame with only the observations the disk cache missed.
    crop_digest_by_observation: dict[str, str] = {}
    pending: list[IdentityFrameRequest] = []
    if cache_contract is None:
        pending = list(frames)
    else:
        accepted_model_contract = dict(cache_contract)
        result.diagnostics["modelContract"] = dict(cache_contract)
        for frame_index, frame_sha256, observations in frames:
            missed: list[dict] = []
            for observation in observations:
                crop_digest = str(observation["cropSha256"])
                try:
                    lookup = lookup_identity_embedding_cache(
                        cache_directory,
                        crop_sha256=crop_digest,
                        model_contract=cache_contract,
                    )
                except IdentityEmbeddingCacheError:
                    lookup = None
                observation_id = str(observation["observationId"])
                if lookup is not None and lookup.status == "hit":
                    disk_hits += 1
                    item = lookup.entry.detached_item()
                    result.items_by_observation_id[observation_id] = {
                        **item,
                        "observationId": observation_id,
                        "frameIndex": int(frame_index),
                        "provider": cache_contract["backend"],
                        "modelVersion": cache_contract["modelVersion"],
                    }
                    usable_count += int(item.get("usable") is True)
                    continue
                disk_misses += 1
                crop_digest_by_observation[observation_id] = crop_digest
                missed.append(observation)
            if missed:
                pending.append((frame_index, frame_sha256, missed))
    fully_cached_frames = len(frames) - len(pending)
    if on_progress is not None and fully_cached_frames:
        on_progress(fully_cached_frames, len(frames), usable_count)

    for start in range(0, len(pending), batch_size):
        batch = list(pending[start : start + batch_size])
        files, manifest, sent_ids, missing = _batch_payload(
            batch,
            store_directory,
            crop_policy,
        )
        for observation, frame_index in missing:
            crop_store_miss_count += 1
            observation_id = str(observation["observationId"])
            result.items_by_observation_id[observation_id] = (
                crop_store_rejected_item(observation, frame_index)
            )
        if files:
            try:
                payload, attempts = _post_batch_with_retry(
                    settings.identity_worker_url,
                    files=files,
                    manifest=manifest,
                    timeout=effective_timeout,
                    retry_count=retry_count,
                )
            except IdentityWorkerTransportError as exc:
                # Keep the embeddings that earlier batches already produced;
                # the caller decides how to present the explicit partial
                # failure.
                result.diagnostics["partialFailure"] = {
                    "failedFrameIndex": int(batch[0][0]),
                    "processedFrameCount": fully_cached_frames + start,
                    "requestedFrameCount": len(frames),
                    "attempts": retry_count + 1,
                    "detail": str(exc),
                }
                break
            retried_batch_count += attempts - 1
            batch_contract, items, diagnostics = validate_embedding_payload(
                payload,
                sent_ids,
            )
            if accepted_model_contract is None:
                accepted_model_contract = batch_contract
                result.diagnostics["modelContract"] = dict(batch_contract)
            elif batch_contract != accepted_model_contract:
                changed_fields = sorted(
                    field
                    for field, value in batch_contract.items()
                    if accepted_model_contract.get(field) != value
                )
                raise IdentityWorkerError(
                    "Identity worker changed model contract between batches: "
                    + ", ".join(changed_fields)
                )
            _merge_diagnostics(result.diagnostics, diagnostics)
            for observation_id, item in items.items():
                result.items_by_observation_id[observation_id] = {
                    **item,
                    "provider": batch_contract["backend"],
                    "modelVersion": batch_contract["modelVersion"],
                }
                usable_count += int(item.get("usable") is True)
                crop_digest = crop_digest_by_observation.get(observation_id)
                if crop_digest is not None and cache_contract is not None:
                    try:
                        store_identity_embedding_cache(
                            cache_directory,
                            crop_sha256=crop_digest,
                            model_contract=cache_contract,
                            item=item,
                        )
                    except (IdentityEmbeddingCacheError, OSError):
                        # Cache IO must never invalidate a healthy worker
                        # result.
                        disk_write_errors += 1
        if on_progress is not None:
            on_progress(
                min(len(frames), fully_cached_frames + start + len(batch)),
                len(frames),
                usable_count,
            )
    if retried_batch_count:
        result.diagnostics["retriedBatchCount"] = retried_batch_count
    if crop_store_miss_count:
        result.diagnostics["cropStoreMissCount"] = crop_store_miss_count
    if cache_contract is not None:
        result.diagnostics["diskCacheHitCount"] = disk_hits
        result.diagnostics["diskCacheMissCount"] = disk_misses
        if disk_write_errors:
            result.diagnostics["diskCacheWriteErrorCount"] = disk_write_errors
    usable_fingerprints = [
        str(item["evidenceFingerprint"])
        for item in result.items_by_observation_id.values()
        if item.get("usable") is True
    ]
    result.diagnostics["uniqueEvidenceFingerprintCount"] = len(
        set(usable_fingerprints)
    )
    result.diagnostics["duplicateEvidenceFingerprintCount"] = (
        len(usable_fingerprints) - len(set(usable_fingerprints))
    )
    return result
