from __future__ import annotations

"""Batch orchestration for the optional provider-neutral identity worker."""

import json
from pathlib import Path
from typing import Callable, Sequence

from .config import get_settings
from .identity_worker_batch_validation import validate_embedding_payload
from .identity_worker_contract import (
    IdentityWorkerBatchResult,
    IdentityWorkerError,
)
from .identity_worker_model_contract import validate_readiness_payload
from .identity_worker_transport import (
    IdentityWorkerTransportError,
    fetch_identity_readiness,
    post_identity_batch,
)


IdentityFrameRequest = tuple[int, Path, list[dict]]


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
    for _frame_index, path, observations in frames:
        if not path.is_file():
            raise IdentityWorkerError(f"Identity source frame is missing: {path}")
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
            requested_ids.add(observation_id)


def _batch_payload(
    batch: Sequence[IdentityFrameRequest],
) -> tuple[list[tuple[str, tuple[str, bytes, str]]], str]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    manifest_frames: list[dict] = []
    for file_index, (frame_index, path, observations) in enumerate(batch):
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise IdentityWorkerError(
                f"Identity source frame could not be read: {path}"
            ) from exc
        files.append(("frames", (path.name, content, "image/jpeg")))
        manifest_frames.append(
            {
                "frameIndex": int(frame_index),
                "fileIndex": file_index,
                "observations": observations,
            }
        )
    return files, json.dumps(
        {"frames": manifest_frames},
        separators=(",", ":"),
    )


def _merge_diagnostics(target: dict, values: dict) -> None:
    for field, value in values.items():
        if field == "cache":
            target[field] = value
        else:
            target[field] = int(target.get(field, 0)) + int(value)


def embed_identity_frames(
    frames: Sequence[IdentityFrameRequest],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> IdentityWorkerBatchResult:
    """Embed observation bboxes, batching whole source frames over HTTP."""

    settings = get_settings()
    if not settings.identity_worker_url or not frames:
        return IdentityWorkerBatchResult()
    _validate_frame_requests(frames)
    batch_size = max(1, int(settings.identity_worker_batch_size))
    effective_timeout = max(
        0.1,
        min(float(timeout), float(settings.identity_worker_timeout))
        if timeout is not None
        else float(settings.identity_worker_timeout),
    )
    result = IdentityWorkerBatchResult()
    usable_count = 0
    accepted_model_contract: dict[str, object] | None = None
    for start in range(0, len(frames), batch_size):
        batch = list(frames[start : start + batch_size])
        files, manifest = _batch_payload(batch)
        try:
            payload = post_identity_batch(
                settings.identity_worker_url,
                files=files,
                manifest=manifest,
                timeout=effective_timeout,
            )
        except IdentityWorkerTransportError as exc:
            raise IdentityWorkerError(f"Identity worker failed: {exc}") from exc
        expected_ids = {
            str(observation["observationId"])
            for _frame_index, _path, observations in batch
            for observation in observations
        }
        batch_contract, items, diagnostics = validate_embedding_payload(
            payload,
            expected_ids,
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
        if on_progress is not None:
            on_progress(
                min(len(frames), start + len(batch)),
                len(frames),
                usable_count,
            )
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
