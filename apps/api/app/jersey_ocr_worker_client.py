from __future__ import annotations

"""Batch orchestration for the optional provider-neutral jersey OCR worker."""

import json
from typing import Callable, Sequence

from .config import get_settings
from .jersey_ocr_worker_contract import (
    CONTRACT_VERSION,
    JerseyCropRequest,
    JerseyOcrBatchResult,
    JerseyOcrWorkerError,
)
from .jersey_ocr_worker_batch_validation import validate_analysis_payload
from .jersey_ocr_worker_model_contract import validate_readiness_payload
from .jersey_ocr_worker_transport import (
    JerseyOcrTransportError,
    fetch_readiness,
    post_analysis_batch,
)


def jersey_ocr_worker_readiness(*, timeout: float = 2.0) -> dict:
    """Report OCR availability without making the main API unhealthy."""

    settings = get_settings()
    if not settings.jersey_ocr_worker_url:
        return {"configured": False, "status": "disabled", "backend": None}
    try:
        payload = fetch_readiness(
            settings.jersey_ocr_worker_url,
            timeout=timeout,
        )
    except JerseyOcrTransportError as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": None,
            "detail": str(exc),
        }
    try:
        value = validate_readiness_payload(payload)
    except JerseyOcrWorkerError:
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": (
                payload.get("backend") if isinstance(payload, dict) else None
            ),
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": value["backend"],
        "providerVersion": value.get("providerVersion"),
        "modelVersion": value["modelVersion"],
        "device": value.get("device"),
        "batchSize": value.get("batchSize"),
        "modelLoadSeconds": value.get("modelLoadSeconds"),
        "contractVersion": CONTRACT_VERSION,
        "inferenceScope": value.get("inferenceScope", "crop"),
    }


def _validate_crop_requests(crops: Sequence[JerseyCropRequest]) -> None:
    requested_ids: set[str] = set()
    for crop in crops:
        if not crop.crop_id:
            raise JerseyOcrWorkerError("Every OCR crop requires crop_id")
        if crop.crop_id in requested_ids:
            raise JerseyOcrWorkerError(f"Duplicate OCR crop: {crop.crop_id}")
        if not crop.path.is_file():
            raise JerseyOcrWorkerError(f"OCR crop is missing: {crop.path}")
        requested_ids.add(crop.crop_id)


def _batch_payload(
    batch: Sequence[JerseyCropRequest],
) -> tuple[list[tuple[str, tuple[str, bytes, str]]], str]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    manifest_items: list[dict] = []
    for file_index, crop in enumerate(batch):
        try:
            content = crop.path.read_bytes()
        except OSError as exc:
            raise JerseyOcrWorkerError(
                f"OCR crop could not be read: {crop.path}"
            ) from exc
        files.append(("crops", (crop.path.name, content, "image/jpeg")))
        item = {"cropId": crop.crop_id, "fileIndex": file_index}
        for key, value in (
            ("observationId", crop.observation_id),
            ("trackletId", crop.tracklet_id),
            ("frameIndex", crop.frame_index),
            ("timestamp", crop.timestamp),
        ):
            if value is not None:
                item[key] = value
        manifest_items.append(item)
    manifest = json.dumps(
        {"contractVersion": CONTRACT_VERSION, "items": manifest_items},
        separators=(",", ":"),
    )
    return files, manifest


def _merge_diagnostics(target: dict, values: dict) -> None:
    for field, value in values.items():
        if field == "cacheEnabled":
            target[field] = value
        else:
            target[field] = int(target.get(field, 0)) + int(value)


def analyze_jersey_crops(
    crops: Sequence[JerseyCropRequest],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> JerseyOcrBatchResult:
    """Analyze crops while preserving their immutable observation identity."""

    settings = get_settings()
    if not settings.jersey_ocr_worker_url or not crops:
        return JerseyOcrBatchResult()
    _validate_crop_requests(crops)
    batch_size = max(1, int(settings.jersey_ocr_worker_batch_size))
    effective_timeout = max(
        0.1,
        min(float(timeout), float(settings.jersey_ocr_worker_timeout))
        if timeout is not None
        else float(settings.jersey_ocr_worker_timeout),
    )
    result = JerseyOcrBatchResult()
    recognized_count = 0
    accepted_model_contract: dict[str, object] | None = None
    for start in range(0, len(crops), batch_size):
        batch = list(crops[start : start + batch_size])
        files, manifest = _batch_payload(batch)
        try:
            payload = post_analysis_batch(
                settings.jersey_ocr_worker_url,
                files=files,
                manifest=manifest,
                timeout=effective_timeout,
            )
        except JerseyOcrTransportError as exc:
            raise JerseyOcrWorkerError(
                f"Jersey OCR worker failed: {exc}"
            ) from exc
        batch_contract, items, diagnostics = validate_analysis_payload(
            payload,
            {crop.crop_id for crop in batch},
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
            raise JerseyOcrWorkerError(
                "Jersey OCR worker changed model contract between batches: "
                + ", ".join(changed_fields)
            )
        _merge_diagnostics(result.diagnostics, diagnostics)
        for crop_id, item in items.items():
            result.items_by_crop_id[crop_id] = {
                **item,
                "provider": batch_contract["backend"],
                "modelVersion": batch_contract["modelVersion"],
            }
            recognized_count += int(item["status"] == "recognized")
        if on_progress is not None:
            on_progress(
                min(len(crops), start + len(batch)),
                len(crops),
                recognized_count,
            )
    usable_fingerprints = [
        str(item["evidenceFingerprint"])
        for item in result.items_by_crop_id.values()
        if item.get("usable") is True
    ]
    result.diagnostics["uniqueEvidenceFingerprintCount"] = len(
        set(usable_fingerprints)
    )
    result.diagnostics["duplicateEvidenceFingerprintCount"] = (
        len(usable_fingerprints) - len(set(usable_fingerprints))
    )
    return result
