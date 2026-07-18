"""HTTP contract translator for the jersey OCR worker."""

from __future__ import annotations

import json
from typing import Any

from .manifest_contract import ValidationManifest
from .worker_transport import (
    WorkerProtocolError,
    WorkerUnavailable,
    batches,
    image_size,
    mime_type,
    require_ready,
    require_success,
)


PROVIDER_KEYS = (
    "backend",
    "providerVersion",
    "modelVersion",
    "contractVersion",
    "device",
    "batchSize",
    "inferenceScope",
)


def fetch_jersey_predictions(
    client: Any,
    base_url: str,
    manifest: ValidationManifest,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ready = require_ready(client, base_url, "jersey-ocr-worker")
    provider = {key: ready.get(key) for key in PROVIDER_KEYS}
    predictions: dict[str, dict[str, Any]] = {}
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    diagnostics: list[dict[str, Any]] = []
    for batch in batches(manifest.crops, batch_size):
        files = []
        items = []
        for index, crop in enumerate(batch):
            data = crop.path.read_bytes()
            image_size(data, crop.crop_id)
            files.append(("crops", (crop.path.name, data, mime_type(crop.path))))
            items.append(
                {
                    "cropId": crop.crop_id,
                    "fileIndex": index,
                    "observationId": crop.crop_id,
                    "trackletId": crop.person_id,
                    "frameIndex": index,
                    "timestamp": float(index),
                }
            )
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/v1/analyze",
                files=files,
                data={
                    "manifest": json.dumps(
                        {"contractVersion": "jersey-ocr.v1", "items": items}
                    )
                },
            )
        except Exception as exc:
            raise WorkerUnavailable(f"jersey-ocr-worker inference request failed: {exc}") from exc
        value = require_success(response, "jersey-ocr-worker")
        if not isinstance(value.get("items"), list):
            raise WorkerProtocolError("jersey-ocr-worker response.items is invalid")
        response_provider = {key: value.get(key) for key in PROVIDER_KEYS}
        if response_provider != provider:
            raise WorkerProtocolError("jersey-ocr-worker provider provenance changed during the run")
        diagnostics.append(
            value.get("diagnostics")
            if isinstance(value.get("diagnostics"), dict)
            else {}
        )
        for item in value["items"]:
            crop_id = item.get("cropId") if isinstance(item, dict) else None
            if (
                not isinstance(crop_id, str)
                or crop_id not in expected_crop_ids
                or crop_id in predictions
            ):
                raise WorkerProtocolError("jersey-ocr-worker returned a missing or duplicate cropId")
            predictions[crop_id] = item
    return provider, predictions, diagnostics
