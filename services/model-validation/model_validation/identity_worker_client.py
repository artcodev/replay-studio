"""HTTP contract translator for the identity worker."""

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
    "dimension",
    "normalized",
    "device",
    "batchSize",
    "modelVersion",
    "checkpointSha256",
    "hrnetCheckpointSha256",
    "soccerNetCommit",
)


def fetch_identity_predictions(
    client: Any,
    base_url: str,
    manifest: ValidationManifest,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ready = require_ready(client, base_url, "identity-worker")
    provider = {key: ready.get(key) for key in PROVIDER_KEYS}
    predictions: dict[str, dict[str, Any]] = {}
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    diagnostics: list[dict[str, Any]] = []
    for batch in batches(manifest.crops, batch_size):
        files = []
        frames = []
        for index, crop in enumerate(batch):
            data = crop.path.read_bytes()
            width, height = image_size(data, crop.crop_id)
            files.append(("frames", (crop.path.name, data, mime_type(crop.path))))
            frames.append(
                {
                    "fileIndex": index,
                    "frameIndex": index,
                    "observations": [
                        {
                            "observationId": crop.crop_id,
                            "bbox": {
                                "x": 0,
                                "y": 0,
                                "width": width,
                                "height": height,
                            },
                        }
                    ],
                }
            )
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/v1/embeddings",
                files=files,
                data={"manifest": json.dumps({"frames": frames})},
            )
        except Exception as exc:
            raise WorkerUnavailable(f"identity-worker inference request failed: {exc}") from exc
        value = require_success(response, "identity-worker")
        if not isinstance(value.get("items"), list):
            raise WorkerProtocolError("identity-worker response.items is invalid")
        response_provider = {key: value.get(key) for key in PROVIDER_KEYS}
        if response_provider != provider:
            raise WorkerProtocolError("identity-worker provider provenance changed during the run")
        diagnostics.append(
            value.get("diagnostics")
            if isinstance(value.get("diagnostics"), dict)
            else {}
        )
        for item in value["items"]:
            crop_id = item.get("observationId") if isinstance(item, dict) else None
            if (
                not isinstance(crop_id, str)
                or crop_id not in expected_crop_ids
                or crop_id in predictions
            ):
                raise WorkerProtocolError("identity-worker returned a missing or duplicate observationId")
            predictions[crop_id] = item
    return provider, predictions, diagnostics
