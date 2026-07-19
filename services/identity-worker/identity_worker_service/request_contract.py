from __future__ import annotations

import json
from typing import Any


REQUEST_CONTRACT_VERSION = 2


class IdentityRequestError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def parse_manifest(raw: str, crop_count: int) -> list[dict[str, Any]]:
    """Validate the v2 flat crop manifest.

    The API cut and QA-gated every crop before upload, so a manifest entry
    references one crop file and echoes the extraction quality evidence.
    """

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IdentityRequestError("manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise IdentityRequestError("manifest must be an object")
    if value.get("contractVersion") != REQUEST_CONTRACT_VERSION:
        raise IdentityRequestError(
            "manifest contractVersion is unsupported; this worker speaks "
            f"v{REQUEST_CONTRACT_VERSION} crop batches"
        )
    crop_items = value.get("crops")
    if not isinstance(crop_items, list) or not crop_items:
        raise IdentityRequestError("manifest.crops must be a non-empty array")
    seen: set[str] = set()
    for item in crop_items:
        if not isinstance(item, dict):
            raise IdentityRequestError("Each manifest crop must be an object")
        file_index = item.get("fileIndex")
        if not isinstance(file_index, int) or not 0 <= file_index < crop_count:
            raise IdentityRequestError("manifest fileIndex is out of range")
        observation_id = item.get("observationId")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise IdentityRequestError("observationId is required")
        if observation_id in seen:
            raise IdentityRequestError(
                f"Duplicate observationId: {observation_id}"
            )
        seen.add(observation_id)
        frame_index = item.get("frameIndex")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise IdentityRequestError(f"Invalid frameIndex for {observation_id}")
        if not isinstance(item.get("quality"), dict):
            raise IdentityRequestError(f"Invalid quality for {observation_id}")
    return crop_items
