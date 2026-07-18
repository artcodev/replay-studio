from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


SERVICE_NAME = "replay-studio-jersey-ocr-worker"
CONTRACT_VERSION = "jersey-ocr.v1"
MAX_JERSEY_DIGITS = 2
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"


def capabilities() -> dict[str, Any]:
    return {
        "digitsOnly": True,
        "maxDigits": MAX_JERSEY_DIGITS,
        "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
        "inputScopes": ["crop", "tracklet"],
    }


class JerseyRequestError(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def decode_image(
    data: bytes,
    file_index: int,
    *,
    max_crop_bytes: int,
    max_crop_pixels: int,
) -> np.ndarray:
    if not data:
        raise JerseyRequestError(422, f"Crop fileIndex={file_index} is empty")
    if len(data) > max_crop_bytes:
        raise JerseyRequestError(
            413, f"Crop fileIndex={file_index} exceeds the byte limit"
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_crop_pixels:
                raise JerseyRequestError(
                    413, f"Crop fileIndex={file_index} exceeds the pixel limit"
                )
            return np.asarray(source.convert("RGB")).copy()
    except JerseyRequestError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise JerseyRequestError(
            422, f"Crop fileIndex={file_index} is not a readable image"
        ) from exc


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JerseyRequestError(422, f"{label} must be a non-empty string")
    return value.strip()


def parse_manifest(
    raw: str,
    crop_count: int,
    *,
    max_batch_size: int,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JerseyRequestError(422, "manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise JerseyRequestError(422, "manifest must be an object")
    if value.get("contractVersion", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise JerseyRequestError(422, "Unsupported manifest contractVersion")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise JerseyRequestError(422, "manifest.items must be a non-empty array")
    if len(items) > max_batch_size:
        raise JerseyRequestError(
            413, "OCR batch exceeds the configured item limit"
        )
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for offset, item in enumerate(items):
        label = f"manifest.items[{offset}]"
        if not isinstance(item, dict):
            raise JerseyRequestError(422, f"{label} must be an object")
        crop_id = _optional_string(item.get("cropId"), f"{label}.cropId")
        assert crop_id is not None
        if crop_id in seen:
            raise JerseyRequestError(422, f"Duplicate cropId: {crop_id}")
        seen.add(crop_id)
        file_index = item.get("fileIndex")
        if (
            isinstance(file_index, bool)
            or not isinstance(file_index, int)
            or not 0 <= file_index < crop_count
        ):
            raise JerseyRequestError(422, f"{label}.fileIndex is invalid")
        frame_index = item.get("frameIndex")
        if frame_index is not None and (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise JerseyRequestError(422, f"{label}.frameIndex is invalid")
        timestamp = item.get("timestamp")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            raise JerseyRequestError(422, f"{label}.timestamp is invalid")
        parsed.append(
            {
                "cropId": crop_id,
                "fileIndex": file_index,
                "observationId": _optional_string(
                    item.get("observationId"), f"{label}.observationId"
                ),
                "trackletId": _optional_string(
                    item.get("trackletId"), f"{label}.trackletId"
                ),
                "frameIndex": frame_index,
                "timestamp": float(timestamp) if timestamp is not None else None,
            }
        )
    return parsed

