from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


SERVICE_NAME = "replay-studio-ball-worker"
CONTRACT_VERSION = 1


class BallRequestError(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BallRequestError(422, f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise BallRequestError(422, f"{label} must not exceed {maximum}")
    return value


def decode_image(
    data: bytes,
    *,
    label: str,
    max_bytes: int,
    max_pixels: int,
) -> np.ndarray:
    if not data:
        raise BallRequestError(422, f"{label} is empty")
    if len(data) > max_bytes:
        raise BallRequestError(
            413, f"{label} exceeds WASB_MAX_FRAME_BYTES={max_bytes}"
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise BallRequestError(
                    413, f"{label} exceeds WASB_MAX_FRAME_PIXELS={max_pixels}"
                )
            return np.asarray(source.convert("RGB")).copy()
    except BallRequestError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BallRequestError(422, f"{label} is not a readable image") from exc


def parse_manifest(
    raw: str,
    frame_count: int,
    max_candidates_default: int,
) -> tuple[list[dict[str, Any]], int, int | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BallRequestError(422, "manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise BallRequestError(422, "manifest must be an object")
    unknown_fields = set(value) - {
        "contractVersion",
        "maxCandidates",
        "targetIndex",
        "frames",
    }
    if unknown_fields:
        raise BallRequestError(
            422,
            f"manifest contains unsupported fields: {', '.join(sorted(unknown_fields))}",
        )
    if value.get("contractVersion") != CONTRACT_VERSION:
        raise BallRequestError(422, "Unsupported manifest contractVersion")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise BallRequestError(422, "manifest.frames must be a non-empty array")
    target_index = value.get("targetIndex")
    if target_index is not None and (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or not 0 <= target_index < len(frames)
    ):
        raise BallRequestError(422, "manifest.targetIndex is invalid")
    max_candidates = positive_int(
        value.get("maxCandidates", max_candidates_default),
        "manifest.maxCandidates",
        maximum=100,
    )
    parsed: list[dict[str, Any]] = []
    for offset, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise BallRequestError(
                422, f"manifest.frames[{offset}] must be an object"
            )
        unknown_frame_fields = set(frame) - {
            "fileIndex",
            "frameIndex",
            "timestamp",
            "timestampMs",
        }
        if unknown_frame_fields:
            raise BallRequestError(
                422,
                "manifest.frames"
                f"[{offset}] contains unsupported fields: "
                f"{', '.join(sorted(unknown_frame_fields))}",
            )
        file_index = frame.get("fileIndex")
        if (
            isinstance(file_index, bool)
            or not isinstance(file_index, int)
            or not 0 <= file_index < frame_count
        ):
            raise BallRequestError(
                422, f"manifest.frames[{offset}].fileIndex is invalid"
            )
        frame_index = frame.get("frameIndex", offset)
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise BallRequestError(
                422, f"manifest.frames[{offset}].frameIndex is invalid"
            )
        timestamp = frame.get("timestamp")
        timestamp_ms = frame.get("timestampMs")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            raise BallRequestError(
                422, f"manifest.frames[{offset}].timestamp is invalid"
            )
        if timestamp_ms is not None and (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
        ):
            raise BallRequestError(
                422, f"manifest.frames[{offset}].timestampMs is invalid"
            )
        parsed.append(
            {
                "fileIndex": file_index,
                "frameIndex": frame_index,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "timestampMs": (
                    float(timestamp_ms) if timestamp_ms is not None else None
                ),
            }
        )
    return parsed, max_candidates, target_index
