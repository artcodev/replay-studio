from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .ball_detection_cache_contract import (
    BALL_DETECTION_CACHE_SCHEMA_VERSION,
    BallDetectionCacheError,
    canonical_json,
    json_value,
)


@dataclass(frozen=True, slots=True)
class BallDetectionCacheEntry:
    cache_key: str
    path: Path
    primary_backend: str
    frames: tuple[dict[str, Any], ...]

    def as_pipeline_data(
        self,
    ) -> tuple[list[tuple[list[dict[str, Any]], float]], list[dict[str, Any]]]:
        resolved = [
            (deepcopy(frame["detections"]), float(frame["t"]))
            for frame in self.frames
        ]
        batches = [deepcopy(frame["batch"]) for frame in self.frames]
        return resolved, batches


def frame_payload(
    resolved_frames: Sequence[tuple[Sequence[Mapping[str, Any]], float]],
    batches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not resolved_frames:
        raise BallDetectionCacheError("at least one resolved frame is required")
    if len(resolved_frames) != len(batches):
        raise BallDetectionCacheError(
            "resolved_frames and batches must contain the same number of frames"
        )
    frames: list[dict[str, Any]] = []
    for frame_index, ((detections, timestamp), batch) in enumerate(
        zip(resolved_frames, batches, strict=True)
    ):
        try:
            time_value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise BallDetectionCacheError(
                f"resolved frame {frame_index} has an invalid timestamp"
            ) from exc
        frames.append(
            json_value(
                {
                    "frameIndex": frame_index,
                    "t": time_value,
                    "detections": [dict(item) for item in detections],
                    "batch": dict(batch),
                },
                label=f"resolved frame {frame_index}",
            )
        )
    return frames


def is_clean_primary_payload(
    payload: Mapping[str, Any],
    *,
    failed_frame_count: int = 0,
    fallback_frame_count: int = 0,
) -> bool:
    if failed_frame_count != 0 or fallback_frame_count != 0:
        return False
    primary_backend = str(payload.get("primaryBackend") or "")
    frames = payload.get("frames")
    if not primary_backend or not isinstance(frames, list) or not frames:
        return False
    if payload.get("frameCount") != len(frames):
        return False
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frameIndex") != expected_index:
            return False
        if not isinstance(frame.get("detections"), list):
            return False
        batch = frame.get("batch")
        if not isinstance(batch, Mapping):
            return False
        if batch.get("frameIndex") != expected_index:
            return False
        if str(batch.get("backend") or "") != primary_backend:
            return False
        if batch.get("fallbackReason") not in (None, ""):
            return False
        metadata = batch.get("metadata")
        if isinstance(metadata, Mapping) and (
            metadata.get("fallback") is True or metadata.get("fallbackReason")
        ):
            return False
        for detection in frame["detections"]:
            if not isinstance(detection, Mapping):
                return False
            detector_backend = detection.get("detectorBackend")
            if detector_backend is not None and str(detector_backend) != primary_backend:
                return False
            provenance = detection.get("provenance")
            if isinstance(provenance, Mapping):
                provenance_backend = provenance.get("backend")
                if (
                    provenance_backend is not None
                    and str(provenance_backend) != primary_backend
                ):
                    return False
    return True


def entry_from_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_key: str,
    path: Path,
) -> BallDetectionCacheEntry | None:
    if envelope.get("schemaVersion") != BALL_DETECTION_CACHE_SCHEMA_VERSION:
        return None
    if envelope.get("cacheKey") != expected_key:
        return None
    if envelope.get("contract") != expected_contract:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return None
    try:
        payload_fingerprint = sha256(canonical_json(payload).encode()).hexdigest()
    except (TypeError, ValueError):
        return None
    if envelope.get("payloadFingerprint") != payload_fingerprint:
        return None
    if not is_clean_primary_payload(payload):
        return None
    frames = payload.get("frames")
    assert isinstance(frames, list)
    return BallDetectionCacheEntry(
        cache_key=expected_key,
        path=path,
        primary_backend=str(payload["primaryBackend"]),
        frames=tuple(deepcopy(frames)),
    )
