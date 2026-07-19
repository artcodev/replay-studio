from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .ball_detection_contract import GENERIC_FALLBACK_BACKEND
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
    failed_frame_count: int
    fallback_frame_count: int

    @property
    def is_clean(self) -> bool:
        return self.failed_frame_count == 0 and self.fallback_frame_count == 0

    def as_pipeline_data(
        self,
    ) -> tuple[list[tuple[list[dict[str, Any]], float]], list[dict[str, Any]]]:
        resolved = [
            (deepcopy(frame["detections"]), float(frame["t"]))
            for frame in self.frames
        ]
        batches = [deepcopy(frame["batch"]) for frame in self.frames]
        return resolved, batches


def frame_degradation(batch: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return the stored (failed, fallback) markers of one dense-frame batch."""

    failed = str(batch.get("backend") or "") == GENERIC_FALLBACK_BACKEND
    fallback = batch.get("fallbackReason") not in (None, "")
    return failed, fallback


def batch_degradation_counts(
    batches: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Count (failed, fallback) frames from their per-frame batch markers."""

    failed_total = 0
    fallback_total = 0
    for batch in batches:
        failed, fallback = frame_degradation(batch)
        failed_total += int(failed)
        fallback_total += int(fallback)
    return failed_total, fallback_total


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


def validate_ball_detection_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Validate stored frames structurally; degraded frames must carry markers.

    A frame is either explicitly degraded (fallback backend and/or a recorded
    ``fallbackReason``) or it must be verifiably clean primary output end to
    end. Unmarked foreign-backend data is rejected rather than silently
    accepted as primary evidence.
    """

    primary_backend = str(payload.get("primaryBackend") or "")
    frames = payload.get("frames")
    if not primary_backend or not isinstance(frames, list) or not frames:
        return None
    if payload.get("frameCount") != len(frames):
        return None
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frameIndex") != expected_index:
            return None
        if not isinstance(frame.get("detections"), list):
            return None
        batch = frame.get("batch")
        if not isinstance(batch, Mapping):
            return None
        if batch.get("frameIndex") != expected_index:
            return None
        if not str(batch.get("backend") or ""):
            return None
        failed, fallback = frame_degradation(batch)
        if failed or fallback:
            continue
        if str(batch.get("backend") or "") != primary_backend:
            return None
        metadata = batch.get("metadata")
        if isinstance(metadata, Mapping) and (
            metadata.get("fallback") is True or metadata.get("fallbackReason")
        ):
            return None
        for detection in frame["detections"]:
            if not isinstance(detection, Mapping):
                return None
            detector_backend = detection.get("detectorBackend")
            if detector_backend is not None and str(detector_backend) != primary_backend:
                return None
            provenance = detection.get("provenance")
            if isinstance(provenance, Mapping):
                provenance_backend = provenance.get("backend")
                if (
                    provenance_backend is not None
                    and str(provenance_backend) != primary_backend
                ):
                    return None
    return tuple(frames)


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
    frames = validate_ball_detection_payload(payload)
    if frames is None:
        return None
    failed_frame_count, fallback_frame_count = batch_degradation_counts(
        [frame["batch"] for frame in frames]
    )
    return BallDetectionCacheEntry(
        cache_key=expected_key,
        path=path,
        primary_backend=str(payload["primaryBackend"]),
        frames=tuple(deepcopy(frames)),
        failed_frame_count=failed_frame_count,
        fallback_frame_count=fallback_frame_count,
    )
