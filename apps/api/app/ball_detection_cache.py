"""Atomic cache for clean, image-space ball detector output.

The dense-frame cache avoids decoding the source video more than once, but the
detector is orders of magnitude more expensive than JPEG decoding.  This
module stores the post-NMS *image-space candidates* so later reconstructions
can still rerun temporal resolution and field projection after calibration or
manual edits.

Only complete primary-detector runs are publishable.  Fallback or partially
failed runs remain useful reconstruction evidence, but caching them would make
a transient worker/model outage sticky across later rebuilds.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


BALL_DETECTION_CACHE_SCHEMA_VERSION = 1


class BallDetectionCacheError(RuntimeError):
    """Raised when a clean cache artifact cannot be serialized or published."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: Any, *, label: str) -> Any:
    """Return a detached JSON value and reject lossy/non-finite inputs."""

    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise BallDetectionCacheError(f"{label} must be finite JSON data") from exc


def ball_detection_input_fingerprint(detector_input: Mapping[str, Any]) -> str:
    """Fingerprint every detector/checkpoint field supplied by reconstruction."""

    normalized = _json_value(dict(detector_input), label="detector_input")
    return sha256(_canonical_json(normalized).encode()).hexdigest()


def build_ball_detection_cache_contract(
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete cache identity used for both lookup and validation."""

    dense_key = str(dense_cache_key).strip()
    if not dense_key:
        raise BallDetectionCacheError("dense_cache_key must not be empty")
    normalized_input = _json_value(dict(detector_input), label="detector_input")
    return {
        "schemaVersion": BALL_DETECTION_CACHE_SCHEMA_VERSION,
        "denseCacheKey": dense_key,
        "detectorInputFingerprint": sha256(
            _canonical_json(normalized_input).encode()
        ).hexdigest(),
        # Keeping the full input makes cache provenance auditable and prevents
        # an opaque digest from hiding a missing checkpoint/config field.
        "detectorInput": normalized_input,
    }


def ball_detection_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = _json_value(dict(contract), label="cache contract")
    return sha256(_canonical_json(normalized).encode()).hexdigest()


def ball_detection_cache_path(
    asset_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    return (
        Path(asset_directory).expanduser().resolve()
        / "ball-detections"
        / f"{ball_detection_cache_key(contract)}.json"
    )


@dataclass(frozen=True, slots=True)
class BallDetectionCacheEntry:
    """A validated clean-primary cache artifact."""

    cache_key: str
    path: Path
    primary_backend: str
    frames: tuple[dict[str, Any], ...]

    def as_pipeline_data(
        self,
    ) -> tuple[list[tuple[list[dict[str, Any]], float]], list[dict[str, Any]]]:
        """Return fresh mutable values accepted by ``_detect_ball_frames``."""

        resolved = [
            (deepcopy(frame["detections"]), float(frame["t"]))
            for frame in self.frames
        ]
        batches = [deepcopy(frame["batch"]) for frame in self.frames]
        return resolved, batches


def _frame_payload(
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
            _json_value(
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


def _is_clean_primary_payload(
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
        if not isinstance(frame, Mapping):
            return False
        if frame.get("frameIndex") != expected_index:
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


def _entry_from_envelope(
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
    if envelope.get("payloadFingerprint") != sha256(
        _canonical_json(payload).encode()
    ).hexdigest():
        return None
    if not _is_clean_primary_payload(payload):
        return None
    frames = payload.get("frames")
    assert isinstance(frames, list)  # established by the clean validator
    return BallDetectionCacheEntry(
        cache_key=expected_key,
        path=path,
        primary_backend=str(payload["primaryBackend"]),
        frames=tuple(deepcopy(frames)),
    )


def load_ball_detection_cache(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
) -> BallDetectionCacheEntry | None:
    """Return a validated cache hit, or ``None`` for absent/corrupt/stale data."""

    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key,
        detector_input=detector_input,
    )
    key = ball_detection_cache_key(contract)
    path = ball_detection_cache_path(asset_directory, contract)
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, Mapping):
        return None
    return _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )


def store_clean_ball_detection_cache(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
    primary_backend: str,
    resolved_frames: Sequence[tuple[Sequence[Mapping[str, Any]], float]],
    batches: Sequence[Mapping[str, Any]],
    failed_frame_count: int = 0,
    fallback_frame_count: int = 0,
) -> BallDetectionCacheEntry | None:
    """Atomically publish clean primary output; return ``None`` when degraded."""

    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key,
        detector_input=detector_input,
    )
    key = ball_detection_cache_key(contract)
    frames = _frame_payload(resolved_frames, batches)
    payload = {
        "primaryBackend": str(primary_backend).strip(),
        "frameCount": len(frames),
        "frames": frames,
    }
    if not _is_clean_primary_payload(
        payload,
        failed_frame_count=int(failed_frame_count),
        fallback_frame_count=int(fallback_frame_count),
    ):
        return None

    normalized_payload = _json_value(payload, label="cache payload")
    envelope = {
        "schemaVersion": BALL_DETECTION_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            _canonical_json(normalized_payload).encode()
        ).hexdigest(),
        "payload": normalized_payload,
    }
    path = ball_detection_cache_path(asset_directory, contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise BallDetectionCacheError(
            f"could not publish ball detection cache {path}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:  # pragma: no cover - defensive invariant
        raise BallDetectionCacheError("published ball detection cache failed validation")
    return entry
