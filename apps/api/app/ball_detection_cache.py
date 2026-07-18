"""Atomic filesystem storage for validated dense-ball detector artifacts."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from .ball_detection_cache_codec import (
    BallDetectionCacheEntry,
    entry_from_envelope,
    frame_payload,
    is_clean_primary_payload,
)
from .ball_detection_cache_contract import (
    BALL_DETECTION_CACHE_SCHEMA_VERSION,
    BallDetectionCacheError,
    ball_detection_cache_key,
    build_ball_detection_cache_contract,
    canonical_json,
    json_value,
)


def ball_detection_cache_path(
    asset_directory: str | Path, contract: Mapping[str, Any]
) -> Path:
    return (
        Path(asset_directory).expanduser().resolve()
        / "ball-detections"
        / f"{ball_detection_cache_key(contract)}.json"
    )


def ball_detection_checkpoint_path(
    asset_directory: str | Path, contract: Mapping[str, Any]
) -> Path:
    return (
        Path(asset_directory).expanduser().resolve()
        / "ball-detections"
        / f"{ball_detection_cache_key(contract)}.partial.json"
    )


def ball_detection_cache_lock_path(
    asset_directory: str | Path, contract: Mapping[str, Any]
) -> Path:
    return (
        Path(asset_directory).expanduser().resolve()
        / "ball-detections"
        / f"{ball_detection_cache_key(contract)}.lock"
    )


@contextmanager
def _contract_lock(path: Path) -> Iterator[None]:
    """Serialize complete-cache and partial-checkpoint publication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("a+b")
    except OSError as exc:
        raise BallDetectionCacheError(
            f"could not open ball detection cache lock {path}: {exc}"
        ) from exc
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            acquired = True
        except OSError as exc:
            raise BallDetectionCacheError(
                f"could not acquire ball detection cache lock {path}: {exc}"
            ) from exc
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def load_ball_detection_cache(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
) -> BallDetectionCacheEntry | None:
    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key, detector_input=detector_input
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
    return entry_from_envelope(
        envelope, expected_contract=contract, expected_key=key, path=path
    )


def load_ball_detection_checkpoint(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
    expected_frame_count: int,
) -> BallDetectionCacheEntry | None:
    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key, detector_input=detector_input
    )
    key = ball_detection_cache_key(contract)
    path = ball_detection_checkpoint_path(asset_directory, contract)
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, Mapping) or envelope.get("state") != "partial":
        return None
    try:
        stored_expected_count = int(envelope.get("expectedFrameCount") or 0)
    except (TypeError, ValueError):
        return None
    if stored_expected_count != int(expected_frame_count):
        return None
    entry = entry_from_envelope(
        envelope, expected_contract=contract, expected_key=key, path=path
    )
    if entry is None or len(entry.frames) >= int(expected_frame_count):
        return None
    return entry


def _atomic_write(path: Path, envelope: Mapping[str, Any], *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise BallDetectionCacheError(f"could not publish {label} {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def store_ball_detection_checkpoint(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
    primary_backend: str,
    resolved_frames: Sequence[tuple[Sequence[Mapping[str, Any]], float]],
    batches: Sequence[Mapping[str, Any]],
    expected_frame_count: int,
) -> BallDetectionCacheEntry:
    expected_count = int(expected_frame_count)
    if expected_count <= 0 or len(resolved_frames) >= expected_count:
        raise BallDetectionCacheError(
            "a checkpoint must be a non-empty prefix shorter than the full run"
        )
    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key, detector_input=detector_input
    )
    key = ball_detection_cache_key(contract)
    frames = frame_payload(resolved_frames, batches)
    payload = {
        "primaryBackend": str(primary_backend).strip(),
        "frameCount": len(frames),
        "frames": frames,
    }
    if not is_clean_primary_payload(payload):
        raise BallDetectionCacheError("checkpoint contains degraded detector output")
    normalized_payload = json_value(payload, label="checkpoint payload")
    envelope = {
        "schemaVersion": BALL_DETECTION_CACHE_SCHEMA_VERSION,
        "state": "partial",
        "expectedFrameCount": expected_count,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            canonical_json(normalized_payload).encode()
        ).hexdigest(),
        "payload": normalized_payload,
    }
    path = ball_detection_checkpoint_path(asset_directory, contract)
    with _contract_lock(ball_detection_cache_lock_path(asset_directory, contract)):
        complete = load_ball_detection_cache(
            asset_directory,
            dense_cache_key=dense_cache_key,
            detector_input=detector_input,
        )
        if complete is not None:
            return complete
        existing = load_ball_detection_checkpoint(
            asset_directory,
            dense_cache_key=dense_cache_key,
            detector_input=detector_input,
            expected_frame_count=expected_count,
        )
        if existing is not None and len(existing.frames) >= len(frames):
            return existing
        _atomic_write(path, envelope, label="ball detection checkpoint")
    entry = entry_from_envelope(
        envelope, expected_contract=contract, expected_key=key, path=path
    )
    if entry is None:  # pragma: no cover
        raise BallDetectionCacheError("published checkpoint failed validation")
    return entry


def delete_ball_detection_checkpoint(
    asset_directory: str | Path,
    *,
    dense_cache_key: str,
    detector_input: Mapping[str, Any],
) -> None:
    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key, detector_input=detector_input
    )
    path = ball_detection_checkpoint_path(asset_directory, contract)
    with _contract_lock(ball_detection_cache_lock_path(asset_directory, contract)):
        path.unlink(missing_ok=True)


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
    contract = build_ball_detection_cache_contract(
        dense_cache_key=dense_cache_key, detector_input=detector_input
    )
    key = ball_detection_cache_key(contract)
    frames = frame_payload(resolved_frames, batches)
    payload = {
        "primaryBackend": str(primary_backend).strip(),
        "frameCount": len(frames),
        "frames": frames,
    }
    if not is_clean_primary_payload(
        payload,
        failed_frame_count=int(failed_frame_count),
        fallback_frame_count=int(fallback_frame_count),
    ):
        return None
    normalized_payload = json_value(payload, label="cache payload")
    envelope = {
        "schemaVersion": BALL_DETECTION_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            canonical_json(normalized_payload).encode()
        ).hexdigest(),
        "payload": normalized_payload,
    }
    path = ball_detection_cache_path(asset_directory, contract)
    checkpoint_path = ball_detection_checkpoint_path(asset_directory, contract)
    with _contract_lock(ball_detection_cache_lock_path(asset_directory, contract)):
        _atomic_write(path, envelope, label="ball detection cache")
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass
    entry = entry_from_envelope(
        envelope, expected_contract=contract, expected_key=key, path=path
    )
    if entry is None:  # pragma: no cover
        raise BallDetectionCacheError("published ball detection cache failed validation")
    return entry
