"""Versioned atomic disk cache for PnLCalib anchor-frame worker results.

The worker's in-process cache dies with the container (TTL one hour), so warm
rebuilds used to re-upload and re-infer every anchor frame. This cache stores
the immutable per-frame provider result — never a decoded ``PitchCalibration``
decision — keyed by the exact frame bytes and the worker's model identity.
"no-solution" outcomes are cached explicitly so uncalibratable frames are not
re-asked forever. Missing, corrupt or tampered artifacts are ordinary misses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


CALIBRATION_ANCHOR_CACHE_SCHEMA_VERSION = 1


class CalibrationAnchorCacheError(RuntimeError):
    """A cache contract or atomic publication could not be completed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise CalibrationAnchorCacheError(
            f"{label} must be finite JSON data"
        ) from exc


def build_calibration_anchor_cache_contract(
    *,
    frame_sha256: str,
    worker_contract: Mapping[str, Any],
) -> dict[str, Any]:
    frame_digest = str(frame_sha256).strip().lower()
    if len(frame_digest) != 64 or any(
        character not in "0123456789abcdef" for character in frame_digest
    ):
        raise CalibrationAnchorCacheError(
            "frame_sha256 must be a SHA-256 hex digest"
        )
    backend = str(worker_contract.get("backend") or "")
    model_version = str(worker_contract.get("modelVersion") or "")
    if not backend or not model_version:
        raise CalibrationAnchorCacheError(
            "worker_contract requires backend and modelVersion"
        )
    return {
        "schemaVersion": CALIBRATION_ANCHOR_CACHE_SCHEMA_VERSION,
        "frameContentSha256": frame_digest,
        "workerContract": {"backend": backend, "modelVersion": model_version},
    }


def calibration_anchor_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = _json_value(dict(contract), label="cache contract")
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def calibration_anchor_cache_path(
    cache_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = calibration_anchor_cache_key(contract)
    return (
        Path(cache_directory).expanduser().resolve()
        / key[:2]
        / f"{key}.json"
    )


@dataclass(frozen=True, slots=True)
class CalibrationAnchorCacheEntry:
    cache_key: str
    path: Path
    frame_sha256: str
    # The exact per-frame worker result, or None for a cached "no solution".
    worker_item: dict[str, Any] | None

    def detached_item(self) -> dict[str, Any] | None:
        return deepcopy(self.worker_item)


@dataclass(frozen=True, slots=True)
class CalibrationAnchorCacheLookup:
    entry: CalibrationAnchorCacheEntry | None
    status: str
    error: str | None = None


def _valid_payload(payload: Mapping[str, Any]) -> bool:
    status = payload.get("status")
    if status == "no-solution":
        return payload.get("workerItem") is None
    if status != "solved":
        return False
    return isinstance(payload.get("workerItem"), Mapping)


def _entry_from_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_key: str,
    path: Path,
) -> CalibrationAnchorCacheEntry | None:
    if envelope.get("schemaVersion") != CALIBRATION_ANCHOR_CACHE_SCHEMA_VERSION:
        return None
    if envelope.get("cacheKey") != expected_key:
        return None
    if envelope.get("contract") != expected_contract:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return None
    try:
        payload_fingerprint = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return None
    if envelope.get("payloadFingerprint") != payload_fingerprint:
        return None
    if not _valid_payload(payload):
        return None
    worker_item = payload.get("workerItem")
    return CalibrationAnchorCacheEntry(
        cache_key=expected_key,
        path=path,
        frame_sha256=str(expected_contract["frameContentSha256"]),
        worker_item=deepcopy(worker_item) if worker_item is not None else None,
    )


def lookup_calibration_anchor_cache(
    cache_directory: str | Path,
    *,
    frame_sha256: str,
    worker_contract: Mapping[str, Any],
) -> CalibrationAnchorCacheLookup:
    """Return hit/absent/corrupt/error without making cache faults fatal."""

    contract = build_calibration_anchor_cache_contract(
        frame_sha256=frame_sha256,
        worker_contract=worker_contract,
    )
    key = calibration_anchor_cache_key(contract)
    path = calibration_anchor_cache_path(cache_directory, contract)
    if not path.exists():
        return CalibrationAnchorCacheLookup(None, "absent")
    if not path.is_file():
        return CalibrationAnchorCacheLookup(
            None, "error", "cache path is not a file"
        )
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CalibrationAnchorCacheLookup(None, "corrupt", str(exc))
    except OSError as exc:
        return CalibrationAnchorCacheLookup(None, "error", str(exc))
    if not isinstance(envelope, Mapping):
        return CalibrationAnchorCacheLookup(
            None, "corrupt", "envelope is not an object"
        )
    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:
        return CalibrationAnchorCacheLookup(
            None,
            "corrupt",
            "contract, checksum or payload validation failed",
        )
    return CalibrationAnchorCacheLookup(entry, "hit")


def store_calibration_anchor_cache(
    cache_directory: str | Path,
    *,
    frame_sha256: str,
    worker_contract: Mapping[str, Any],
    worker_item: Mapping[str, Any] | None,
) -> CalibrationAnchorCacheEntry:
    """Atomically publish one per-frame worker result or explicit no-solution."""

    contract = build_calibration_anchor_cache_contract(
        frame_sha256=frame_sha256,
        worker_contract=worker_contract,
    )
    key = calibration_anchor_cache_key(contract)
    payload = _json_value(
        {
            "status": "solved" if worker_item is not None else "no-solution",
            "workerItem": dict(worker_item) if worker_item is not None else None,
        },
        label="cache payload",
    )
    if not _valid_payload(payload):
        raise CalibrationAnchorCacheError(
            "calibration anchor payload is invalid"
        )
    envelope = {
        "schemaVersion": CALIBRATION_ANCHOR_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "payload": payload,
    }
    path = calibration_anchor_cache_path(cache_directory, contract)
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
        raise CalibrationAnchorCacheError(
            f"could not publish calibration anchor cache {path}: {exc}"
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
        raise CalibrationAnchorCacheError(
            "published calibration anchor cache failed validation"
        )
    return entry


__all__ = (
    "CALIBRATION_ANCHOR_CACHE_SCHEMA_VERSION",
    "CalibrationAnchorCacheEntry",
    "CalibrationAnchorCacheError",
    "CalibrationAnchorCacheLookup",
    "build_calibration_anchor_cache_contract",
    "calibration_anchor_cache_key",
    "calibration_anchor_cache_path",
    "lookup_calibration_anchor_cache",
    "store_calibration_anchor_cache",
)
