"""Versioned atomic disk cache for per-crop jersey OCR worker results.

The OCR worker's in-memory cache dies with the container, so warm rebuilds
used to re-upload and re-read every shirt-number crop. This cache stores the
immutable per-crop provider item — recognized, ambiguous, rejected or
no-number, all are valid evidence — keyed by the exact crop bytes and the
worker's complete model contract. Missing, corrupt or tampered artifacts are
ordinary misses; cache IO never invalidates a healthy worker result.
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


JERSEY_OCR_CACHE_SCHEMA_VERSION = 1

_CONTRACT_FIELDS = (
    "contractVersion",
    "backend",
    "providerVersion",
    "modelVersion",
    "inferenceScope",
    "digitsOnly",
    "maxDigits",
    "evidenceFingerprintVersion",
)


class JerseyOcrCacheError(RuntimeError):
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
        raise JerseyOcrCacheError(f"{label} must be finite JSON data") from exc


def build_jersey_ocr_cache_contract(
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    crop_digest = str(crop_sha256).strip().lower()
    if len(crop_digest) != 64 or any(
        character not in "0123456789abcdef" for character in crop_digest
    ):
        raise JerseyOcrCacheError("crop_sha256 must be a SHA-256 hex digest")
    missing = [
        field
        for field in _CONTRACT_FIELDS
        if field != "providerVersion" and model_contract.get(field) in (None, "")
    ]
    if missing:
        raise JerseyOcrCacheError(
            f"model_contract is missing: {', '.join(missing)}"
        )
    return {
        "schemaVersion": JERSEY_OCR_CACHE_SCHEMA_VERSION,
        "cropContentSha256": crop_digest,
        "modelContract": {
            field: model_contract.get(field) for field in _CONTRACT_FIELDS
        },
    }


def jersey_ocr_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = _json_value(dict(contract), label="cache contract")
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def jersey_ocr_cache_path(
    cache_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = jersey_ocr_cache_key(contract)
    return (
        Path(cache_directory).expanduser().resolve()
        / key[:2]
        / f"{key}.json"
    )


@dataclass(frozen=True, slots=True)
class JerseyOcrCacheEntry:
    cache_key: str
    path: Path
    crop_sha256: str
    # The exact per-crop worker item regardless of recognition status.
    item: dict[str, Any]

    def detached_item(self) -> dict[str, Any]:
        return deepcopy(self.item)


@dataclass(frozen=True, slots=True)
class JerseyOcrCacheLookup:
    entry: JerseyOcrCacheEntry | None
    status: str
    error: str | None = None


def _valid_payload(payload: Mapping[str, Any]) -> bool:
    item = payload.get("item")
    if not isinstance(item, Mapping):
        return False
    if not isinstance(item.get("usable"), bool):
        return False
    status = item.get("status")
    if not isinstance(status, str) or not status:
        return False
    fingerprint = item.get("evidenceFingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    return True


def _entry_from_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_key: str,
    path: Path,
) -> JerseyOcrCacheEntry | None:
    if envelope.get("schemaVersion") != JERSEY_OCR_CACHE_SCHEMA_VERSION:
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
    return JerseyOcrCacheEntry(
        cache_key=expected_key,
        path=path,
        crop_sha256=str(expected_contract["cropContentSha256"]),
        item=deepcopy(payload["item"]),
    )


def lookup_jersey_ocr_cache(
    cache_directory: str | Path,
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
) -> JerseyOcrCacheLookup:
    """Return hit/absent/corrupt/error without making cache faults fatal."""

    contract = build_jersey_ocr_cache_contract(
        crop_sha256=crop_sha256,
        model_contract=model_contract,
    )
    key = jersey_ocr_cache_key(contract)
    path = jersey_ocr_cache_path(cache_directory, contract)
    if not path.exists():
        return JerseyOcrCacheLookup(None, "absent")
    if not path.is_file():
        return JerseyOcrCacheLookup(None, "error", "cache path is not a file")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return JerseyOcrCacheLookup(None, "corrupt", str(exc))
    except OSError as exc:
        return JerseyOcrCacheLookup(None, "error", str(exc))
    if not isinstance(envelope, Mapping):
        return JerseyOcrCacheLookup(None, "corrupt", "envelope is not an object")
    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:
        return JerseyOcrCacheLookup(
            None,
            "corrupt",
            "contract, checksum or payload validation failed",
        )
    return JerseyOcrCacheLookup(entry, "hit")


def store_jersey_ocr_cache(
    cache_directory: str | Path,
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
    item: Mapping[str, Any],
) -> JerseyOcrCacheEntry:
    """Atomically publish one validated per-crop worker item."""

    contract = build_jersey_ocr_cache_contract(
        crop_sha256=crop_sha256,
        model_contract=model_contract,
    )
    key = jersey_ocr_cache_key(contract)
    payload = _json_value({"item": dict(item)}, label="cache payload")
    if not _valid_payload(payload):
        raise JerseyOcrCacheError("jersey OCR item is invalid")
    envelope = {
        "schemaVersion": JERSEY_OCR_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "payload": payload,
    }
    path = jersey_ocr_cache_path(cache_directory, contract)
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
        raise JerseyOcrCacheError(
            f"could not publish jersey OCR cache {path}: {exc}"
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
        raise JerseyOcrCacheError(
            "published jersey OCR cache failed validation"
        )
    return entry


__all__ = (
    "JERSEY_OCR_CACHE_SCHEMA_VERSION",
    "JerseyOcrCacheEntry",
    "JerseyOcrCacheError",
    "JerseyOcrCacheLookup",
    "build_jersey_ocr_cache_contract",
    "jersey_ocr_cache_key",
    "jersey_ocr_cache_path",
    "lookup_jersey_ocr_cache",
    "store_jersey_ocr_cache",
)
