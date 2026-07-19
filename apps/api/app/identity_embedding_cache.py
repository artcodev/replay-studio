"""Versioned atomic disk cache for per-observation PRTReID worker results.

The identity worker's in-memory cache dies with the container, so warm
rebuilds used to re-upload and re-embed every crop. This cache stores the
immutable per-observation provider item — usable or rejected, both are valid
evidence — keyed by the exact crop bytes (person crop store digest) and the
worker's model contract. Missing, corrupt or tampered artifacts are ordinary
misses; cache IO never invalidates a healthy worker result.
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


# v2: the key is the crop content digest; frame bytes and bbox left the
# contract with the person crop store (older entries are ordinary misses).
IDENTITY_EMBEDDING_CACHE_SCHEMA_VERSION = 2

_CONTRACT_FIELDS = (
    "backend",
    "modelVersion",
    "dimension",
    "normalized",
    "evidenceFingerprintVersion",
)


class IdentityEmbeddingCacheError(RuntimeError):
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
        raise IdentityEmbeddingCacheError(
            f"{label} must be finite JSON data"
        ) from exc


def build_identity_embedding_cache_contract(
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    crop_digest = str(crop_sha256).strip().lower()
    if len(crop_digest) != 64 or any(
        character not in "0123456789abcdef" for character in crop_digest
    ):
        raise IdentityEmbeddingCacheError(
            "crop_sha256 must be a SHA-256 hex digest"
        )
    missing = [
        field for field in _CONTRACT_FIELDS if model_contract.get(field) in (None, "")
    ]
    if missing:
        raise IdentityEmbeddingCacheError(
            f"model_contract is missing: {', '.join(missing)}"
        )
    return {
        "schemaVersion": IDENTITY_EMBEDDING_CACHE_SCHEMA_VERSION,
        "cropContentSha256": crop_digest,
        "modelContract": {
            field: model_contract[field] for field in _CONTRACT_FIELDS
        },
    }


def identity_embedding_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = _json_value(dict(contract), label="cache contract")
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def identity_embedding_cache_path(
    cache_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = identity_embedding_cache_key(contract)
    return (
        Path(cache_directory).expanduser().resolve()
        / key[:2]
        / f"{key}.json"
    )


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingCacheEntry:
    cache_key: str
    path: Path
    crop_sha256: str
    # The exact per-observation worker item (usable or rejected).
    item: dict[str, Any]

    def detached_item(self) -> dict[str, Any]:
        return deepcopy(self.item)


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingCacheLookup:
    entry: IdentityEmbeddingCacheEntry | None
    status: str
    error: str | None = None


def _valid_payload(payload: Mapping[str, Any]) -> bool:
    item = payload.get("item")
    if not isinstance(item, Mapping):
        return False
    if not isinstance(item.get("usable"), bool):
        return False
    fingerprint = item.get("evidenceFingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    embedding = item.get("embedding")
    if item["usable"] and not isinstance(embedding, list):
        return False
    return True


def _entry_from_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_key: str,
    path: Path,
) -> IdentityEmbeddingCacheEntry | None:
    if envelope.get("schemaVersion") != IDENTITY_EMBEDDING_CACHE_SCHEMA_VERSION:
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
    return IdentityEmbeddingCacheEntry(
        cache_key=expected_key,
        path=path,
        crop_sha256=str(expected_contract["cropContentSha256"]),
        item=deepcopy(payload["item"]),
    )


def lookup_identity_embedding_cache(
    cache_directory: str | Path,
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
) -> IdentityEmbeddingCacheLookup:
    """Return hit/absent/corrupt/error without making cache faults fatal."""

    contract = build_identity_embedding_cache_contract(
        crop_sha256=crop_sha256,
        model_contract=model_contract,
    )
    key = identity_embedding_cache_key(contract)
    path = identity_embedding_cache_path(cache_directory, contract)
    if not path.exists():
        return IdentityEmbeddingCacheLookup(None, "absent")
    if not path.is_file():
        return IdentityEmbeddingCacheLookup(
            None, "error", "cache path is not a file"
        )
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return IdentityEmbeddingCacheLookup(None, "corrupt", str(exc))
    except OSError as exc:
        return IdentityEmbeddingCacheLookup(None, "error", str(exc))
    if not isinstance(envelope, Mapping):
        return IdentityEmbeddingCacheLookup(
            None, "corrupt", "envelope is not an object"
        )
    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:
        return IdentityEmbeddingCacheLookup(
            None,
            "corrupt",
            "contract, checksum or payload validation failed",
        )
    return IdentityEmbeddingCacheLookup(entry, "hit")


def store_identity_embedding_cache(
    cache_directory: str | Path,
    *,
    crop_sha256: str,
    model_contract: Mapping[str, Any],
    item: Mapping[str, Any],
) -> IdentityEmbeddingCacheEntry:
    """Atomically publish one validated per-observation worker item."""

    contract = build_identity_embedding_cache_contract(
        crop_sha256=crop_sha256,
        model_contract=model_contract,
    )
    key = identity_embedding_cache_key(contract)
    payload = _json_value({"item": dict(item)}, label="cache payload")
    if not _valid_payload(payload):
        raise IdentityEmbeddingCacheError("identity embedding item is invalid")
    envelope = {
        "schemaVersion": IDENTITY_EMBEDDING_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "payload": payload,
    }
    path = identity_embedding_cache_path(cache_directory, contract)
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
        raise IdentityEmbeddingCacheError(
            f"could not publish identity embedding cache {path}: {exc}"
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
        raise IdentityEmbeddingCacheError(
            "published identity embedding cache failed validation"
        )
    return entry


__all__ = (
    "IDENTITY_EMBEDDING_CACHE_SCHEMA_VERSION",
    "IdentityEmbeddingCacheEntry",
    "IdentityEmbeddingCacheError",
    "IdentityEmbeddingCacheLookup",
    "build_identity_embedding_cache_contract",
    "identity_embedding_cache_key",
    "identity_embedding_cache_path",
    "lookup_identity_embedding_cache",
    "store_identity_embedding_cache",
)
