from __future__ import annotations

from hashlib import sha256
import io
import json
from typing import Any

import numpy as np
from PIL import Image

from .cache import CACHE_SCHEMA_VERSION, IdentityCacheEntry
from .provider_contract import EMBEDDING_DIMENSION, ProviderEmbedding, ProviderUnavailable
from .request_contract import IdentityRequestError


# v2: fingerprints and cache keys are computed over decoded store-crop
# pixels — the worker no longer sees frames, bboxes or crop policies.
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v2"
KNOWN_IDENTITY_ROLES = {"ball", "goalkeeper", "other", "player", "referee"}


def decode_image(data: bytes, file_index: int) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(image)
    except Exception as exc:
        raise IdentityRequestError(
            f"Crop fileIndex={file_index} is not a readable image"
        ) from exc


def cache_key(
    crop: np.ndarray,
    provider_info: dict[str, Any],
) -> str:
    namespace = {
        "schema": CACHE_SCHEMA_VERSION,
        "backend": provider_info.get("backend"),
        "dimension": provider_info.get("dimension"),
        "normalized": provider_info.get("normalized"),
        "modelVersion": provider_info.get("modelVersion"),
        "checkpointSha256": provider_info.get("checkpointSha256")
        or provider_info.get("modelVersion"),
        "hrnetCheckpointSha256": provider_info.get("hrnetCheckpointSha256"),
        "soccerNetCommit": provider_info.get("soccerNetCommit"),
        "cropShape": list(crop.shape),
        "cropDtype": str(crop.dtype),
    }
    digest = sha256(
        json.dumps(namespace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(crop).tobytes())
    return digest.hexdigest()


def evidence_fingerprint(crop: np.ndarray) -> str:
    digest = sha256()
    digest.update(EVIDENCE_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(crop.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(crop.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(crop).tobytes())
    return f"{EVIDENCE_FINGERPRINT_VERSION}:{digest.hexdigest()}"


def provider_cache_entry(
    item: ProviderEmbedding,
    quality: dict,
) -> IdentityCacheEntry:
    vector = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
        raise ProviderUnavailable("Provider returned an invalid embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ProviderUnavailable("Provider returned a zero embedding")
    vector = vector / norm

    visibility_scores: tuple[float, ...] | None = None
    if item.visibility_scores is not None:
        raw_visibility = np.asarray(item.visibility_scores, dtype=np.float32).reshape(-1)
        if not np.isfinite(raw_visibility).all():
            raise ProviderUnavailable("Provider returned non-finite visibility scores")
        visibility_scores = tuple(float(value) for value in raw_visibility)
    role = item.role
    if role is not None and role not in KNOWN_IDENTITY_ROLES:
        raise ProviderUnavailable("Provider returned an invalid role")
    role_confidence = item.role_confidence
    if role is None and role_confidence is not None:
        raise ProviderUnavailable("Provider returned role confidence without a role")
    if role is not None and role_confidence is None:
        raise ProviderUnavailable("Provider returned a role without confidence")
    if role_confidence is not None:
        role_confidence = float(role_confidence)
        if not np.isfinite(role_confidence) or not 0.0 <= role_confidence <= 1.0:
            raise ProviderUnavailable("Provider returned an invalid role confidence")
    return IdentityCacheEntry(
        usable=True,
        quality=dict(quality),
        rejection_reasons=(),
        embedding=tuple(float(value) for value in vector),
        visibility_scores=visibility_scores,
        role=role,
        role_confidence=role_confidence,
    )


def apply_cache_entry(
    response: dict,
    entry: IdentityCacheEntry,
    *,
    cache_source: str,
) -> None:
    response.update(
        {
            "usable": entry.usable,
            "quality": dict(entry.quality),
            "rejectionReasons": list(entry.rejection_reasons),
            "embedding": list(entry.embedding) if entry.embedding is not None else None,
            "visibilityScores": (
                list(entry.visibility_scores)
                if entry.visibility_scores is not None
                else None
            ),
            "role": entry.role,
            "roleConfidence": entry.role_confidence,
            "cacheHit": cache_source == "cache-hit",
            "cacheSource": cache_source,
        }
    )
