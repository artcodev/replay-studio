from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .cache import CACHE_SCHEMA_VERSION, IdentityCacheEntry
from .provider_contract import EMBEDDING_DIMENSION, ProviderEmbedding, ProviderUnavailable
from .request_contract import IdentityRequestError


EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
KNOWN_IDENTITY_ROLES = {"ball", "goalkeeper", "other", "player", "referee"}


@dataclass(frozen=True)
class QualityPolicy:
    minimum_width: int = 16
    minimum_height: int = 30
    minimum_sharpness: float = 12.0
    padding_ratio: float = 0.08

    @classmethod
    def from_environment(cls) -> "QualityPolicy":
        return cls(
            minimum_width=max(1, int(os.environ.get("REID_MIN_CROP_WIDTH", "16"))),
            minimum_height=max(1, int(os.environ.get("REID_MIN_CROP_HEIGHT", "30"))),
            minimum_sharpness=max(
                0.0, float(os.environ.get("REID_MIN_SHARPNESS", "12"))
            ),
            padding_ratio=max(0.0, float(os.environ.get("REID_CROP_PADDING", "0.08"))),
        )


def decode_image(data: bytes, file_index: int) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(image)
    except Exception as exc:
        raise IdentityRequestError(
            f"Frame fileIndex={file_index} is not a readable image"
        ) from exc


def crop_observation(
    image: np.ndarray,
    bbox: dict,
    policy: QualityPolicy,
) -> tuple[np.ndarray, dict, list[str]]:
    height, width = image.shape[:2]
    x = float(bbox["x"])
    y = float(bbox["y"])
    box_width = float(bbox["width"])
    box_height = float(bbox["height"])
    padding_x = box_width * policy.padding_ratio
    padding_y = box_height * policy.padding_ratio
    requested = (
        int(np.floor(x - padding_x)),
        int(np.floor(y - padding_y)),
        int(np.ceil(x + box_width + padding_x)),
        int(np.ceil(y + box_height + padding_y)),
    )
    x1 = min(width, max(0, requested[0]))
    y1 = min(height, max(0, requested[1]))
    x2 = min(width, max(0, requested[2]))
    y2 = min(height, max(0, requested[3]))
    crop = image[y1:y2, x1:x2]
    border_clipped = requested != (x1, y1, x2, y2)
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if box_width < policy.minimum_width or box_height < policy.minimum_height:
        reasons.append("crop-too-small")
    if not crop.size:
        reasons.append("crop-outside-frame")
    elif sharpness < policy.minimum_sharpness:
        reasons.append("crop-too-blurry")
    quality = {
        "cropWidth": int(max(0, x2 - x1)),
        "cropHeight": int(max(0, y2 - y1)),
        "sourceBoxWidth": round(box_width, 3),
        "sourceBoxHeight": round(box_height, 3),
        "borderClipped": border_clipped,
        "sharpness": round(sharpness, 4),
    }
    return crop, quality, reasons


def cache_key(
    crop: np.ndarray,
    bbox: dict,
    policy: QualityPolicy,
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
        "bbox": {
            key: float(bbox[key]).hex() for key in ("x", "y", "width", "height")
        },
        "cropPolicy": {
            "minimumWidth": policy.minimum_width,
            "minimumHeight": policy.minimum_height,
            "minimumSharpness": float(policy.minimum_sharpness).hex(),
            "paddingRatio": float(policy.padding_ratio).hex(),
        },
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


def rejected_cache_entry(quality: dict, reasons: list[str]) -> IdentityCacheEntry:
    return IdentityCacheEntry(
        usable=False,
        quality=dict(quality),
        rejection_reasons=tuple(str(reason) for reason in reasons),
    )


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
