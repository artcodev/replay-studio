from __future__ import annotations

"""Strict validation for one identity-worker observation result."""

from math import isfinite, sqrt

from .identity_worker_contract import (
    IDENTITY_EMBEDDING_DIMENSION,
    KNOWN_IDENTITY_ROLES,
    IdentityWorkerError,
)
from .identity_worker_wire_validation import reject_unknown_fields


ITEM_FIELDS = frozenset(
    {
        "observationId",
        "frameIndex",
        "usable",
        "quality",
        "rejectionReasons",
        "embedding",
        "visibilityScores",
        "role",
        "roleConfidence",
        "evidenceFingerprint",
        "cacheHit",
        "cacheSource",
    }
)
QUALITY_FIELDS = frozenset(
    {
        "cropWidth",
        "cropHeight",
        "sourceBoxWidth",
        "sourceBoxHeight",
        "borderClipped",
        "sharpness",
    }
)


def _validated_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise IdentityWorkerError(
            "Identity worker returned an invalid evidence fingerprint"
        )
    return value


def _validated_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise IdentityWorkerError(f"Identity worker returned malformed {label}")
    return list(value)


def _validated_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentityWorkerError(f"Identity worker returned invalid {label}")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise IdentityWorkerError(f"Identity worker returned invalid {label}")
    return number


def _validated_quality(value: object) -> dict:
    if not isinstance(value, dict):
        raise IdentityWorkerError("Identity worker returned malformed quality")
    reject_unknown_fields(value, QUALITY_FIELDS, "Identity worker quality")
    for field in (
        "cropWidth",
        "cropHeight",
        "sourceBoxWidth",
        "sourceBoxHeight",
        "sharpness",
    ):
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or float(number) < 0.0
        ):
            raise IdentityWorkerError(
                f"Identity worker returned invalid quality.{field}"
            )
    if not isinstance(value.get("borderClipped"), bool):
        raise IdentityWorkerError(
            "Identity worker returned invalid quality.borderClipped"
        )
    return dict(value)


def validate_identity_item(item: object) -> dict:
    if not isinstance(item, dict):
        raise IdentityWorkerError("Identity worker returned a malformed item")
    reject_unknown_fields(item, ITEM_FIELDS, "Identity worker item")
    observation_id = item.get("observationId")
    if not isinstance(observation_id, str) or not observation_id:
        raise IdentityWorkerError(
            "Identity worker returned an item without observationId"
        )
    usable = item.get("usable")
    if not isinstance(usable, bool):
        raise IdentityWorkerError(
            "Identity worker item has no explicit usable boolean"
        )
    frame_index = item.get("frameIndex")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise IdentityWorkerError("Identity worker returned an invalid frameIndex")
    quality = _validated_quality(item.get("quality"))
    rejection_reasons = _validated_string_list(
        item.get("rejectionReasons"), "rejectionReasons"
    )
    fingerprint = _validated_fingerprint(item.get("evidenceFingerprint"))
    cache_hit = item.get("cacheHit")
    if cache_hit is not None and not isinstance(cache_hit, bool):
        raise IdentityWorkerError("Identity worker returned invalid cacheHit")
    cache_source = item.get("cacheSource")
    if cache_source is not None and (
        not isinstance(cache_source, str) or not cache_source
    ):
        raise IdentityWorkerError("Identity worker returned invalid cacheSource")

    vector = item.get("embedding")
    if not usable:
        if vector is not None:
            raise IdentityWorkerError(
                "Rejected identity crop unexpectedly contains an embedding"
            )
        if not rejection_reasons:
            raise IdentityWorkerError(
                "Rejected identity crop has no rejection reason"
            )
        if any(
            item.get(field) is not None
            for field in ("visibilityScores", "role", "roleConfidence")
        ):
            raise IdentityWorkerError(
                "Rejected identity crop unexpectedly contains identity evidence"
            )
        return {
            **item,
            "quality": quality,
            "rejectionReasons": rejection_reasons,
            "evidenceFingerprint": fingerprint,
        }

    if rejection_reasons:
        raise IdentityWorkerError(
            "Usable identity crop unexpectedly has rejection reasons"
        )
    if not isinstance(vector, list) or len(vector) != IDENTITY_EMBEDDING_DIMENSION:
        raise IdentityWorkerError(
            "Identity worker returned an invalid embedding dimension"
        )
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise IdentityWorkerError(
            "Identity worker returned a non-numeric embedding"
        ) from exc
    if not all(isfinite(value) for value in values):
        raise IdentityWorkerError(
            "Identity worker returned a non-finite embedding"
        )
    norm = sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 1e-3:
        raise IdentityWorkerError(
            f"Identity worker returned a non-normalized embedding (norm={norm:.6f})"
        )

    visibility = item.get("visibilityScores")
    normalized_visibility: list[float] | None = None
    if visibility is not None:
        if not isinstance(visibility, list) or not visibility:
            raise IdentityWorkerError(
                "Identity worker returned malformed visibilityScores"
            )
        normalized_visibility = [
            _validated_probability(value, "visibility score")
            for value in visibility
        ]
    role = item.get("role")
    role_confidence = item.get("roleConfidence")
    if role is None:
        if role_confidence is not None:
            raise IdentityWorkerError(
                "Identity worker returned roleConfidence without role"
            )
        normalized_role_confidence = None
    else:
        if role not in KNOWN_IDENTITY_ROLES:
            raise IdentityWorkerError("Identity worker returned an unknown role")
        normalized_role_confidence = _validated_probability(
            role_confidence, "roleConfidence"
        )
    return {
        **item,
        "embedding": values,
        "quality": quality,
        "rejectionReasons": rejection_reasons,
        "visibilityScores": normalized_visibility,
        "roleConfidence": normalized_role_confidence,
        "evidenceFingerprint": fingerprint,
    }
