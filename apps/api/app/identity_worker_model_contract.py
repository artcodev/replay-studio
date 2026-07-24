from __future__ import annotations

"""Readiness and immutable model identity parsing for the identity worker."""

from math import isfinite
from typing import Any, Mapping

from .identity_worker_contract import (
    EVIDENCE_FINGERPRINT_VERSION,
    IDENTITY_BACKEND,
    IDENTITY_EMBEDDING_DIMENSION,
    IdentityWorkerError,
)
from .identity_worker_wire_validation import reject_unknown_fields


MODEL_FIELDS = frozenset(
    {
        "backend",
        "dimension",
        "normalized",
        "device",
        "batchSize",
        "modelVersion",
        "checkpointSha256",
        "hrnetCheckpointSha256",
        "modelLoadSeconds",
        "soccerNetCommit",
        "torchVersion",
        "mpsFallbackEnabled",
    }
)
READINESS_FIELDS = MODEL_FIELDS | {
    "status",
    "evidenceFingerprintVersion",
    "cache",
}
ANALYSIS_FIELDS = MODEL_FIELDS | {
    "evidenceFingerprintVersion",
    "items",
    "diagnostics",
}
CACHE_FIELDS = frozenset(
    {
        "schemaVersion",
        "enabled",
        "maxEntries",
        "ttlSeconds",
        "waitTimeoutSeconds",
        "size",
        "inFlight",
        "configurationError",
        "hits",
        "misses",
        "stores",
        "evictions",
        "expirations",
        "corruptMisses",
        "inRequestDeduplicated",
        "concurrentDeduplicated",
        "waitTimeouts",
        "providerFailures",
    }
)


def validate_cache_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IdentityWorkerError(
            "Identity worker returned malformed cache diagnostics"
        )
    reject_unknown_fields(value, CACHE_FIELDS, "Identity worker cache diagnostics")
    if value.get("schemaVersion") != "identity-embedding-cache.v3":
        raise IdentityWorkerError(
            "Identity worker returned an unsupported cache contract"
        )
    if not isinstance(value.get("enabled"), bool):
        raise IdentityWorkerError("Identity worker returned invalid cache enabled flag")
    configuration_error = value.get("configurationError")
    if configuration_error is not None and not isinstance(configuration_error, str):
        raise IdentityWorkerError(
            "Identity worker returned invalid cache configurationError"
        )
    for field in CACHE_FIELDS - {
        "schemaVersion",
        "enabled",
        "configurationError",
    }:
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or float(number) < 0.0
        ):
            raise IdentityWorkerError(
                f"Identity worker returned invalid cache diagnostic {field}"
            )
    return dict(value)


def validate_model_payload(
    payload: object,
    *,
    allowed_fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise IdentityWorkerError(
            "Identity worker returned malformed top-level JSON"
        )
    reject_unknown_fields(payload, allowed_fields, "Identity worker response")
    if (
        payload.get("backend") != IDENTITY_BACKEND
        or payload.get("dimension") != IDENTITY_EMBEDDING_DIMENSION
        or payload.get("normalized") is not True
        or payload.get("evidenceFingerprintVersion")
        != EVIDENCE_FINGERPRINT_VERSION
        or not isinstance(payload.get("modelVersion"), str)
        or not str(payload.get("modelVersion")).strip()
    ):
        raise IdentityWorkerError(
            "Identity worker returned an unsupported model contract"
        )
    device = payload.get("device")
    batch_size = payload.get("batchSize")
    torch_version = payload.get("torchVersion")
    mps_fallback = payload.get("mpsFallbackEnabled")
    if device is not None and (
        not isinstance(device, str) or not device.strip()
    ):
        raise IdentityWorkerError("Identity worker returned an invalid device")
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise IdentityWorkerError("Identity worker returned an invalid batch size")
    if torch_version is not None and not isinstance(torch_version, str):
        raise IdentityWorkerError(
            "Identity worker returned an invalid PyTorch version"
        )
    if mps_fallback is not None and not isinstance(mps_fallback, bool):
        raise IdentityWorkerError(
            "Identity worker returned an invalid MPS fallback flag"
        )
    return payload


def validate_readiness_payload(payload: object) -> dict[str, Any]:
    value = validate_model_payload(payload, allowed_fields=READINESS_FIELDS)
    if value.get("status") != "ready":
        raise IdentityWorkerError("Identity worker is not ready")
    validate_cache_diagnostics(value.get("cache"))
    return value


def project_model_contract(payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        "backend": payload["backend"],
        "modelVersion": str(payload["modelVersion"]),
        "dimension": int(payload["dimension"]),
        "normalized": True,
        "evidenceFingerprintVersion": payload["evidenceFingerprintVersion"],
    }


def project_runtime_contract(payload: Mapping[str, Any]) -> dict[str, object]:
    """Observable inference runtime, deliberately separate from model identity."""

    return {
        "device": str(payload.get("device") or "unknown"),
        "batchSize": int(payload.get("batchSize") or 0),
        "torchVersion": payload.get("torchVersion"),
        "mpsFallbackEnabled": bool(payload.get("mpsFallbackEnabled", False)),
    }
