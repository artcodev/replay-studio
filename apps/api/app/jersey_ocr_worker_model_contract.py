from __future__ import annotations

"""Readiness and immutable model identity parsing for the jersey OCR worker."""

from typing import Any, Mapping

from .jersey_ocr_worker_contract import (
    CONTRACT_VERSION,
    EVIDENCE_FINGERPRINT_VERSION,
    JerseyOcrWorkerError,
)
from .jersey_ocr_worker_wire_validation import reject_unknown_fields


CAPABILITY_FIELDS = frozenset(
    {
        "digitsOnly",
        "maxDigits",
        "evidenceFingerprintVersion",
        "inputScopes",
    }
)
MODEL_FIELDS = frozenset(
    {
        "contractVersion",
        "backend",
        "providerVersion",
        "modelVersion",
        "device",
        "batchSize",
        "modelLoadSeconds",
        "inferenceScope",
        "capabilities",
    }
)
READINESS_FIELDS = MODEL_FIELDS | {"status", "service"}


def validate_model_payload(
    payload: object,
    *,
    allowed_fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JerseyOcrWorkerError("Jersey OCR worker returned malformed JSON")
    reject_unknown_fields(payload, allowed_fields, "Jersey OCR response")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        raise JerseyOcrWorkerError("Jersey OCR worker returned no capabilities")
    reject_unknown_fields(
        capabilities,
        CAPABILITY_FIELDS,
        "Jersey OCR capabilities",
    )
    if (
        payload.get("contractVersion") != CONTRACT_VERSION
        or not isinstance(payload.get("backend"), str)
        or not payload.get("backend")
        or not isinstance(payload.get("modelVersion"), str)
        or not payload.get("modelVersion")
        or capabilities.get("digitsOnly") is not True
        or capabilities.get("maxDigits") != 2
        or capabilities.get("evidenceFingerprintVersion")
        != EVIDENCE_FINGERPRINT_VERSION
    ):
        raise JerseyOcrWorkerError(
            "Jersey OCR worker returned an unsupported contract"
        )
    return payload


def validate_readiness_payload(payload: object) -> dict[str, Any]:
    value = validate_model_payload(payload, allowed_fields=READINESS_FIELDS)
    if value.get("status") != "ready":
        raise JerseyOcrWorkerError("Jersey OCR worker is not ready")
    return value


def project_model_contract(payload: Mapping[str, Any]) -> dict[str, object]:
    capabilities = payload["capabilities"]
    return {
        "contractVersion": payload["contractVersion"],
        "backend": payload["backend"],
        "providerVersion": payload.get("providerVersion"),
        "modelVersion": payload["modelVersion"],
        "inferenceScope": payload.get("inferenceScope", "crop"),
        "digitsOnly": capabilities["digitsOnly"],
        "maxDigits": capabilities["maxDigits"],
        "evidenceFingerprintVersion": capabilities[
            "evidenceFingerprintVersion"
        ],
    }
