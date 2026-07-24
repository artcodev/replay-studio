from __future__ import annotations

"""Strict batch identity and diagnostics validation for ReID analysis."""

from math import isfinite

from .identity_worker_contract import IdentityWorkerError
from .identity_worker_item_validation import validate_identity_item
from .identity_worker_model_contract import (
    ANALYSIS_FIELDS,
    project_model_contract,
    validate_cache_diagnostics,
    validate_model_payload,
)
from .identity_worker_wire_validation import reject_unknown_fields


DIAGNOSTIC_FIELDS = frozenset(
    {
        "requestedObservationCount",
        "usableObservationCount",
        "rejectedObservationCount",
        "cacheHitCount",
        "cacheMissCount",
        "deduplicatedObservationCount",
        "concurrentDeduplicatedCount",
        "providerInferenceCount",
        "providerCallCount",
        "providerInferenceSeconds",
        "requestSeconds",
        "corruptCacheMissCount",
        "expiredCacheMissCount",
        "uniqueEvidenceFingerprintCount",
        "duplicateEvidenceFingerprintCount",
        "cache",
    }
)
TIMING_DIAGNOSTIC_FIELDS = frozenset(
    {"providerInferenceSeconds", "requestSeconds"}
)


def _validated_diagnostics(value: object) -> dict:
    if not isinstance(value, dict):
        raise IdentityWorkerError("Identity worker returned malformed diagnostics")
    reject_unknown_fields(value, DIAGNOSTIC_FIELDS, "Identity worker diagnostics")
    normalized: dict = {}
    for field, diagnostic in value.items():
        if field == "cache":
            normalized[field] = validate_cache_diagnostics(diagnostic)
            continue
        if (
            isinstance(diagnostic, bool)
            or not isinstance(diagnostic, (int, float))
            or not isfinite(float(diagnostic))
            or diagnostic < 0
        ):
            raise IdentityWorkerError(
                f"Identity worker returned invalid diagnostic {field}"
            )
        normalized[field] = (
            float(diagnostic)
            if field in TIMING_DIAGNOSTIC_FIELDS
            else int(diagnostic)
        )
    return normalized


def _validated_items(value: object, expected_ids: set[str]) -> dict[str, dict]:
    if not isinstance(value, list):
        raise IdentityWorkerError("Identity worker response has no items array")
    received: dict[str, dict] = {}
    for raw_item in value:
        item = validate_identity_item(raw_item)
        observation_id = str(item["observationId"])
        if observation_id in received or observation_id not in expected_ids:
            raise IdentityWorkerError(
                "Identity worker changed observation identity"
            )
        received[observation_id] = item
    if set(received) != expected_ids:
        raise IdentityWorkerError(
            "Identity worker returned an incomplete observation batch"
        )
    return received


def validate_embedding_payload(
    payload: object,
    expected_ids: set[str],
) -> tuple[dict[str, object], dict[str, dict], dict]:
    value = validate_model_payload(payload, allowed_fields=ANALYSIS_FIELDS)
    items = _validated_items(value.get("items"), expected_ids)
    diagnostics = _validated_diagnostics(value.get("diagnostics", {}))
    return project_model_contract(value), items, diagnostics
