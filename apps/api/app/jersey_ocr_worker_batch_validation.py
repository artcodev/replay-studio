from __future__ import annotations

"""Strict batch identity and diagnostics validation for jersey OCR analysis."""

from math import isfinite

from .jersey_ocr_worker_contract import JerseyOcrWorkerError
from .jersey_ocr_worker_item_validation import validate_ocr_item
from .jersey_ocr_worker_model_contract import (
    MODEL_FIELDS,
    project_model_contract,
    validate_model_payload,
)
from .jersey_ocr_worker_wire_validation import reject_unknown_fields


ANALYSIS_FIELDS = MODEL_FIELDS | {"items", "diagnostics"}
DIAGNOSTIC_FIELDS = frozenset(
    {
        "requestedCropCount",
        "usableCropCount",
        "recognizedCropCount",
        "ambiguousCropCount",
        "rejectedCropCount",
        "providerInferenceCropCount",
        "cacheHitCount",
        "requestDeduplicatedCount",
        "uniqueEvidenceFingerprintCount",
        "duplicateEvidenceFingerprintCount",
        "cacheEnabled",
    }
)


def _validated_diagnostics(value: object) -> dict[str, int | bool]:
    if not isinstance(value, dict):
        raise JerseyOcrWorkerError(
            "Jersey OCR worker returned malformed diagnostics"
        )
    reject_unknown_fields(value, DIAGNOSTIC_FIELDS, "Jersey OCR diagnostics")
    normalized: dict[str, int | bool] = {}
    for field, diagnostic in value.items():
        if field == "cacheEnabled":
            if not isinstance(diagnostic, bool):
                raise JerseyOcrWorkerError(
                    "Jersey OCR worker returned invalid diagnostic cacheEnabled"
                )
            normalized[field] = diagnostic
            continue
        if (
            isinstance(diagnostic, bool)
            or not isinstance(diagnostic, (int, float))
            or not isfinite(float(diagnostic))
            or diagnostic < 0
        ):
            raise JerseyOcrWorkerError(
                f"Jersey OCR worker returned invalid diagnostic {field}"
            )
        normalized[field] = int(diagnostic)
    return normalized


def _validated_items(
    value: object,
    expected_ids: set[str],
) -> dict[str, dict]:
    if not isinstance(value, list):
        raise JerseyOcrWorkerError("Jersey OCR response has no items array")
    received: dict[str, dict] = {}
    for raw in value:
        item = validate_ocr_item(raw)
        crop_id = item["cropId"]
        if crop_id in received or crop_id not in expected_ids:
            raise JerseyOcrWorkerError(
                "Jersey OCR worker changed crop identity"
            )
        received[crop_id] = item
    if set(received) != expected_ids:
        raise JerseyOcrWorkerError(
            "Jersey OCR worker returned an incomplete crop batch"
        )
    return received


def validate_analysis_payload(
    payload: object,
    expected_ids: set[str],
) -> tuple[dict[str, object], dict[str, dict], dict[str, int | bool]]:
    value = validate_model_payload(payload, allowed_fields=ANALYSIS_FIELDS)
    items = _validated_items(value.get("items"), expected_ids)
    diagnostics = _validated_diagnostics(value.get("diagnostics", {}))
    return project_model_contract(value), items, diagnostics
