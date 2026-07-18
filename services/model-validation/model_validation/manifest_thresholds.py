"""Strict acceptance-threshold parser for model validation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .manifest_contract import IDENTITY_THRESHOLD_KEYS, OCR_THRESHOLD_KEYS, ManifestError
from .manifest_parsing import bounded_number, reject_unknown


def parse_thresholds(value: Any) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, dict):
        raise ManifestError("thresholds must be an object")
    reject_unknown(value, {"identity", "jerseyOcr"}, "thresholds")
    identity_raw = value.get("identity")
    ocr_raw = value.get("jerseyOcr")
    if not isinstance(identity_raw, dict) or not isinstance(ocr_raw, dict):
        raise ManifestError("thresholds.identity and thresholds.jerseyOcr are required")
    reject_unknown(identity_raw, set(IDENTITY_THRESHOLD_KEYS), "thresholds.identity")
    reject_unknown(ocr_raw, set(OCR_THRESHOLD_KEYS), "thresholds.jerseyOcr")

    identity_thresholds: dict[str, float] = {}
    for key in IDENTITY_THRESHOLD_KEYS:
        maximum = (
            0.1
            if key == "normalizationTolerance"
            else 2.0
            if "Distance" in key
            else 1.0
        )
        identity_thresholds[key] = bounded_number(
            identity_raw.get(key),
            f"thresholds.identity.{key}",
            minimum=0.0,
            maximum=maximum,
        )
    ocr_thresholds = {
        key: bounded_number(
            ocr_raw.get(key),
            f"thresholds.jerseyOcr.{key}",
            minimum=0.0,
            maximum=1.0,
        )
        for key in OCR_THRESHOLD_KEYS
    }
    return MappingProxyType(
        {
            "identity": MappingProxyType(identity_thresholds),
            "jerseyOcr": MappingProxyType(ocr_thresholds),
        }
    )

