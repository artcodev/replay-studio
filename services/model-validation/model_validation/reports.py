"""Pure construction of versioned validation reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from .manifest_contract import MANIFEST_SCHEMA_VERSION, ValidationManifest


REPORT_VERSION = "football-model-validation-report.v1"


def _dataset_summary(manifest: ValidationManifest) -> dict[str, Any]:
    return {
        **manifest.dataset,
        "fingerprint": manifest.fingerprint,
        "cropCount": len(manifest.crops),
        "identityPairCount": len(manifest.identity_pairs),
        "manifest": str(manifest.source_path),
    }


def _threshold_summary(manifest: ValidationManifest) -> dict[str, dict[str, float]]:
    return {
        capability: dict(values)
        for capability, values in manifest.thresholds.items()
    }


def build_report(
    manifest: ValidationManifest,
    *,
    identity: dict[str, Any] | None = None,
    jersey_ocr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workers = {
        key: value
        for key, value in (("identity", identity), ("jerseyOcr", jersey_ocr))
        if value is not None
    }
    if not workers:
        raise ValueError("At least one worker result is required")
    status = "pass" if all(value.get("status") == "pass" for value in workers.values()) else "fail"
    return {
        "reportVersion": REPORT_VERSION,
        "manifestSchemaVersion": MANIFEST_SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset": _dataset_summary(manifest),
        "thresholds": _threshold_summary(manifest),
        "workers": workers,
    }


def build_unavailable_report(
    manifest: ValidationManifest,
    *,
    selected_workers: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "reportVersion": REPORT_VERSION,
        "manifestSchemaVersion": MANIFEST_SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "dataset": _dataset_summary(manifest),
        "thresholds": _threshold_summary(manifest),
        "selectedWorkers": list(selected_workers),
        "reason": reason,
        "workers": {},
    }
