"""Provider-neutral client for the optional jersey-number OCR service."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Callable, Sequence

import httpx

from .config import get_settings


CONTRACT_VERSION = "jersey-ocr.v1"
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
VALID_STATUSES = {
    "recognized",
    "no-number",
    "low-confidence",
    "ambiguous",
    "rejected",
}


class JerseyOcrWorkerError(RuntimeError):
    pass


class JerseyOcrWorkerResults(dict[str, dict]):
    """Mapping-compatible OCR results with aggregate cache diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.diagnostics: dict = {}


@dataclass(frozen=True, slots=True)
class JerseyCropRequest:
    crop_id: str
    path: Path
    observation_id: str | None = None
    tracklet_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None


def _base_contract_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    capabilities = payload.get("capabilities")
    return (
        payload.get("contractVersion") == CONTRACT_VERSION
        and isinstance(payload.get("backend"), str)
        and bool(payload.get("backend"))
        and isinstance(payload.get("modelVersion"), str)
        and bool(payload.get("modelVersion"))
        and isinstance(capabilities, dict)
        and capabilities.get("digitsOnly") is True
        and capabilities.get("maxDigits") == 2
        and capabilities.get("evidenceFingerprintVersion")
        == EVIDENCE_FINGERPRINT_VERSION
    )


def jersey_ocr_worker_readiness(*, timeout: float = 2.0) -> dict:
    """Report OCR availability without making the main API unhealthy."""

    settings = get_settings()
    if not settings.jersey_ocr_worker_url:
        return {"configured": False, "status": "disabled", "backend": None}
    endpoint = f"{settings.jersey_ocr_worker_url.rstrip('/')}/health/ready"
    try:
        response = httpx.get(endpoint, timeout=max(0.1, float(timeout)))
        response.raise_for_status()
        payload = response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": None,
            "detail": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": None,
        }
    if payload.get("status") != "ready" or not _base_contract_valid(payload):
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": payload.get("backend") if isinstance(payload, dict) else None,
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": payload["backend"],
        "providerVersion": payload.get("providerVersion"),
        "modelVersion": payload["modelVersion"],
        "device": payload.get("device"),
        "batchSize": payload.get("batchSize"),
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
        "contractVersion": CONTRACT_VERSION,
        "inferenceScope": payload.get("inferenceScope", "crop"),
    }


def _optional_confidence(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise JerseyOcrWorkerError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JerseyOcrWorkerError(f"{label} must be numeric") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise JerseyOcrWorkerError(f"{label} must be between 0 and 1")
    return number


def _validated_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise JerseyOcrWorkerError("Jersey OCR item has an invalid evidence fingerprint")
    return value


def _validated_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise JerseyOcrWorkerError(f"Jersey OCR item has malformed {label}")
    return list(value)


def _validated_quality(value: object) -> dict:
    if not isinstance(value, dict):
        raise JerseyOcrWorkerError("Jersey OCR item has malformed quality")
    for field in ("cropWidth", "cropHeight", "sharpness", "contrast"):
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or float(number) < 0.0
        ):
            raise JerseyOcrWorkerError(f"Jersey OCR item has invalid quality.{field}")
    return dict(value)


def _validated_polygon(value: object) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 2:
        raise JerseyOcrWorkerError("Jersey OCR candidate has a malformed polygon")
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise JerseyOcrWorkerError("Jersey OCR candidate has a malformed polygon")
        coordinates: list[float] = []
        for coordinate in point:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not isfinite(float(coordinate))
            ):
                raise JerseyOcrWorkerError("Jersey OCR candidate has a malformed polygon")
            coordinates.append(float(coordinate))
        points.append(coordinates)
    return points


def _validated_item(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise JerseyOcrWorkerError("Jersey OCR worker returned a malformed item")
    crop_id = raw.get("cropId")
    if not isinstance(crop_id, str) or not crop_id:
        raise JerseyOcrWorkerError("Jersey OCR item has no cropId")
    status = raw.get("status")
    if status not in VALID_STATUSES:
        raise JerseyOcrWorkerError(f"Jersey OCR item has invalid status: {status!r}")
    usable = raw.get("usable")
    if not isinstance(usable, bool):
        raise JerseyOcrWorkerError("Jersey OCR item has no explicit usable boolean")
    fingerprint = _validated_fingerprint(raw.get("evidenceFingerprint"))
    quality = _validated_quality(raw.get("quality"))
    rejection_reasons = _validated_string_list(
        raw.get("rejectionReasons"), "rejectionReasons"
    )
    decision_reasons = _validated_string_list(
        raw.get("decisionReasons"), "decisionReasons"
    )
    number = raw.get("number")
    confidence = _optional_confidence(raw.get("confidence"), "item confidence")
    if status == "recognized":
        if (
            not isinstance(number, str)
            or not number.isascii()
            or not number.isdigit()
            or not 1 <= len(number) <= 2
            or confidence is None
        ):
            raise JerseyOcrWorkerError("Recognized OCR item has an invalid number")
    elif number is not None or confidence is not None:
        raise JerseyOcrWorkerError("Unaccepted OCR item unexpectedly carries a number")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise JerseyOcrWorkerError("Jersey OCR item has no candidates array")
    normalized_candidates: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise JerseyOcrWorkerError("Jersey OCR candidate is malformed")
        candidate_number = candidate.get("number")
        if (
            not isinstance(candidate_number, str)
            or not candidate_number.isascii()
            or not candidate_number.isdigit()
            or not 1 <= len(candidate_number) <= 2
        ):
            raise JerseyOcrWorkerError("Jersey OCR candidate has an invalid number")
        candidate_confidence = _optional_confidence(
            candidate.get("confidence"), "candidate confidence"
        )
        if candidate_confidence is None:
            raise JerseyOcrWorkerError("Jersey OCR candidate has no confidence")
        raw_text = candidate.get("rawText")
        if not isinstance(raw_text, str):
            raise JerseyOcrWorkerError("Jersey OCR candidate has malformed rawText")
        normalized_candidates.append(
            {
                **candidate,
                "confidence": candidate_confidence,
                "polygon": _validated_polygon(candidate.get("polygon")),
            }
        )
    if len({candidate["number"] for candidate in normalized_candidates}) != len(
        normalized_candidates
    ):
        raise JerseyOcrWorkerError("Jersey OCR item has duplicate candidate numbers")
    if usable:
        if status == "rejected" or rejection_reasons:
            raise JerseyOcrWorkerError("Usable OCR item has inconsistent rejection state")
    elif status != "rejected" or not rejection_reasons:
        raise JerseyOcrWorkerError("Rejected OCR item has inconsistent usable state")
    if status == "rejected" and (normalized_candidates or decision_reasons):
        raise JerseyOcrWorkerError("Rejected OCR item unexpectedly carries OCR evidence")
    if status == "recognized":
        if decision_reasons or not any(
            candidate["number"] == number for candidate in normalized_candidates
        ):
            raise JerseyOcrWorkerError("Recognized OCR item has inconsistent candidates")
    elif status == "no-number":
        if normalized_candidates or not decision_reasons:
            raise JerseyOcrWorkerError("No-number OCR item has inconsistent candidates")
    elif status == "low-confidence":
        if not normalized_candidates or not decision_reasons:
            raise JerseyOcrWorkerError("Low-confidence OCR item has inconsistent candidates")
    elif status == "ambiguous":
        if len({candidate["number"] for candidate in normalized_candidates}) < 2 or not decision_reasons:
            raise JerseyOcrWorkerError("Ambiguous OCR item has inconsistent candidates")
    return {
        **raw,
        "confidence": confidence,
        "candidates": normalized_candidates,
        "quality": quality,
        "rejectionReasons": rejection_reasons,
        "decisionReasons": decision_reasons,
        "evidenceFingerprint": fingerprint,
    }


def analyze_jersey_crops(
    crops: Sequence[JerseyCropRequest],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> JerseyOcrWorkerResults:
    """Analyze person crops while preserving immutable observation identity.

    The service currently emits crop evidence. ``tracklet_id`` is already part
    of the wire contract so a future PARSeq/tracklet provider can aggregate
    temporally diverse views without changing API callers.
    """

    settings = get_settings()
    if not settings.jersey_ocr_worker_url or not crops:
        return JerseyOcrWorkerResults()
    requested_ids: set[str] = set()
    for crop in crops:
        if not crop.crop_id:
            raise JerseyOcrWorkerError("Every OCR crop requires crop_id")
        if crop.crop_id in requested_ids:
            raise JerseyOcrWorkerError(f"Duplicate OCR crop: {crop.crop_id}")
        if not crop.path.is_file():
            raise JerseyOcrWorkerError(f"OCR crop is missing: {crop.path}")
        requested_ids.add(crop.crop_id)

    batch_size = max(1, int(settings.jersey_ocr_worker_batch_size))
    results = JerseyOcrWorkerResults()
    recognized_count = 0
    accepted_model_contract: dict[str, object] | None = None
    for start in range(0, len(crops), batch_size):
        batch = list(crops[start : start + batch_size])
        files = [
            ("crops", (crop.path.name, crop.path.read_bytes(), "image/jpeg"))
            for crop in batch
        ]
        manifest_items: list[dict] = []
        for file_index, crop in enumerate(batch):
            item = {"cropId": crop.crop_id, "fileIndex": file_index}
            for key, value in (
                ("observationId", crop.observation_id),
                ("trackletId", crop.tracklet_id),
                ("frameIndex", crop.frame_index),
                ("timestamp", crop.timestamp),
            ):
                if value is not None:
                    item[key] = value
            manifest_items.append(item)
        try:
            response = httpx.post(
                f"{settings.jersey_ocr_worker_url.rstrip('/')}/v1/analyze",
                files=files,
                data={
                    "manifest": json.dumps(
                        {
                            "contractVersion": CONTRACT_VERSION,
                            "items": manifest_items,
                        },
                        separators=(",", ":"),
                    )
                },
                timeout=(
                    min(float(timeout), float(settings.jersey_ocr_worker_timeout))
                    if timeout is not None
                    else settings.jersey_ocr_worker_timeout
                ),
            )
            response.raise_for_status()
            payload = response.json()
        except (OSError, ValueError, httpx.HTTPError) as exc:
            raise JerseyOcrWorkerError(f"Jersey OCR worker failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise JerseyOcrWorkerError("Jersey OCR worker returned malformed top-level JSON")
        if not _base_contract_valid(payload):
            raise JerseyOcrWorkerError("Jersey OCR worker returned an unsupported contract")
        capabilities = payload["capabilities"]
        batch_model_contract: dict[str, object] = {
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
        if accepted_model_contract is None:
            accepted_model_contract = batch_model_contract
            results.diagnostics["modelContract"] = dict(accepted_model_contract)
        elif batch_model_contract != accepted_model_contract:
            changed_fields = sorted(
                field
                for field, value in batch_model_contract.items()
                if accepted_model_contract.get(field) != value
            )
            raise JerseyOcrWorkerError(
                "Jersey OCR worker changed model contract between batches: "
                + ", ".join(changed_fields)
            )
        items = payload.get("items")
        if not isinstance(items, list):
            raise JerseyOcrWorkerError("Jersey OCR response has no items array")
        batch_diagnostics = payload.get("diagnostics", {})
        if not isinstance(batch_diagnostics, dict):
            raise JerseyOcrWorkerError("Jersey OCR worker returned malformed diagnostics")
        for field in (
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
        ):
            value = batch_diagnostics.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise JerseyOcrWorkerError(
                    f"Jersey OCR worker returned invalid diagnostic {field}"
                )
            results.diagnostics[field] = int(results.diagnostics.get(field, 0)) + int(value)
        if "cacheEnabled" in batch_diagnostics:
            results.diagnostics["cacheEnabled"] = bool(
                batch_diagnostics.get("cacheEnabled")
            )
        expected_ids = {crop.crop_id for crop in batch}
        received_ids: set[str] = set()
        for raw in items:
            item = _validated_item(raw)
            crop_id = item["cropId"]
            if crop_id in received_ids or crop_id not in expected_ids:
                raise JerseyOcrWorkerError("Jersey OCR worker changed crop identity")
            received_ids.add(crop_id)
            results[crop_id] = {
                **item,
                "provider": payload["backend"],
                "modelVersion": payload["modelVersion"],
            }
            recognized_count += int(item["status"] == "recognized")
        if received_ids != expected_ids:
            raise JerseyOcrWorkerError("Jersey OCR worker returned an incomplete crop batch")
        if on_progress is not None:
            on_progress(min(len(crops), start + len(batch)), len(crops), recognized_count)
    usable_fingerprints = [
        str(item["evidenceFingerprint"])
        for item in results.values()
        if item.get("usable") is True
    ]
    results.diagnostics["uniqueEvidenceFingerprintCount"] = len(
        set(usable_fingerprints)
    )
    results.diagnostics["duplicateEvidenceFingerprintCount"] = (
        len(usable_fingerprints) - len(set(usable_fingerprints))
    )
    return results
