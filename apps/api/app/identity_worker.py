from __future__ import annotations

import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Callable

import httpx

from .config import get_settings


IDENTITY_BACKEND = "prtreid-bpbreid-soccernet"
IDENTITY_EMBEDDING_DIMENSION = 256
EVIDENCE_FINGERPRINT_VERSION = "pixel-evidence-v1"
KNOWN_IDENTITY_ROLES = {"ball", "goalkeeper", "other", "player", "referee"}


class IdentityWorkerError(RuntimeError):
    pass


class IdentityWorkerResults(dict[str, dict]):
    """Mapping-compatible result with aggregate worker/cache diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.diagnostics: dict = {}


def identity_worker_readiness(*, timeout: float = 2.0) -> dict:
    """Report the optional ReID dependency without making the API unhealthy."""

    settings = get_settings()
    if not settings.identity_worker_url:
        return {"configured": False, "status": "disabled", "backend": None}
    endpoint = f"{settings.identity_worker_url.rstrip('/')}/health/ready"
    try:
        response = httpx.get(endpoint, timeout=max(0.1, float(timeout)))
        response.raise_for_status()
        payload = response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": IDENTITY_BACKEND,
            "detail": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": None,
        }
    if (
        payload.get("status") != "ready"
        or payload.get("backend") != IDENTITY_BACKEND
        or payload.get("dimension") != IDENTITY_EMBEDDING_DIMENSION
        or payload.get("normalized") is not True
        or payload.get("evidenceFingerprintVersion") != EVIDENCE_FINGERPRINT_VERSION
        or not isinstance(payload.get("modelVersion"), str)
        or not str(payload.get("modelVersion")).strip()
    ):
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": payload.get("backend"),
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": payload["backend"],
        "device": payload.get("device"),
        "batchSize": payload.get("batchSize"),
        "dimension": payload["dimension"],
        "normalized": True,
        "modelVersion": payload["modelVersion"],
        "evidenceFingerprintVersion": payload["evidenceFingerprintVersion"],
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
        "soccerNetCommit": payload.get("soccerNetCommit"),
    }


def _validated_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise IdentityWorkerError("Identity worker returned an invalid evidence fingerprint")
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
    required_non_negative = (
        "cropWidth",
        "cropHeight",
        "sourceBoxWidth",
        "sourceBoxHeight",
        "sharpness",
    )
    for field in required_non_negative:
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or float(number) < 0.0
        ):
            raise IdentityWorkerError(f"Identity worker returned invalid quality.{field}")
    if not isinstance(value.get("borderClipped"), bool):
        raise IdentityWorkerError("Identity worker returned invalid quality.borderClipped")
    return dict(value)


def _validated_item(item: object) -> dict:
    if not isinstance(item, dict):
        raise IdentityWorkerError("Identity worker returned a malformed item")
    observation_id = item.get("observationId")
    if not isinstance(observation_id, str) or not observation_id:
        raise IdentityWorkerError("Identity worker returned an item without observationId")
    usable_value = item.get("usable")
    if not isinstance(usable_value, bool):
        raise IdentityWorkerError("Identity worker item has no explicit usable boolean")
    usable = usable_value
    frame_index = item.get("frameIndex")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise IdentityWorkerError("Identity worker returned an invalid frameIndex")
    quality = _validated_quality(item.get("quality"))
    rejection_reasons = _validated_string_list(
        item.get("rejectionReasons"), "rejectionReasons"
    )
    fingerprint = _validated_fingerprint(item.get("evidenceFingerprint"))
    vector = item.get("embedding")
    if not usable:
        if vector is not None:
            raise IdentityWorkerError("Rejected identity crop unexpectedly contains an embedding")
        if not rejection_reasons:
            raise IdentityWorkerError("Rejected identity crop has no rejection reason")
        if any(item.get(field) is not None for field in ("visibilityScores", "role", "roleConfidence")):
            raise IdentityWorkerError("Rejected identity crop unexpectedly contains identity evidence")
        return {
            **item,
            "quality": quality,
            "rejectionReasons": rejection_reasons,
            "evidenceFingerprint": fingerprint,
        }
    if rejection_reasons:
        raise IdentityWorkerError("Usable identity crop unexpectedly has rejection reasons")
    if not isinstance(vector, list) or len(vector) != IDENTITY_EMBEDDING_DIMENSION:
        raise IdentityWorkerError("Identity worker returned an invalid embedding dimension")
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise IdentityWorkerError("Identity worker returned a non-numeric embedding") from exc
    if not all(isfinite(value) for value in values):
        raise IdentityWorkerError("Identity worker returned a non-finite embedding")
    norm = sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 1e-3:
        raise IdentityWorkerError(
            f"Identity worker returned a non-normalized embedding (norm={norm:.6f})"
        )
    visibility = item.get("visibilityScores")
    normalized_visibility: list[float] | None = None
    if visibility is not None:
        if not isinstance(visibility, list) or not visibility:
            raise IdentityWorkerError("Identity worker returned malformed visibilityScores")
        normalized_visibility = [
            _validated_probability(value, "visibility score") for value in visibility
        ]
    role = item.get("role")
    role_confidence = item.get("roleConfidence")
    if role is None:
        if role_confidence is not None:
            raise IdentityWorkerError("Identity worker returned roleConfidence without role")
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


def embed_identity_frames(
    frames: list[tuple[int, Path, list[dict]]],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> IdentityWorkerResults:
    """Embed observation bboxes, batching whole source frames over HTTP.

    Each tuple contains ``(frame_index, jpeg_path, observations)``. An
    observation must contain a stable ``observationId`` and an image-space
    ``bbox``. Results include rejected crops so the caller can audit ReID
    coverage instead of interpreting missing vectors as negative evidence.
    """

    settings = get_settings()
    if not settings.identity_worker_url or not frames:
        return IdentityWorkerResults()
    requested_ids: set[str] = set()
    for _, _, observations in frames:
        for observation in observations:
            observation_id = observation.get("observationId")
            if not isinstance(observation_id, str) or not observation_id:
                raise IdentityWorkerError("Every identity observation requires observationId")
            if observation_id in requested_ids:
                raise IdentityWorkerError(f"Duplicate identity observation: {observation_id}")
            requested_ids.add(observation_id)
    batch_size = max(1, int(settings.identity_worker_batch_size))
    results = IdentityWorkerResults()
    usable_count = 0
    accepted_model_contract: dict[str, object] | None = None
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        files = [
            ("frames", (path.name, path.read_bytes(), "image/jpeg"))
            for _, path, _ in batch
        ]
        manifest = {
            "frames": [
                {
                    "frameIndex": int(frame_index),
                    "fileIndex": file_index,
                    "observations": observations,
                }
                for file_index, (frame_index, _path, observations) in enumerate(batch)
            ]
        }
        try:
            response = httpx.post(
                f"{settings.identity_worker_url.rstrip('/')}/v1/embeddings",
                data={"manifest": json.dumps(manifest, separators=(",", ":"))},
                files=files,
                timeout=(
                    min(float(timeout), float(settings.identity_worker_timeout))
                    if timeout is not None
                    else settings.identity_worker_timeout
                ),
            )
            response.raise_for_status()
            payload = response.json()
        except (OSError, ValueError, httpx.HTTPError) as exc:
            raise IdentityWorkerError(f"Identity worker failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise IdentityWorkerError("Identity worker returned malformed top-level JSON")
        if (
            payload.get("backend") != IDENTITY_BACKEND
            or payload.get("dimension") != IDENTITY_EMBEDDING_DIMENSION
            or payload.get("normalized") is not True
            or payload.get("evidenceFingerprintVersion") != EVIDENCE_FINGERPRINT_VERSION
            or not isinstance(payload.get("modelVersion"), str)
            or not str(payload.get("modelVersion")).strip()
        ):
            raise IdentityWorkerError("Identity worker returned an unsupported model contract")
        batch_model_contract: dict[str, object] = {
            "backend": payload["backend"],
            "modelVersion": str(payload["modelVersion"]),
            "dimension": int(payload["dimension"]),
            "normalized": True,
            "evidenceFingerprintVersion": payload["evidenceFingerprintVersion"],
        }
        if accepted_model_contract is None:
            accepted_model_contract = batch_model_contract
            # Publish the one accepted contract, rather than reducing several
            # batch versions to whichever response happened to be processed
            # last. Callers can therefore audit the exact embedding space used
            # by every item in this result.
            results.diagnostics["modelContract"] = dict(accepted_model_contract)
        elif batch_model_contract != accepted_model_contract:
            changed_fields = sorted(
                field
                for field, value in batch_model_contract.items()
                if accepted_model_contract.get(field) != value
            )
            raise IdentityWorkerError(
                "Identity worker changed model contract between batches: "
                + ", ".join(changed_fields)
            )
        items = payload.get("items")
        if not isinstance(items, list):
            raise IdentityWorkerError("Identity worker response has no items array")
        batch_diagnostics = payload.get("diagnostics", {})
        if not isinstance(batch_diagnostics, dict):
            raise IdentityWorkerError("Identity worker returned malformed diagnostics")
        additive_fields = (
            "requestedObservationCount",
            "usableObservationCount",
            "rejectedObservationCount",
            "cacheHitCount",
            "cacheMissCount",
            "deduplicatedObservationCount",
            "concurrentDeduplicatedCount",
            "providerInferenceCount",
            "corruptCacheMissCount",
            "expiredCacheMissCount",
            "uniqueEvidenceFingerprintCount",
            "duplicateEvidenceFingerprintCount",
        )
        for field in additive_fields:
            value = batch_diagnostics.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise IdentityWorkerError(
                    f"Identity worker returned invalid diagnostic {field}"
                )
            results.diagnostics[field] = int(results.diagnostics.get(field, 0)) + int(value)
        if isinstance(batch_diagnostics.get("cache"), dict):
            results.diagnostics["cache"] = batch_diagnostics["cache"]
        expected_ids = {
            str(observation.get("observationId"))
            for _, _, observations in batch
            for observation in observations
        }
        received_ids: set[str] = set()
        for raw_item in items:
            item = _validated_item(raw_item)
            observation_id = str(item["observationId"])
            if observation_id in received_ids or observation_id not in expected_ids:
                raise IdentityWorkerError("Identity worker changed observation identity")
            received_ids.add(observation_id)
            results[observation_id] = {
                **item,
                "provider": payload["backend"],
                "modelVersion": payload["modelVersion"],
            }
            usable_count += int(item.get("usable") is True)
        if received_ids != expected_ids:
            raise IdentityWorkerError("Identity worker returned an incomplete observation batch")
        if on_progress is not None:
            on_progress(min(len(frames), start + len(batch)), len(frames), usable_count)
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
