"""Labelled, provider-neutral validation for the two identity workers.

The module intentionally owns no model loader and cannot download weights.  A
real run talks only to already configured, ready HTTP workers.  Pure metric
functions accept explicit predictions so the harness itself can be tested with
tiny fixtures without turning fake-provider results into accuracy claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
import mimetypes
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "football-model-validation-manifest.v1"
REPORT_VERSION = "football-model-validation-report.v1"
IDENTITY_DIMENSION = 256
ROLES = ("ball", "goalkeeper", "other", "player", "referee")

IDENTITY_THRESHOLD_KEYS = (
    "normalizationTolerance",
    "minimumUsableCropRatio",
    "minimumPairCoverage",
    "maximumSamePersonDistanceP95",
    "minimumDifferentPersonDistanceP05",
    "minimumMedianDistanceSeparation",
    "minimumRoleAccuracy",
)
OCR_THRESHOLD_KEYS = (
    "minimumUsableCropRatio",
    "minimumReadableExactAccuracy",
    "minimumExpectedAbstentionAccuracy",
    "maximumReadableAbstentionRate",
    "maximumSubstitutionRate",
    "maximumConflictGroupRate",
)


class ManifestError(ValueError):
    """The labelled validation manifest is incomplete or inconsistent."""


class WorkerUnavailable(RuntimeError):
    """A real worker or its configured assets are not ready."""


class WorkerProtocolError(RuntimeError):
    """A ready worker violated its published response contract."""


@dataclass(frozen=True)
class CropLabel:
    crop_id: str
    path: Path
    relative_path: str
    person_id: str
    role: str
    jersey_readable: bool
    jersey_number: str | None


@dataclass(frozen=True)
class IdentityPair:
    pair_id: str
    left_crop_id: str
    right_crop_id: str
    same_person: bool


@dataclass(frozen=True)
class ValidationManifest:
    source_path: Path
    dataset: dict[str, str]
    crops: tuple[CropLabel, ...]
    identity_pairs: tuple[IdentityPair, ...]
    thresholds: dict[str, dict[str, float]]
    fingerprint: str

    @property
    def crop_by_id(self) -> dict[str, CropLabel]:
        return {crop.crop_id: crop for crop in self.crops}


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value.strip()


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ManifestError(f"{label} has unsupported fields: {', '.join(unexpected)}")


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result) or not minimum <= result <= maximum:
        raise ManifestError(f"{label} must be between {minimum} and {maximum}")
    return result


def _safe_crop_path(base: Path, relative_path: str, label: str) -> Path:
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ManifestError(f"{label} must be relative to the manifest")
    resolved = (base / raw).resolve()
    try:
        common = os.path.commonpath((str(base), str(resolved)))
    except ValueError as exc:
        raise ManifestError(f"{label} is outside the manifest directory") from exc
    if common != str(base):
        raise ManifestError(f"{label} is outside the manifest directory")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ManifestError(f"{label} does not reference a non-empty file")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(
    schema_version: str,
    dataset: Mapping[str, str],
    crops: Sequence[CropLabel],
    pairs: Sequence[IdentityPair],
) -> str:
    # Thresholds are intentionally excluded: this identifies labels and bytes,
    # while every report records the independently selected acceptance gates.
    payload = {
        "schemaVersion": schema_version,
        "dataset": dict(dataset),
        "crops": [
            {
                "id": crop.crop_id,
                "path": crop.relative_path,
                "personId": crop.person_id,
                "role": crop.role,
                "jerseyLabel": {
                    "readable": crop.jersey_readable,
                    "number": crop.jersey_number,
                },
                "byteLength": crop.path.stat().st_size,
                "sha256": _file_sha256(crop.path),
            }
            for crop in sorted(crops, key=lambda item: item.crop_id)
        ],
        "identityPairs": [
            {
                "id": pair.pair_id,
                "leftCropId": pair.left_crop_id,
                "rightCropId": pair.right_crop_id,
                "samePerson": pair.same_person,
            }
            for pair in sorted(pairs, key=lambda item: item.pair_id)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(canonical).hexdigest()}"


def load_manifest(path: str | Path) -> ValidationManifest:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ManifestError(f"Manifest does not exist: {source_path}")
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("Manifest root must be an object")
    _reject_unknown(
        raw,
        {"$schema", "schemaVersion", "dataset", "crops", "identityPairs", "thresholds"},
        "manifest",
    )
    if raw.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"schemaVersion must equal {MANIFEST_SCHEMA_VERSION!r}"
        )

    raw_dataset = raw.get("dataset")
    if not isinstance(raw_dataset, dict):
        raise ManifestError("dataset must be an object")
    _reject_unknown(raw_dataset, {"name", "version", "license", "source"}, "dataset")
    dataset = {
        key: _required_string(raw_dataset.get(key), f"dataset.{key}")
        for key in ("name", "version", "license", "source")
    }

    raw_crops = raw.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ManifestError("crops must be a non-empty array")
    if len(raw_crops) > 5000:
        raise ManifestError("crops exceeds the 5000-item safety limit")
    base = source_path.parent.resolve()
    crops: list[CropLabel] = []
    crop_ids: set[str] = set()
    readable_count = 0
    unreadable_count = 0
    expected_numbers_by_person: dict[str, set[str]] = {}
    for index, raw_crop in enumerate(raw_crops):
        label = f"crops[{index}]"
        if not isinstance(raw_crop, dict):
            raise ManifestError(f"{label} must be an object")
        _reject_unknown(
            raw_crop,
            {"id", "path", "personId", "role", "jerseyLabel"},
            label,
        )
        crop_id = _required_string(raw_crop.get("id"), f"{label}.id")
        if crop_id in crop_ids:
            raise ManifestError(f"Duplicate crop id: {crop_id}")
        crop_ids.add(crop_id)
        relative_path = _required_string(raw_crop.get("path"), f"{label}.path")
        person_id = _required_string(raw_crop.get("personId"), f"{label}.personId")
        role = _required_string(raw_crop.get("role"), f"{label}.role")
        if role not in ROLES:
            raise ManifestError(f"{label}.role must be one of {', '.join(ROLES)}")
        jersey = raw_crop.get("jerseyLabel")
        if not isinstance(jersey, dict) or not isinstance(jersey.get("readable"), bool):
            raise ManifestError(f"{label}.jerseyLabel.readable must be boolean")
        _reject_unknown(jersey, {"readable", "number"}, f"{label}.jerseyLabel")
        readable = bool(jersey["readable"])
        number = jersey.get("number")
        if readable:
            if not isinstance(number, str) or re.fullmatch(r"[0-9]{1,2}", number) is None:
                raise ManifestError(
                    f"{label}.jerseyLabel.number must be one or two ASCII digits when readable"
                )
            readable_count += 1
            expected_numbers_by_person.setdefault(person_id, set()).add(number)
        else:
            if number is not None:
                raise ManifestError(
                    f"{label}.jerseyLabel.number must be null when unreadable"
                )
            unreadable_count += 1
        crops.append(
            CropLabel(
                crop_id=crop_id,
                path=_safe_crop_path(base, relative_path, f"{label}.path"),
                relative_path=relative_path,
                person_id=person_id,
                role=role,
                jersey_readable=readable,
                jersey_number=number,
            )
        )
    if readable_count == 0 or unreadable_count == 0:
        raise ManifestError(
            "crops must contain readable jersey labels and expected-abstention labels"
        )
    inconsistent_people = sorted(
        person_id
        for person_id, numbers in expected_numbers_by_person.items()
        if len(numbers) > 1
    )
    if inconsistent_people:
        raise ManifestError(
            "A person has conflicting readable ground-truth jersey labels: "
            + ", ".join(inconsistent_people)
        )

    raw_pairs = raw.get("identityPairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ManifestError("identityPairs must be a non-empty array")
    if len(raw_pairs) > 100000:
        raise ManifestError("identityPairs exceeds the 100000-item safety limit")
    crop_by_id = {crop.crop_id: crop for crop in crops}
    pairs: list[IdentityPair] = []
    pair_ids: set[str] = set()
    same_count = 0
    different_count = 0
    for index, raw_pair in enumerate(raw_pairs):
        label = f"identityPairs[{index}]"
        if not isinstance(raw_pair, dict):
            raise ManifestError(f"{label} must be an object")
        _reject_unknown(
            raw_pair,
            {"id", "leftCropId", "rightCropId", "samePerson"},
            label,
        )
        pair_id = _required_string(raw_pair.get("id"), f"{label}.id")
        if pair_id in pair_ids:
            raise ManifestError(f"Duplicate identity pair id: {pair_id}")
        pair_ids.add(pair_id)
        left_id = _required_string(raw_pair.get("leftCropId"), f"{label}.leftCropId")
        right_id = _required_string(raw_pair.get("rightCropId"), f"{label}.rightCropId")
        if left_id == right_id:
            raise ManifestError(f"{label} must reference two different crops")
        if left_id not in crop_by_id or right_id not in crop_by_id:
            raise ManifestError(f"{label} references an unknown crop")
        same_person = raw_pair.get("samePerson")
        if not isinstance(same_person, bool):
            raise ManifestError(f"{label}.samePerson must be boolean")
        ground_truth_same = (
            crop_by_id[left_id].person_id == crop_by_id[right_id].person_id
        )
        if same_person != ground_truth_same:
            raise ManifestError(
                f"{label}.samePerson conflicts with the two personId labels"
            )
        same_count += int(same_person)
        different_count += int(not same_person)
        pairs.append(IdentityPair(pair_id, left_id, right_id, same_person))
    if same_count == 0 or different_count == 0:
        raise ManifestError(
            "identityPairs must contain at least one same-person and one different-person pair"
        )

    raw_thresholds = raw.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ManifestError("thresholds must be an object")
    _reject_unknown(raw_thresholds, {"identity", "jerseyOcr"}, "thresholds")
    identity_raw = raw_thresholds.get("identity")
    ocr_raw = raw_thresholds.get("jerseyOcr")
    if not isinstance(identity_raw, dict) or not isinstance(ocr_raw, dict):
        raise ManifestError("thresholds.identity and thresholds.jerseyOcr are required")
    _reject_unknown(identity_raw, set(IDENTITY_THRESHOLD_KEYS), "thresholds.identity")
    _reject_unknown(ocr_raw, set(OCR_THRESHOLD_KEYS), "thresholds.jerseyOcr")
    identity_thresholds: dict[str, float] = {}
    for key in IDENTITY_THRESHOLD_KEYS:
        maximum = 0.1 if key == "normalizationTolerance" else 2.0 if "Distance" in key else 1.0
        identity_thresholds[key] = _number(
            identity_raw.get(key),
            f"thresholds.identity.{key}",
            minimum=0.0,
            maximum=maximum,
        )
    ocr_thresholds = {
        key: _number(
            ocr_raw.get(key),
            f"thresholds.jerseyOcr.{key}",
            minimum=0.0,
            maximum=1.0,
        )
        for key in OCR_THRESHOLD_KEYS
    }
    thresholds = {"identity": identity_thresholds, "jerseyOcr": ocr_thresholds}
    return ValidationManifest(
        source_path=source_path,
        dataset=dataset,
        crops=tuple(crops),
        identity_pairs=tuple(pairs),
        thresholds=thresholds,
        fingerprint=_dataset_fingerprint(
            MANIFEST_SCHEMA_VERSION,
            dataset,
            crops,
            pairs,
        ),
    )


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": round(float(array.min()), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "maximum": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
    }


def _check(
    check_id: str,
    passed: bool,
    *,
    actual: Any,
    operator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
    }


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def evaluate_identity(
    manifest: ValidationManifest,
    provider: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = manifest.thresholds["identity"]
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    response_contract_valid = set(predictions) == expected_crop_ids
    checkpoint_sha = provider.get("checkpointSha256")
    soccer_commit = provider.get("soccerNetCommit")
    model_version = provider.get("modelVersion")
    provenance_valid = (
        provider.get("backend") == "prtreid-bpbreid-soccernet"
        and provider.get("dimension") == IDENTITY_DIMENSION
        and provider.get("normalized") is True
        and _non_empty_string(model_version)
        and _sha256_string(checkpoint_sha)
        and _sha256_string(provider.get("hrnetCheckpointSha256"))
        and isinstance(soccer_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", soccer_commit) is not None
        and str(model_version).startswith(str(checkpoint_sha)[:16])
        and soccer_commit[:12] in str(model_version)
    )

    vectors: dict[str, np.ndarray] = {}
    contract_rows: list[dict[str, Any]] = []
    norm_errors: list[float] = []
    valid_count = 0
    confusion: dict[str, dict[str, int]] = {role: {} for role in ROLES}
    role_correct = 0
    for crop in manifest.crops:
        prediction = predictions.get(crop.crop_id) or {}
        predicted_role = "__abstain__"
        reasons: list[str] = []
        vector: np.ndarray | None = None
        if prediction.get("usable") is not True:
            reasons.append("worker-marked-unusable")
        else:
            try:
                candidate = np.asarray(prediction.get("embedding"), dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                candidate = np.asarray([], dtype=np.float64)
            if candidate.size != IDENTITY_DIMENSION:
                reasons.append("embedding-dimension-invalid")
            elif not np.isfinite(candidate).all():
                reasons.append("embedding-non-finite")
            else:
                norm = float(np.linalg.norm(candidate))
                norm_error = abs(norm - 1.0)
                norm_errors.append(norm_error)
                if norm <= 1e-12:
                    reasons.append("embedding-zero")
                elif norm_error > thresholds["normalizationTolerance"]:
                    reasons.append("embedding-not-normalized")
                else:
                    vector = candidate / norm
                    vectors[crop.crop_id] = vector
                    valid_count += 1
            role_value = prediction.get("role")
            if isinstance(role_value, str) and role_value in ROLES:
                predicted_role = role_value
        confusion[crop.role][predicted_role] = confusion[crop.role].get(predicted_role, 0) + 1
        role_correct += int(predicted_role == crop.role)
        contract_rows.append(
            {
                "cropId": crop.crop_id,
                "usable": vector is not None,
                "expectedRole": crop.role,
                "predictedRole": predicted_role,
                "reasons": reasons,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    same_distances: list[float] = []
    different_distances: list[float] = []
    for pair in manifest.identity_pairs:
        left = vectors.get(pair.left_crop_id)
        right = vectors.get(pair.right_crop_id)
        if left is None or right is None:
            pair_rows.append(
                {
                    "pairId": pair.pair_id,
                    "samePerson": pair.same_person,
                    "evaluated": False,
                    "distance": None,
                    "reason": "one-or-both-embeddings-unusable",
                }
            )
            continue
        distance = float(np.clip(1.0 - float(np.dot(left, right)), 0.0, 2.0))
        (same_distances if pair.same_person else different_distances).append(distance)
        pair_rows.append(
            {
                "pairId": pair.pair_id,
                "samePerson": pair.same_person,
                "evaluated": True,
                "distance": round(distance, 6),
                "reason": None,
            }
        )

    same_distribution = _distribution(same_distances)
    different_distribution = _distribution(different_distances)
    usable_ratio = valid_count / len(manifest.crops)
    pair_coverage = (len(same_distances) + len(different_distances)) / len(
        manifest.identity_pairs
    )
    role_accuracy = role_correct / len(manifest.crops)
    same_p95 = same_distribution["p95"]
    different_p05 = different_distribution["p05"]
    same_p50 = same_distribution["p50"]
    different_p50 = different_distribution["p50"]
    separation = (
        float(different_p50) - float(same_p50)
        if different_p50 is not None and same_p50 is not None
        else None
    )
    max_norm_error = max(norm_errors, default=None)
    embedding_contract_valid = valid_count == len(manifest.crops)
    checks = [
        _check(
            "identity-provider-provenance",
            provenance_valid,
            actual={
                key: provider.get(key)
                for key in (
                    "backend",
                    "dimension",
                    "normalized",
                    "modelVersion",
                    "checkpointSha256",
                    "hrnetCheckpointSha256",
                    "soccerNetCommit",
                )
            },
            operator="contract",
            threshold="256D normalized + model/checkpoint provenance",
        ),
        _check(
            "identity-response-contract",
            response_contract_valid,
            actual={
                "expectedCropCount": len(expected_crop_ids),
                "responseCount": len(predictions),
                "missingCropIds": sorted(expected_crop_ids - set(predictions)),
                "unexpectedCropIds": sorted(set(predictions) - expected_crop_ids),
            },
            operator="exact-id-set",
            threshold=sorted(expected_crop_ids),
        ),
        _check(
            "identity-embedding-contract",
            embedding_contract_valid,
            actual={
                "valid": valid_count,
                "total": len(manifest.crops),
                "maximumNormError": (
                    round(float(max_norm_error), 8) if max_norm_error is not None else None
                ),
            },
            operator="all",
            threshold={
                "dimension": IDENTITY_DIMENSION,
                "normalizationTolerance": thresholds["normalizationTolerance"],
            },
        ),
        _check(
            "identity-usable-crop-ratio",
            usable_ratio >= thresholds["minimumUsableCropRatio"],
            actual=round(usable_ratio, 6),
            operator=">=",
            threshold=thresholds["minimumUsableCropRatio"],
        ),
        _check(
            "identity-pair-coverage",
            pair_coverage >= thresholds["minimumPairCoverage"],
            actual=round(pair_coverage, 6),
            operator=">=",
            threshold=thresholds["minimumPairCoverage"],
        ),
        _check(
            "identity-same-person-distance-p95",
            same_p95 is not None
            and float(same_p95) <= thresholds["maximumSamePersonDistanceP95"],
            actual=same_p95,
            operator="<=",
            threshold=thresholds["maximumSamePersonDistanceP95"],
        ),
        _check(
            "identity-different-person-distance-p05",
            different_p05 is not None
            and float(different_p05) >= thresholds["minimumDifferentPersonDistanceP05"],
            actual=different_p05,
            operator=">=",
            threshold=thresholds["minimumDifferentPersonDistanceP05"],
        ),
        _check(
            "identity-median-distance-separation",
            separation is not None
            and separation >= thresholds["minimumMedianDistanceSeparation"],
            actual=round(separation, 6) if separation is not None else None,
            operator=">=",
            threshold=thresholds["minimumMedianDistanceSeparation"],
        ),
        _check(
            "identity-role-accuracy",
            role_accuracy >= thresholds["minimumRoleAccuracy"],
            actual=round(role_accuracy, 6),
            operator=">=",
            threshold=thresholds["minimumRoleAccuracy"],
        ),
    ]
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "provider": dict(provider),
        "thresholds": dict(thresholds),
        "metrics": {
            "embeddingContract": {
                "dimension": IDENTITY_DIMENSION,
                "validCropCount": valid_count,
                "totalCropCount": len(manifest.crops),
                "usableCropRatio": round(usable_ratio, 6),
                "maximumNormError": (
                    round(float(max_norm_error), 8) if max_norm_error is not None else None
                ),
            },
            "pairCoverage": round(pair_coverage, 6),
            "samePersonDistance": same_distribution,
            "differentPersonDistance": different_distribution,
            "medianDistanceSeparation": (
                round(separation, 6) if separation is not None else None
            ),
            "roleConfusion": {
                "labels": [*ROLES, "__abstain__"],
                "matrix": confusion,
                "correct": role_correct,
                "total": len(manifest.crops),
                "accuracy": round(role_accuracy, 6),
            },
        },
        "checks": checks,
        "samples": {"crops": contract_rows, "pairs": pair_rows},
    }


def evaluate_jersey_ocr(
    manifest: ValidationManifest,
    provider: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = manifest.thresholds["jerseyOcr"]
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    backend = provider.get("backend")
    provider_version = provider.get("providerVersion")
    model_version = provider.get("modelVersion")
    provenance_valid = (
        backend in ("mmocr-dbnet18-sar", "easyocr-english-digits")
        and _non_empty_string(provider_version)
        and provider_version not in {"unknown", "unavailable"}
        and _non_empty_string(model_version)
        and "unknown" not in str(model_version).lower()
        and str(provider_version) in str(model_version)
        and provider.get("contractVersion") == "jersey-ocr.v1"
    )
    usable_count = 0
    readable_count = 0
    exact_count = 0
    substitution_count = 0
    readable_abstention_count = 0
    expected_abstention_count = 0
    expected_abstention_success_count = 0
    predicted_abstention_count = 0
    correct_predicted_abstention_count = 0
    sample_rows: list[dict[str, Any]] = []
    recognized_by_person: dict[str, set[str]] = {}
    crops_by_person: dict[str, int] = {}
    allowed_statuses = {
        "recognized",
        "no-number",
        "low-confidence",
        "ambiguous",
        "rejected",
    }
    protocol_valid = True
    for crop in manifest.crops:
        prediction = predictions.get(crop.crop_id) or {}
        usable = prediction.get("usable") is True
        usable_count += int(usable)
        status = prediction.get("status")
        if status not in allowed_statuses:
            status = "invalid"
            protocol_valid = False
        raw_number = prediction.get("number")
        number = raw_number if isinstance(raw_number, str) else None
        recognized = status == "recognized" and number is not None
        if status == "recognized" and number is None:
            protocol_valid = False
            recognized = False
        crops_by_person[crop.person_id] = crops_by_person.get(crop.person_id, 0) + 1
        if recognized:
            recognized_by_person.setdefault(crop.person_id, set()).add(number)
        else:
            predicted_abstention_count += 1
            if not crop.jersey_readable:
                correct_predicted_abstention_count += 1
        if crop.jersey_readable:
            readable_count += 1
            if recognized and number == crop.jersey_number:
                exact_count += 1
                outcome = "exact"
            elif recognized:
                substitution_count += 1
                outcome = "substitution"
            else:
                readable_abstention_count += 1
                outcome = "readable-abstention"
        else:
            expected_abstention_count += 1
            if not recognized:
                expected_abstention_success_count += 1
                outcome = "expected-abstention"
            else:
                outcome = "false-read"
        sample_rows.append(
            {
                "cropId": crop.crop_id,
                "personId": crop.person_id,
                "expectedNumber": crop.jersey_number,
                "expectedReadable": crop.jersey_readable,
                "usable": usable,
                "status": status,
                "predictedNumber": number,
                "outcome": outcome,
            }
        )

    conflict_eligible = sorted(
        person_id for person_id, count in crops_by_person.items() if count >= 2
    )
    conflict_people = sorted(
        person_id
        for person_id in conflict_eligible
        if len(recognized_by_person.get(person_id, set())) > 1
    )
    usable_ratio = usable_count / len(manifest.crops)
    exact_accuracy = exact_count / readable_count
    expected_abstention_accuracy = (
        expected_abstention_success_count / expected_abstention_count
    )
    readable_abstention_rate = readable_abstention_count / readable_count
    substitution_rate = substitution_count / readable_count
    conflict_rate = len(conflict_people) / len(conflict_eligible) if conflict_eligible else 0.0
    abstention_precision = (
        correct_predicted_abstention_count / predicted_abstention_count
        if predicted_abstention_count
        else None
    )
    checks = [
        _check(
            "jersey-provider-provenance",
            provenance_valid,
            actual={
                key: provider.get(key)
                for key in (
                    "backend",
                    "providerVersion",
                    "modelVersion",
                    "contractVersion",
                    "inferenceScope",
                )
            },
            operator="contract",
            threshold="backend + provider/model version provenance",
        ),
        _check(
            "jersey-response-contract",
            protocol_valid and set(predictions) == expected_crop_ids,
            actual={
                "valid": protocol_valid,
                "responseCount": len(predictions),
                "missingCropIds": sorted(expected_crop_ids - set(predictions)),
                "unexpectedCropIds": sorted(set(predictions) - expected_crop_ids),
            },
            operator="contract",
            threshold={"cropIds": sorted(expected_crop_ids)},
        ),
        _check(
            "jersey-usable-crop-ratio",
            usable_ratio >= thresholds["minimumUsableCropRatio"],
            actual=round(usable_ratio, 6),
            operator=">=",
            threshold=thresholds["minimumUsableCropRatio"],
        ),
        _check(
            "jersey-readable-exact-accuracy",
            exact_accuracy >= thresholds["minimumReadableExactAccuracy"],
            actual=round(exact_accuracy, 6),
            operator=">=",
            threshold=thresholds["minimumReadableExactAccuracy"],
        ),
        _check(
            "jersey-expected-abstention-accuracy",
            expected_abstention_accuracy
            >= thresholds["minimumExpectedAbstentionAccuracy"],
            actual=round(expected_abstention_accuracy, 6),
            operator=">=",
            threshold=thresholds["minimumExpectedAbstentionAccuracy"],
        ),
        _check(
            "jersey-readable-abstention-rate",
            readable_abstention_rate <= thresholds["maximumReadableAbstentionRate"],
            actual=round(readable_abstention_rate, 6),
            operator="<=",
            threshold=thresholds["maximumReadableAbstentionRate"],
        ),
        _check(
            "jersey-substitution-rate",
            substitution_rate <= thresholds["maximumSubstitutionRate"],
            actual=round(substitution_rate, 6),
            operator="<=",
            threshold=thresholds["maximumSubstitutionRate"],
        ),
        _check(
            "jersey-conflict-group-rate",
            conflict_rate <= thresholds["maximumConflictGroupRate"],
            actual=round(conflict_rate, 6),
            operator="<=",
            threshold=thresholds["maximumConflictGroupRate"],
        ),
    ]
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "provider": dict(provider),
        "thresholds": dict(thresholds),
        "metrics": {
            "usableCropCount": usable_count,
            "totalCropCount": len(manifest.crops),
            "usableCropRatio": round(usable_ratio, 6),
            "readableCropCount": readable_count,
            "exactCount": exact_count,
            "readableExactAccuracy": round(exact_accuracy, 6),
            "substitutionCount": substitution_count,
            "substitutionRate": round(substitution_rate, 6),
            "readableAbstentionCount": readable_abstention_count,
            "readableAbstentionRate": round(readable_abstention_rate, 6),
            "expectedAbstentionCropCount": expected_abstention_count,
            "expectedAbstentionSuccessCount": expected_abstention_success_count,
            "expectedAbstentionAccuracy": round(expected_abstention_accuracy, 6),
            "predictedAbstentionCount": predicted_abstention_count,
            "abstentionPrecision": (
                round(abstention_precision, 6)
                if abstention_precision is not None
                else None
            ),
            "conflictEligibleGroupCount": len(conflict_eligible),
            "conflictGroupCount": len(conflict_people),
            "conflictGroupRate": round(conflict_rate, 6),
            "conflictPersonIds": conflict_people,
        },
        "checks": checks,
        "samples": {"crops": sample_rows},
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
        "dataset": {
            **manifest.dataset,
            "fingerprint": manifest.fingerprint,
            "cropCount": len(manifest.crops),
            "identityPairCount": len(manifest.identity_pairs),
            "manifest": str(manifest.source_path),
        },
        "thresholds": manifest.thresholds,
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
        "dataset": {
            **manifest.dataset,
            "fingerprint": manifest.fingerprint,
            "cropCount": len(manifest.crops),
            "identityPairCount": len(manifest.identity_pairs),
            "manifest": str(manifest.source_path),
        },
        "thresholds": manifest.thresholds,
        "selectedWorkers": list(selected_workers),
        "reason": reason,
        "workers": {},
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _chunks(values: Sequence[CropLabel], size: int) -> Iterable[Sequence[CropLabel]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _worker_error(response: Any) -> str:
    try:
        value = response.json()
    except Exception:
        return str(response.text)[:500]
    if isinstance(value, dict) and value.get("detail"):
        return str(value["detail"])[:500]
    return str(value)[:500]


def _response_object(response: Any, worker_name: str) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception as exc:
        raise WorkerProtocolError(f"{worker_name} did not return JSON") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"{worker_name} JSON response is not an object")
    return value


def _ready(client: Any, base_url: str, worker_name: str) -> dict[str, Any]:
    try:
        response = client.get(f"{base_url.rstrip('/')}/health/ready")
    except Exception as exc:
        raise WorkerUnavailable(f"{worker_name} readiness request failed: {exc}") from exc
    if response.status_code != 200:
        raise WorkerUnavailable(
            f"{worker_name} is not ready (HTTP {response.status_code}): {_worker_error(response)}"
        )
    value = _response_object(response, worker_name)
    if value.get("status") != "ready":
        raise WorkerProtocolError(f"{worker_name} readiness response is invalid")
    return value


def _image_size(data: bytes, crop_id: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:
        raise ManifestError(f"Crop {crop_id!r} is not a readable image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ManifestError(f"Crop {crop_id!r} has an empty image size")
    return int(width), int(height)


def _identity_http_predictions(
    client: Any,
    base_url: str,
    manifest: ValidationManifest,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ready = _ready(client, base_url, "identity-worker")
    provider_keys = (
        "backend",
        "dimension",
        "normalized",
        "device",
        "batchSize",
        "modelVersion",
        "checkpointSha256",
        "hrnetCheckpointSha256",
        "soccerNetCommit",
    )
    provider = {key: ready.get(key) for key in provider_keys}
    predictions: dict[str, dict[str, Any]] = {}
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    diagnostics: list[dict[str, Any]] = []
    for batch in _chunks(manifest.crops, batch_size):
        files = []
        frames = []
        for index, crop in enumerate(batch):
            data = crop.path.read_bytes()
            width, height = _image_size(data, crop.crop_id)
            files.append(("frames", (crop.path.name, data, _mime_type(crop.path))))
            frames.append(
                {
                    "fileIndex": index,
                    "frameIndex": index,
                    "observations": [
                        {
                            "observationId": crop.crop_id,
                            "bbox": {
                                "x": 0,
                                "y": 0,
                                "width": width,
                                "height": height,
                            },
                        }
                    ],
                }
            )
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/v1/embeddings",
                files=files,
                data={"manifest": json.dumps({"frames": frames})},
            )
        except Exception as exc:
            raise WorkerUnavailable(f"identity-worker inference request failed: {exc}") from exc
        if response.status_code == 503:
            raise WorkerUnavailable(f"identity-worker inference unavailable: {_worker_error(response)}")
        if response.status_code != 200:
            raise WorkerProtocolError(
                f"identity-worker inference failed (HTTP {response.status_code}): {_worker_error(response)}"
            )
        value = _response_object(response, "identity-worker")
        if not isinstance(value.get("items"), list):
            raise WorkerProtocolError("identity-worker response.items is invalid")
        response_provider = {key: value.get(key) for key in provider_keys}
        if response_provider != provider:
            raise WorkerProtocolError("identity-worker provider provenance changed during the run")
        diagnostics.append(
            value.get("diagnostics")
            if isinstance(value.get("diagnostics"), dict)
            else {}
        )
        for item in value["items"]:
            crop_id = item.get("observationId") if isinstance(item, dict) else None
            if (
                not isinstance(crop_id, str)
                or crop_id not in expected_crop_ids
                or crop_id in predictions
            ):
                raise WorkerProtocolError("identity-worker returned a missing or duplicate observationId")
            predictions[crop_id] = item
    return provider, predictions, diagnostics


def _jersey_http_predictions(
    client: Any,
    base_url: str,
    manifest: ValidationManifest,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ready = _ready(client, base_url, "jersey-ocr-worker")
    provider_keys = (
        "backend",
        "providerVersion",
        "modelVersion",
        "contractVersion",
        "device",
        "batchSize",
        "inferenceScope",
    )
    provider = {key: ready.get(key) for key in provider_keys}
    predictions: dict[str, dict[str, Any]] = {}
    expected_crop_ids = {crop.crop_id for crop in manifest.crops}
    diagnostics: list[dict[str, Any]] = []
    for batch in _chunks(manifest.crops, batch_size):
        files = []
        items = []
        for index, crop in enumerate(batch):
            data = crop.path.read_bytes()
            _image_size(data, crop.crop_id)
            files.append(("crops", (crop.path.name, data, _mime_type(crop.path))))
            items.append(
                {
                    "cropId": crop.crop_id,
                    "fileIndex": index,
                    "observationId": crop.crop_id,
                    "trackletId": crop.person_id,
                    "frameIndex": index,
                    "timestamp": float(index),
                }
            )
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/v1/analyze",
                files=files,
                data={
                    "manifest": json.dumps(
                        {"contractVersion": "jersey-ocr.v1", "items": items}
                    )
                },
            )
        except Exception as exc:
            raise WorkerUnavailable(f"jersey-ocr-worker inference request failed: {exc}") from exc
        if response.status_code == 503:
            raise WorkerUnavailable(
                f"jersey-ocr-worker inference unavailable: {_worker_error(response)}"
            )
        if response.status_code != 200:
            raise WorkerProtocolError(
                f"jersey-ocr-worker inference failed (HTTP {response.status_code}): {_worker_error(response)}"
            )
        value = _response_object(response, "jersey-ocr-worker")
        if not isinstance(value.get("items"), list):
            raise WorkerProtocolError("jersey-ocr-worker response.items is invalid")
        response_provider = {key: value.get(key) for key in provider_keys}
        if response_provider != provider:
            raise WorkerProtocolError("jersey-ocr-worker provider provenance changed during the run")
        diagnostics.append(
            value.get("diagnostics")
            if isinstance(value.get("diagnostics"), dict)
            else {}
        )
        for item in value["items"]:
            crop_id = item.get("cropId") if isinstance(item, dict) else None
            if (
                not isinstance(crop_id, str)
                or crop_id not in expected_crop_ids
                or crop_id in predictions
            ):
                raise WorkerProtocolError("jersey-ocr-worker returned a missing or duplicate cropId")
            predictions[crop_id] = item
    return provider, predictions, diagnostics


def run_http_validation(
    manifest: ValidationManifest,
    *,
    workers: Sequence[str],
    identity_url: str = "http://127.0.0.1:8091",
    jersey_ocr_url: str = "http://127.0.0.1:8093",
    batch_size: int = 16,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    selected = tuple(dict.fromkeys(workers))
    if not selected or any(worker not in {"identity", "jersey-ocr"} for worker in selected):
        raise ValueError("workers must contain identity and/or jersey-ocr")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        import httpx
    except ImportError as exc:
        raise WorkerUnavailable("httpx is required to call the configured workers") from exc
    identity_result = None
    jersey_result = None
    with httpx.Client(timeout=timeout_seconds) as client:
        if "identity" in selected:
            started = perf_counter()
            provider, predictions, diagnostics = _identity_http_predictions(
                client,
                identity_url,
                manifest,
                batch_size,
            )
            identity_result = evaluate_identity(manifest, provider, predictions)
            identity_seconds = perf_counter() - started
            identity_result["benchmark"] = {
                "scope": "ready-check+http+crop-qa+provider-inference",
                "wallSeconds": round(identity_seconds, 6),
                "cropCount": len(manifest.crops),
                "cropsPerSecond": round(
                    len(manifest.crops) / max(identity_seconds, 1e-9),
                    6,
                ),
                "requestBatchSize": batch_size,
                "requestCount": len(diagnostics),
                "batchDiagnostics": diagnostics,
            }
        if "jersey-ocr" in selected:
            started = perf_counter()
            provider, predictions, diagnostics = _jersey_http_predictions(
                client,
                jersey_ocr_url,
                manifest,
                batch_size,
            )
            jersey_result = evaluate_jersey_ocr(manifest, provider, predictions)
            jersey_seconds = perf_counter() - started
            jersey_result["benchmark"] = {
                "scope": "ready-check+http+crop-qa+provider-inference",
                "wallSeconds": round(jersey_seconds, 6),
                "cropCount": len(manifest.crops),
                "cropsPerSecond": round(
                    len(manifest.crops) / max(jersey_seconds, 1e-9),
                    6,
                ),
                "requestBatchSize": batch_size,
                "requestCount": len(diagnostics),
                "batchDiagnostics": diagnostics,
            }
    return build_report(
        manifest,
        identity=identity_result,
        jersey_ocr=jersey_result,
    )


__all__ = [
    "IDENTITY_DIMENSION",
    "MANIFEST_SCHEMA_VERSION",
    "REPORT_VERSION",
    "ManifestError",
    "ValidationManifest",
    "WorkerProtocolError",
    "WorkerUnavailable",
    "build_report",
    "build_unavailable_report",
    "evaluate_identity",
    "evaluate_jersey_ocr",
    "load_manifest",
    "run_http_validation",
    "write_report",
]
