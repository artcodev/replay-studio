"""Pure identity-embedding and role quality evaluation."""

from __future__ import annotations

from typing import Any, Mapping
import re

import numpy as np

from .evaluation_primitives import check, distribution, is_non_empty_string, is_sha256
from .manifest_contract import ROLES, ValidationManifest


IDENTITY_DIMENSION = 256


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
        and is_non_empty_string(model_version)
        and is_sha256(checkpoint_sha)
        and is_sha256(provider.get("hrnetCheckpointSha256"))
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

    same_distribution = distribution(same_distances)
    different_distribution = distribution(different_distances)
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
        check(
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
        check(
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
        check(
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
        check(
            "identity-usable-crop-ratio",
            usable_ratio >= thresholds["minimumUsableCropRatio"],
            actual=round(usable_ratio, 6),
            operator=">=",
            threshold=thresholds["minimumUsableCropRatio"],
        ),
        check(
            "identity-pair-coverage",
            pair_coverage >= thresholds["minimumPairCoverage"],
            actual=round(pair_coverage, 6),
            operator=">=",
            threshold=thresholds["minimumPairCoverage"],
        ),
        check(
            "identity-same-person-distance-p95",
            same_p95 is not None
            and float(same_p95) <= thresholds["maximumSamePersonDistanceP95"],
            actual=same_p95,
            operator="<=",
            threshold=thresholds["maximumSamePersonDistanceP95"],
        ),
        check(
            "identity-different-person-distance-p05",
            different_p05 is not None
            and float(different_p05) >= thresholds["minimumDifferentPersonDistanceP05"],
            actual=different_p05,
            operator=">=",
            threshold=thresholds["minimumDifferentPersonDistanceP05"],
        ),
        check(
            "identity-median-distance-separation",
            separation is not None
            and separation >= thresholds["minimumMedianDistanceSeparation"],
            actual=round(separation, 6) if separation is not None else None,
            operator=">=",
            threshold=thresholds["minimumMedianDistanceSeparation"],
        ),
        check(
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
