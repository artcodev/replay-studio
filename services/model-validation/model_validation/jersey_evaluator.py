"""Pure jersey-number OCR quality evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from .evaluation_primitives import check, is_non_empty_string
from .manifest_contract import ValidationManifest


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
        and is_non_empty_string(provider_version)
        and provider_version not in {"unknown", "unavailable"}
        and is_non_empty_string(model_version)
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
        check(
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
        check(
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
        check(
            "jersey-usable-crop-ratio",
            usable_ratio >= thresholds["minimumUsableCropRatio"],
            actual=round(usable_ratio, 6),
            operator=">=",
            threshold=thresholds["minimumUsableCropRatio"],
        ),
        check(
            "jersey-readable-exact-accuracy",
            exact_accuracy >= thresholds["minimumReadableExactAccuracy"],
            actual=round(exact_accuracy, 6),
            operator=">=",
            threshold=thresholds["minimumReadableExactAccuracy"],
        ),
        check(
            "jersey-expected-abstention-accuracy",
            expected_abstention_accuracy
            >= thresholds["minimumExpectedAbstentionAccuracy"],
            actual=round(expected_abstention_accuracy, 6),
            operator=">=",
            threshold=thresholds["minimumExpectedAbstentionAccuracy"],
        ),
        check(
            "jersey-readable-abstention-rate",
            readable_abstention_rate <= thresholds["maximumReadableAbstentionRate"],
            actual=round(readable_abstention_rate, 6),
            operator="<=",
            threshold=thresholds["maximumReadableAbstentionRate"],
        ),
        check(
            "jersey-substitution-rate",
            substitution_rate <= thresholds["maximumSubstitutionRate"],
            actual=round(substitution_rate, 6),
            operator="<=",
            threshold=thresholds["maximumSubstitutionRate"],
        ),
        check(
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
