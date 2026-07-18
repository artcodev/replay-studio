from __future__ import annotations

"""Strict item, candidate, quality and polygon parsing for jersey OCR."""

from math import isfinite

from .jersey_ocr_worker_contract import VALID_STATUSES, JerseyOcrWorkerError
from .jersey_ocr_worker_wire_validation import reject_unknown_fields


ITEM_FIELDS = frozenset(
    {
        "cropId",
        "observationId",
        "trackletId",
        "frameIndex",
        "timestamp",
        "evidenceFingerprint",
        "usable",
        "status",
        "number",
        "confidence",
        "candidates",
        "quality",
        "rejectionReasons",
        "decisionReasons",
        "cacheHit",
    }
)
CANDIDATE_FIELDS = frozenset({"number", "confidence", "rawText", "polygon"})
QUALITY_FIELDS = frozenset(
    {"cropWidth", "cropHeight", "sharpness", "contrast"}
)


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
        raise JerseyOcrWorkerError(
            "Jersey OCR item has an invalid evidence fingerprint"
        )
    return value


def _validated_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise JerseyOcrWorkerError(f"Jersey OCR item has malformed {label}")
    return list(value)


def _validated_quality(value: object) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise JerseyOcrWorkerError("Jersey OCR item has malformed quality")
    reject_unknown_fields(value, QUALITY_FIELDS, "Jersey OCR quality")
    result: dict[str, float | int] = {}
    for field in ("cropWidth", "cropHeight", "sharpness", "contrast"):
        number = value.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or float(number) < 0.0
        ):
            raise JerseyOcrWorkerError(
                f"Jersey OCR item has invalid quality.{field}"
            )
        result[field] = number
    return result


def _validated_polygon(value: object) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 2:
        raise JerseyOcrWorkerError(
            "Jersey OCR candidate has a malformed polygon"
        )
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise JerseyOcrWorkerError(
                "Jersey OCR candidate has a malformed polygon"
            )
        coordinates: list[float] = []
        for coordinate in point:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not isfinite(float(coordinate))
            ):
                raise JerseyOcrWorkerError(
                    "Jersey OCR candidate has a malformed polygon"
                )
            coordinates.append(float(coordinate))
        points.append(coordinates)
    return points


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JerseyOcrWorkerError(f"Jersey OCR item has malformed {label}")
    return value


def _validated_candidate(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise JerseyOcrWorkerError("Jersey OCR candidate is malformed")
    reject_unknown_fields(raw, CANDIDATE_FIELDS, "Jersey OCR candidate")
    number = raw.get("number")
    if (
        not isinstance(number, str)
        or not number.isascii()
        or not number.isdigit()
        or not 1 <= len(number) <= 2
    ):
        raise JerseyOcrWorkerError(
            "Jersey OCR candidate has an invalid number"
        )
    confidence = _optional_confidence(
        raw.get("confidence"),
        "candidate confidence",
    )
    if confidence is None:
        raise JerseyOcrWorkerError("Jersey OCR candidate has no confidence")
    raw_text = raw.get("rawText")
    if not isinstance(raw_text, str):
        raise JerseyOcrWorkerError(
            "Jersey OCR candidate has malformed rawText"
        )
    return {
        "number": number,
        "confidence": confidence,
        "rawText": raw_text,
        "polygon": _validated_polygon(raw.get("polygon")),
    }


def _validated_candidates(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise JerseyOcrWorkerError("Jersey OCR item has no candidates array")
    candidates = [_validated_candidate(candidate) for candidate in value]
    if len({candidate["number"] for candidate in candidates}) != len(candidates):
        raise JerseyOcrWorkerError(
            "Jersey OCR item has duplicate candidate numbers"
        )
    return candidates


def _validate_recognition_state(
    *,
    status: str,
    usable: bool,
    number: object,
    confidence: float | None,
    candidates: list[dict],
    rejection_reasons: list[str],
    decision_reasons: list[str],
) -> None:
    if status == "recognized":
        if (
            not isinstance(number, str)
            or not number.isascii()
            or not number.isdigit()
            or not 1 <= len(number) <= 2
            or confidence is None
        ):
            raise JerseyOcrWorkerError(
                "Recognized OCR item has an invalid number"
            )
    elif number is not None or confidence is not None:
        raise JerseyOcrWorkerError(
            "Unaccepted OCR item unexpectedly carries a number"
        )

    if usable:
        if status == "rejected" or rejection_reasons:
            raise JerseyOcrWorkerError(
                "Usable OCR item has inconsistent rejection state"
            )
    elif status != "rejected" or not rejection_reasons:
        raise JerseyOcrWorkerError(
            "Rejected OCR item has inconsistent usable state"
        )

    if status == "rejected" and (candidates or decision_reasons):
        raise JerseyOcrWorkerError(
            "Rejected OCR item unexpectedly carries OCR evidence"
        )
    if status == "recognized":
        if decision_reasons or not any(
            candidate["number"] == number for candidate in candidates
        ):
            raise JerseyOcrWorkerError(
                "Recognized OCR item has inconsistent candidates"
            )
    elif status == "no-number":
        if candidates or not decision_reasons:
            raise JerseyOcrWorkerError(
                "No-number OCR item has inconsistent candidates"
            )
    elif status == "low-confidence":
        if not candidates or not decision_reasons:
            raise JerseyOcrWorkerError(
                "Low-confidence OCR item has inconsistent candidates"
            )
    elif status == "ambiguous":
        if (
            len({candidate["number"] for candidate in candidates}) < 2
            or not decision_reasons
        ):
            raise JerseyOcrWorkerError(
                "Ambiguous OCR item has inconsistent candidates"
            )


def _validated_frame_index(value: object) -> int | None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise JerseyOcrWorkerError("Jersey OCR item has malformed frameIndex")
    return value


def _validated_timestamp(value: object) -> float | None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise JerseyOcrWorkerError("Jersey OCR item has malformed timestamp")
    return float(value) if value is not None else None


def _validated_cache_hit(value: object) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise JerseyOcrWorkerError("Jersey OCR item has malformed cacheHit")
    return value


def validate_ocr_item(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise JerseyOcrWorkerError(
            "Jersey OCR worker returned a malformed item"
        )
    reject_unknown_fields(raw, ITEM_FIELDS, "Jersey OCR item")
    crop_id = raw.get("cropId")
    if not isinstance(crop_id, str) or not crop_id:
        raise JerseyOcrWorkerError("Jersey OCR item has no cropId")
    status = raw.get("status")
    if status not in VALID_STATUSES:
        raise JerseyOcrWorkerError(
            f"Jersey OCR item has invalid status: {status!r}"
        )
    usable = raw.get("usable")
    if not isinstance(usable, bool):
        raise JerseyOcrWorkerError(
            "Jersey OCR item has no explicit usable boolean"
        )

    number = raw.get("number")
    confidence = _optional_confidence(raw.get("confidence"), "item confidence")
    candidates = _validated_candidates(raw.get("candidates"))
    rejection_reasons = _validated_string_list(
        raw.get("rejectionReasons"),
        "rejectionReasons",
    )
    decision_reasons = _validated_string_list(
        raw.get("decisionReasons"),
        "decisionReasons",
    )
    _validate_recognition_state(
        status=status,
        usable=usable,
        number=number,
        confidence=confidence,
        candidates=candidates,
        rejection_reasons=rejection_reasons,
        decision_reasons=decision_reasons,
    )
    return {
        "cropId": crop_id,
        "observationId": _optional_string(
            raw.get("observationId"),
            "observationId",
        ),
        "trackletId": _optional_string(raw.get("trackletId"), "trackletId"),
        "frameIndex": _validated_frame_index(raw.get("frameIndex")),
        "timestamp": _validated_timestamp(raw.get("timestamp")),
        "usable": usable,
        "status": status,
        "number": number,
        "confidence": confidence,
        "candidates": candidates,
        "quality": _validated_quality(raw.get("quality")),
        "rejectionReasons": rejection_reasons,
        "decisionReasons": decision_reasons,
        "evidenceFingerprint": _validated_fingerprint(
            raw.get("evidenceFingerprint")
        ),
        "cacheHit": _validated_cache_hit(raw.get("cacheHit")),
    }
