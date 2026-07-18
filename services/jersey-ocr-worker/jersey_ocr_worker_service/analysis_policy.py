from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re
from typing import Any, Sequence
import unicodedata

import cv2
import numpy as np

from .provider_contract import ProviderUnavailable, RawTextCandidate
from .request_contract import EVIDENCE_FINGERPRINT_VERSION


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    minimum_width: int = 20
    minimum_height: int = 36
    minimum_sharpness: float = 5.0
    minimum_contrast: float = 4.0
    minimum_confidence: float = 0.25
    ambiguity_margin: float = 0.05
    max_crop_bytes: int = 8_000_000
    max_crop_pixels: int = 4_000_000
    max_batch_size: int = 128

    @classmethod
    def from_environment(cls) -> "QualityPolicy":
        return cls(
            minimum_width=max(1, int(os.environ.get("JERSEY_OCR_MIN_CROP_WIDTH", "20"))),
            minimum_height=max(1, int(os.environ.get("JERSEY_OCR_MIN_CROP_HEIGHT", "36"))),
            minimum_sharpness=max(
                0.0, float(os.environ.get("JERSEY_OCR_MIN_SHARPNESS", "5"))
            ),
            minimum_contrast=max(
                0.0, float(os.environ.get("JERSEY_OCR_MIN_CONTRAST", "4"))
            ),
            minimum_confidence=float(
                os.environ.get("JERSEY_OCR_MIN_CONFIDENCE", "0.25")
            ),
            ambiguity_margin=max(
                0.0, float(os.environ.get("JERSEY_OCR_AMBIGUITY_MARGIN", "0.05"))
            ),
            max_crop_bytes=max(
                1024, int(os.environ.get("JERSEY_OCR_MAX_CROP_BYTES", "8000000"))
            ),
            max_crop_pixels=max(
                1, int(os.environ.get("JERSEY_OCR_MAX_CROP_PIXELS", "4000000"))
            ),
            max_batch_size=max(
                1, int(os.environ.get("JERSEY_OCR_MAX_BATCH_SIZE", "128"))
            ),
        )

    def validate(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ProviderUnavailable(
                "JERSEY_OCR_MIN_CONFIDENCE must be between 0 and 1"
            )
        if self.ambiguity_margin > 1.0:
            raise ProviderUnavailable(
                "JERSEY_OCR_AMBIGUITY_MARGIN must not exceed 1"
            )


def cache_key(image: np.ndarray, info: dict[str, Any], policy: QualityPolicy) -> str:
    digest = sha256()
    digest.update(str(info["modelVersion"]).encode("utf-8"))
    digest.update(
        (
            f"|{policy.minimum_width}|{policy.minimum_height}|"
            f"{policy.minimum_sharpness}|{policy.minimum_contrast}|"
            f"{policy.minimum_confidence}|{policy.ambiguity_margin}|"
        ).encode("ascii")
    )
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def evidence_fingerprint(image: np.ndarray) -> str:
    digest = sha256()
    digest.update(EVIDENCE_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(image.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes())
    return f"{EVIDENCE_FINGERPRINT_VERSION}:{digest.hexdigest()}"


def assess_quality(
    image: np.ndarray,
    policy: QualityPolicy,
) -> tuple[dict[str, Any], list[str]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    reasons: list[str] = []
    if width < policy.minimum_width or height < policy.minimum_height:
        reasons.append("crop-too-small")
    if sharpness < policy.minimum_sharpness:
        reasons.append("crop-too-blurry")
    if contrast < policy.minimum_contrast:
        reasons.append("crop-too-low-contrast")
    return (
        {
            "cropWidth": width,
            "cropHeight": height,
            "sharpness": round(sharpness, 4),
            "contrast": round(contrast, 4),
        },
        reasons,
    )


def _ascii_digits(value: str) -> str:
    converted: list[str] = []
    for character in value:
        if not character.isdigit():
            converted.append(character)
            continue
        try:
            converted.append(str(unicodedata.digit(character)))
        except (TypeError, ValueError):
            converted.append(character)
    return "".join(converted)


def _number_candidates(raw: Sequence[RawTextCandidate]) -> list[dict[str, Any]]:
    by_number: dict[str, dict[str, Any]] = {}
    for candidate in raw:
        text = str(candidate.text)
        confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
        for match in re.finditer(r"(?<!\d)(\d{1,2})(?!\d)", _ascii_digits(text)):
            number = match.group(1)
            current = by_number.get(number)
            value = {
                "number": number,
                "confidence": round(confidence, 6),
                "rawText": text,
                "polygon": candidate.polygon,
            }
            if current is None or confidence > float(current["confidence"]):
                by_number[number] = value
    return sorted(
        by_number.values(),
        key=lambda item: (-float(item["confidence"]), item["number"]),
    )[:5]


def decide_number(
    raw: Sequence[RawTextCandidate],
    policy: QualityPolicy,
) -> tuple[str, str | None, float | None, list[dict[str, Any]], list[str]]:
    candidates = _number_candidates(raw)
    if not candidates:
        return "no-number", None, None, [], ["no-numeric-text"]
    best = candidates[0]
    confidence = float(best["confidence"])
    if confidence < policy.minimum_confidence:
        return "low-confidence", None, None, candidates, ["confidence-below-threshold"]
    if (
        len(candidates) > 1
        and candidates[1]["number"] != best["number"]
        and confidence - float(candidates[1]["confidence"]) <= policy.ambiguity_margin
    ):
        return "ambiguous", None, None, candidates, ["competing-numbers"]
    return "recognized", str(best["number"]), confidence, candidates, []
