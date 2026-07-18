from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any


BALL_DETECTION_CACHE_SCHEMA_VERSION = 1


class BallDetectionCacheError(RuntimeError):
    """A clean cache artifact could not be validated or published."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def json_value(value: Any, *, label: str) -> Any:
    """Return a detached JSON value and reject lossy/non-finite inputs."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise BallDetectionCacheError(f"{label} must be finite JSON data") from exc


def ball_detection_input_fingerprint(detector_input: Mapping[str, Any]) -> str:
    normalized = json_value(dict(detector_input), label="detector_input")
    return sha256(canonical_json(normalized).encode()).hexdigest()


def build_ball_detection_cache_contract(
    *, dense_cache_key: str, detector_input: Mapping[str, Any]
) -> dict[str, Any]:
    dense_key = str(dense_cache_key).strip()
    if not dense_key:
        raise BallDetectionCacheError("dense_cache_key must not be empty")
    normalized_input = json_value(dict(detector_input), label="detector_input")
    return {
        "schemaVersion": BALL_DETECTION_CACHE_SCHEMA_VERSION,
        "denseCacheKey": dense_key,
        "detectorInputFingerprint": sha256(
            canonical_json(normalized_input).encode()
        ).hexdigest(),
        "detectorInput": normalized_input,
    }


def ball_detection_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = json_value(dict(contract), label="cache contract")
    return sha256(canonical_json(normalized).encode()).hexdigest()
