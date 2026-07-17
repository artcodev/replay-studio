"""Versioned atomic cache for sampled-frame base object detections.

The cache boundary is deliberately before manual annotations, calibration,
projection, tracking, ReID and jersey OCR.  A correction rebuild may therefore
reuse expensive person/legacy-COCO-ball inference without making any downstream
decision sticky.

Every artifact represents one complete primary-provider result for one exact
JPEG.  Its identity includes the frame content hash and the complete detector,
checkpoint, NMS and filtering contract.  Missing, corrupt or tampered artifacts
are ordinary cache misses; fallback and partial results are never published.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


PERSON_DETECTION_CACHE_SCHEMA_VERSION = 1


class PersonDetectionCacheError(RuntimeError):
    """A cache contract or atomic publication could not be completed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise PersonDetectionCacheError(f"{label} must be finite JSON data") from exc


def frame_content_sha256(path: str | Path) -> str:
    """Hash the exact encoded frame bytes, not decoded pixels or a filename."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_person_detection_cache_contract(
    *,
    frame_sha256: str,
    detector_input: Mapping[str, Any],
) -> dict[str, Any]:
    frame_digest = str(frame_sha256).strip().lower()
    if len(frame_digest) != 64 or any(
        character not in "0123456789abcdef" for character in frame_digest
    ):
        raise PersonDetectionCacheError("frame_sha256 must be a SHA-256 hex digest")
    normalized_input = _json_value(dict(detector_input), label="detector_input")
    input_fingerprint = sha256(
        _canonical_json(normalized_input).encode("utf-8")
    ).hexdigest()
    return {
        "schemaVersion": PERSON_DETECTION_CACHE_SCHEMA_VERSION,
        "frameContentSha256": frame_digest,
        "detectorInputFingerprint": input_fingerprint,
        # Retain full provenance so a hit can be audited without guessing what
        # a digest represented at the time it was written.
        "detectorInput": normalized_input,
    }


def person_detection_cache_key(contract: Mapping[str, Any]) -> str:
    normalized = _json_value(dict(contract), label="cache contract")
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def person_detection_cache_path(
    asset_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = person_detection_cache_key(contract)
    return (
        Path(asset_directory).expanduser().resolve()
        / "person-detections"
        / key[:2]
        / f"{key}.json"
    )


@dataclass(frozen=True, slots=True)
class PersonDetectionCacheEntry:
    cache_key: str
    path: Path
    frame_sha256: str
    image_size: tuple[int, int]
    people: tuple[dict[str, Any], ...]
    legacy_ball_candidates: tuple[dict[str, Any], ...]

    def as_pipeline_data(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
        """Return detached mutable payloads for one reconstruction invocation."""

        return (
            deepcopy(list(self.people)),
            deepcopy(list(self.legacy_ball_candidates)),
            tuple(self.image_size),
        )


@dataclass(frozen=True, slots=True)
class PersonDetectionCacheLookup:
    entry: PersonDetectionCacheEntry | None
    status: str
    error: str | None = None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _valid_payload(payload: Mapping[str, Any]) -> bool:
    if payload.get("providerStatus") != "primary-complete":
        return False
    image_size = payload.get("imageSize")
    people = payload.get("people")
    balls = payload.get("legacyBallCandidates")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in image_size)
        or not isinstance(people, list)
        or not isinstance(balls, list)
    ):
        return False
    width, height = image_size
    for person in people:
        if not isinstance(person, Mapping):
            return False
        x = _finite_number(person.get("x"))
        y = _finite_number(person.get("y"))
        box_width = _finite_number(person.get("width"))
        box_height = _finite_number(person.get("height"))
        confidence = _finite_number(person.get("confidence"))
        feature = person.get("feature")
        if (
            x is None
            or y is None
            or box_width is None
            or box_height is None
            or confidence is None
            or box_width <= 0.0
            or box_height <= 0.0
            or not 0.0 <= confidence <= 1.0
            or x < -box_width
            or x > width + box_width
            or y < -box_height
            or y > height + box_height
            or not isinstance(feature, list)
            or not feature
            or any(_finite_number(value) is None for value in feature)
        ):
            return False
    for ball in balls:
        if not isinstance(ball, Mapping):
            return False
        x = _finite_number(ball.get("x"))
        y = _finite_number(ball.get("y"))
        confidence = _finite_number(ball.get("confidence"))
        if (
            x is None
            or y is None
            or confidence is None
            or not 0.0 <= confidence <= 1.0
            or not -1.0 <= x <= width + 1.0
            or not -1.0 <= y <= height + 1.0
        ):
            return False
    return True


def _entry_from_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
    expected_key: str,
    path: Path,
) -> PersonDetectionCacheEntry | None:
    if envelope.get("schemaVersion") != PERSON_DETECTION_CACHE_SCHEMA_VERSION:
        return None
    if envelope.get("cacheKey") != expected_key:
        return None
    if envelope.get("contract") != expected_contract:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return None
    try:
        payload_fingerprint = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return None
    if envelope.get("payloadFingerprint") != payload_fingerprint:
        return None
    if not _valid_payload(payload):
        return None
    image_size = payload["imageSize"]
    assert isinstance(image_size, list)
    return PersonDetectionCacheEntry(
        cache_key=expected_key,
        path=path,
        frame_sha256=str(expected_contract["frameContentSha256"]),
        image_size=(int(image_size[0]), int(image_size[1])),
        people=tuple(deepcopy(payload["people"])),
        legacy_ball_candidates=tuple(deepcopy(payload["legacyBallCandidates"])),
    )


def lookup_person_detection_cache(
    asset_directory: str | Path,
    *,
    frame_sha256: str,
    detector_input: Mapping[str, Any],
) -> PersonDetectionCacheLookup:
    """Return hit/absent/corrupt/error without making cache faults fatal."""

    contract = build_person_detection_cache_contract(
        frame_sha256=frame_sha256,
        detector_input=detector_input,
    )
    key = person_detection_cache_key(contract)
    path = person_detection_cache_path(asset_directory, contract)
    if not path.exists():
        return PersonDetectionCacheLookup(None, "absent")
    if not path.is_file():
        return PersonDetectionCacheLookup(None, "error", "cache path is not a file")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return PersonDetectionCacheLookup(None, "corrupt", str(exc))
    except OSError as exc:
        return PersonDetectionCacheLookup(None, "error", str(exc))
    if not isinstance(envelope, Mapping):
        return PersonDetectionCacheLookup(None, "corrupt", "envelope is not an object")
    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:
        return PersonDetectionCacheLookup(
            None,
            "corrupt",
            "contract, checksum or payload validation failed",
        )
    return PersonDetectionCacheLookup(entry, "hit")


def store_person_detection_cache(
    asset_directory: str | Path,
    *,
    frame_sha256: str,
    detector_input: Mapping[str, Any],
    image_size: tuple[int, int],
    people: Sequence[Mapping[str, Any]],
    legacy_ball_candidates: Sequence[Mapping[str, Any]],
    provider_status: str = "primary-complete",
) -> PersonDetectionCacheEntry | None:
    """Atomically publish one complete primary frame result.

    Explicitly degraded/partial/fallback statuses return ``None`` and leave an
    existing good artifact untouched.
    """

    if provider_status != "primary-complete":
        return None
    contract = build_person_detection_cache_contract(
        frame_sha256=frame_sha256,
        detector_input=detector_input,
    )
    key = person_detection_cache_key(contract)
    payload = _json_value(
        {
            "providerStatus": provider_status,
            "imageSize": [int(image_size[0]), int(image_size[1])],
            "people": [dict(item) for item in people],
            "legacyBallCandidates": [dict(item) for item in legacy_ball_candidates],
        },
        label="cache payload",
    )
    if not _valid_payload(payload):
        raise PersonDetectionCacheError(
            "person detection payload is incomplete or invalid"
        )
    envelope = {
        "schemaVersion": PERSON_DETECTION_CACHE_SCHEMA_VERSION,
        "cacheKey": key,
        "contract": contract,
        "payloadFingerprint": sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "payload": payload,
    }
    path = person_detection_cache_path(asset_directory, contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Make the rename durable where the platform supports directory fsync.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise PersonDetectionCacheError(
            f"could not publish person detection cache {path}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    entry = _entry_from_envelope(
        envelope,
        expected_contract=contract,
        expected_key=key,
        path=path,
    )
    if entry is None:  # pragma: no cover - defensive invariant
        raise PersonDetectionCacheError(
            "published person detection cache failed validation"
        )
    return entry
