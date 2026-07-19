"""Content-addressed store of per-observation person crops.

The sampled-frame detection pass is the only place that decodes frame
pixels. Crops for every person observation are cut there once — with the
ReID padding policy — and published as one atomic per-frame envelope. The
ReID client and jersey OCR later read crop bytes from this store instead of
decoding frames again, and each crop's content digest is the cache key for
downstream model results. Missing, corrupt or tampered envelopes are
ordinary misses: the next detection pass rebuilds them from pixels.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .reconstruction_person_detection_contract import Detection


PERSON_CROP_STORE_SCHEMA_VERSION = 1
PERSON_CROP_ALGORITHM = "person-crop-v1"
_CROP_JPEG_QUALITY = 95


class PersonCropStoreError(RuntimeError):
    """A store contract or atomic publication could not be completed."""


@dataclass(frozen=True, slots=True)
class PersonCropPolicy:
    """Crop geometry and QA thresholds applied at extraction time."""

    padding_ratio: float = 0.08
    minimum_width: int = 16
    minimum_height: int = 30
    minimum_sharpness: float = 12.0

    def contract(self) -> dict[str, Any]:
        return {
            "paddingRatio": float(self.padding_ratio),
            "minimumWidth": int(self.minimum_width),
            "minimumHeight": int(self.minimum_height),
            "minimumSharpness": float(self.minimum_sharpness),
        }


@dataclass(frozen=True, slots=True)
class PersonCropRecord:
    observation_id: str
    crop_sha256: str
    crop_jpeg: bytes
    bbox: dict[str, float]
    padded_rect: tuple[int, int, int, int]
    quality: dict[str, Any]
    rejection_reasons: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_person_crop_contract(
    *,
    frame_sha256: str,
    policy: PersonCropPolicy,
) -> dict[str, Any]:
    frame_digest = str(frame_sha256).strip().lower()
    if len(frame_digest) != 64 or any(
        character not in "0123456789abcdef" for character in frame_digest
    ):
        raise PersonCropStoreError("frame_sha256 must be a SHA-256 hex digest")
    return {
        "schemaVersion": PERSON_CROP_STORE_SCHEMA_VERSION,
        "frameContentSha256": frame_digest,
        "algorithm": PERSON_CROP_ALGORITHM,
        "policy": policy.contract(),
    }


def person_crop_envelope_path(
    store_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()
    return Path(store_directory).expanduser().resolve() / key[:2] / f"{key}.json"


def detection_crop_bbox(detection: Detection) -> dict[str, float]:
    """Frozen detector-space bbox, identical to the tracker's point bbox."""

    image_x = (
        detection.image_x if detection.image_x is not None else detection.x
    )
    image_y = (
        detection.image_y if detection.image_y is not None else detection.y
    )
    return {
        "x": float(image_x) - detection.width / 2,
        "y": float(image_y) - detection.height,
        "width": float(detection.width),
        "height": float(detection.height),
    }


def extract_person_crop(
    image: np.ndarray,
    bbox: Mapping[str, float],
    policy: PersonCropPolicy,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any], list[str]]:
    """Cut one padded crop with the same geometry the ReID worker used."""

    height, width = image.shape[:2]
    x = float(bbox["x"])
    y = float(bbox["y"])
    box_width = float(bbox["width"])
    box_height = float(bbox["height"])
    padding_x = box_width * policy.padding_ratio
    padding_y = box_height * policy.padding_ratio
    requested = (
        int(np.floor(x - padding_x)),
        int(np.floor(y - padding_y)),
        int(np.ceil(x + box_width + padding_x)),
        int(np.ceil(y + box_height + padding_y)),
    )
    x1 = min(width, max(0, requested[0]))
    y1 = min(height, max(0, requested[1]))
    x2 = min(width, max(0, requested[2]))
    y2 = min(height, max(0, requested[3]))
    crop = image[y1:y2, x1:x2]
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if box_width < policy.minimum_width or box_height < policy.minimum_height:
        reasons.append("crop-too-small")
    if not crop.size:
        reasons.append("crop-outside-frame")
    elif sharpness < policy.minimum_sharpness:
        reasons.append("crop-too-blurry")
    quality = {
        "cropWidth": int(max(0, x2 - x1)),
        "cropHeight": int(max(0, y2 - y1)),
        "sourceBoxWidth": round(box_width, 3),
        "sourceBoxHeight": round(box_height, 3),
        "borderClipped": requested != (x1, y1, x2, y2),
        "sharpness": round(sharpness, 4),
    }
    return crop, (x1, y1, x2, y2), quality, reasons


def _record_payload(record: PersonCropRecord) -> dict[str, Any]:
    return {
        "cropSha256": record.crop_sha256,
        "cropJpegBase64": base64.b64encode(record.crop_jpeg).decode("ascii"),
        "bbox": {key: float(value) for key, value in record.bbox.items()},
        "paddedRect": list(record.padded_rect),
        "quality": record.quality,
        "rejectionReasons": list(record.rejection_reasons),
    }


def _record_from_payload(
    observation_id: str, payload: Mapping[str, Any]
) -> PersonCropRecord | None:
    encoded = payload.get("cropJpegBase64")
    digest = payload.get("cropSha256")
    bbox = payload.get("bbox")
    rect = payload.get("paddedRect")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        return None
    if not isinstance(bbox, Mapping) or not isinstance(rect, list):
        return None
    try:
        crop_jpeg = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        return None
    if sha256(crop_jpeg).hexdigest() != digest:
        return None
    if len(rect) != 4 or not all(isinstance(value, int) for value in rect):
        return None
    return PersonCropRecord(
        observation_id=observation_id,
        crop_sha256=digest,
        crop_jpeg=crop_jpeg,
        bbox={key: float(value) for key, value in bbox.items()},
        padded_rect=(rect[0], rect[1], rect[2], rect[3]),
        quality=dict(payload.get("quality") or {}),
        rejection_reasons=tuple(
            str(reason) for reason in payload.get("rejectionReasons") or ()
        ),
    )


def lookup_person_crop_envelope(
    store_directory: str | Path,
    *,
    frame_sha256: str,
    policy: PersonCropPolicy,
) -> dict[str, PersonCropRecord] | None:
    """Return all crop records of one frame, or None on any fault."""

    try:
        contract = build_person_crop_contract(
            frame_sha256=frame_sha256, policy=policy
        )
        path = person_crop_envelope_path(store_directory, contract)
        if not path.is_file():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (PersonCropStoreError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, Mapping) or envelope.get("contract") != contract:
        return None
    crops = envelope.get("crops")
    if not isinstance(crops, Mapping):
        return None
    records: dict[str, PersonCropRecord] = {}
    for observation_id, payload in crops.items():
        if not isinstance(payload, Mapping):
            return None
        record = _record_from_payload(str(observation_id), payload)
        if record is None:
            return None
        records[str(observation_id)] = record
    return records


def store_person_crop_envelope(
    store_directory: str | Path,
    *,
    frame_sha256: str,
    policy: PersonCropPolicy,
    records: Mapping[str, PersonCropRecord],
) -> None:
    """Atomically publish every crop of one frame as a single envelope."""

    contract = build_person_crop_contract(frame_sha256=frame_sha256, policy=policy)
    envelope = {
        "contract": contract,
        "crops": {
            observation_id: _record_payload(record)
            for observation_id, record in sorted(records.items())
        },
    }
    path = person_crop_envelope_path(store_directory, contract)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise PersonCropStoreError(
            f"could not publish person crop envelope {path}: {exc}"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass


def extract_and_store_person_crops(
    store_directory: str | Path,
    *,
    image: np.ndarray,
    frame_sha256: str,
    detections: Sequence[Detection],
    policy: PersonCropPolicy,
    diagnostics: dict | None = None,
) -> dict[str, PersonCropRecord]:
    """Reuse the stored envelope when it covers every current observation,
    otherwise cut crops from the already-decoded image and publish them.

    Store IO failures degrade to explicit `crop-store-unavailable` records so
    a full-disk situation cannot fail an otherwise healthy detection pass.
    """

    wanted = [
        detection
        for detection in detections
        if detection.observation_id
    ]
    stored = lookup_person_crop_envelope(
        store_directory, frame_sha256=frame_sha256, policy=policy
    )
    if stored is not None and all(
        str(detection.observation_id) in stored for detection in wanted
    ):
        if diagnostics is not None:
            diagnostics["hits"] = int(diagnostics.get("hits", 0)) + 1
        return stored

    records: dict[str, PersonCropRecord] = {}
    for detection in wanted:
        bbox = detection_crop_bbox(detection)
        crop, rect, quality, reasons = extract_person_crop(image, bbox, policy)
        crop_jpeg = b""
        digest = ""
        if crop.size:
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), _CROP_JPEG_QUALITY],
            )
            if encoded_ok:
                crop_jpeg = encoded.tobytes()
                digest = sha256(crop_jpeg).hexdigest()
            else:
                reasons = [*reasons, "crop-encode-failed"]
        records[str(detection.observation_id)] = PersonCropRecord(
            observation_id=str(detection.observation_id),
            crop_sha256=digest,
            crop_jpeg=crop_jpeg,
            bbox=bbox,
            padded_rect=rect,
            quality=quality,
            rejection_reasons=tuple(reasons),
        )
    try:
        store_person_crop_envelope(
            store_directory,
            frame_sha256=frame_sha256,
            policy=policy,
            records=records,
        )
        if diagnostics is not None:
            diagnostics["stores"] = int(diagnostics.get("stores", 0)) + 1
    except PersonCropStoreError:
        if diagnostics is not None:
            diagnostics["storeErrors"] = (
                int(diagnostics.get("storeErrors", 0)) + 1
            )
        records = {
            observation_id: PersonCropRecord(
                observation_id=record.observation_id,
                crop_sha256="",
                crop_jpeg=b"",
                bbox=record.bbox,
                padded_rect=record.padded_rect,
                quality=record.quality,
                rejection_reasons=(
                    *record.rejection_reasons,
                    "crop-store-unavailable",
                ),
            )
            for observation_id, record in records.items()
        }
    return records


def person_crop_store_runtime() -> tuple[Path, PersonCropPolicy]:
    """The configured store directory and extraction policy of this deploy."""

    from .config import get_settings

    settings = get_settings()
    return (
        Path(settings.media_root) / "person-crops",
        PersonCropPolicy(
            padding_ratio=float(settings.person_crop_padding_ratio),
            minimum_width=int(settings.person_crop_minimum_width),
            minimum_height=int(settings.person_crop_minimum_height),
            minimum_sharpness=float(settings.person_crop_minimum_sharpness),
        ),
    )


def attach_crop_records(
    detections: Sequence[Detection],
    records: Mapping[str, PersonCropRecord],
    *,
    frame_sha256: str,
) -> None:
    """Stamp each detection with its crop identity for later phases."""

    for detection in detections:
        record = records.get(str(detection.observation_id or ""))
        if record is None:
            continue
        detection.crop_frame_sha256 = frame_sha256
        detection.crop_sha256 = record.crop_sha256 or None
        detection.crop_quality = dict(record.quality)
        detection.crop_rejection_reasons = tuple(record.rejection_reasons)


__all__ = (
    "PERSON_CROP_ALGORITHM",
    "PERSON_CROP_STORE_SCHEMA_VERSION",
    "PersonCropPolicy",
    "PersonCropRecord",
    "PersonCropStoreError",
    "attach_crop_records",
    "build_person_crop_contract",
    "detection_crop_bbox",
    "extract_and_store_person_crops",
    "extract_person_crop",
    "lookup_person_crop_envelope",
    "person_crop_envelope_path",
    "person_crop_store_runtime",
    "store_person_crop_envelope",
)
