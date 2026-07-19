"""Content-addressed disk cache for observed pitch-line masks.

Temporal calibration validation needs exactly one thing from a frame's
pixels: the binary mask of white line paint on the pitch. The mask is a pure
function of the frame bytes (no model involved), so caching it by content
hash removes the second full decode pass of a segment — both across repeated
runs and, once the detection pass warms the cache, inside the first run too.
Missing, corrupt or tampered artifacts are ordinary misses.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


PITCH_LINE_MASK_CACHE_SCHEMA_VERSION = 1
# Bump together with any change to pitch_line_mask so stale masks are misses.
PITCH_LINE_MASK_ALGORITHM = "pitch-line-mask-v1"


class PitchLineMaskCacheError(RuntimeError):
    """A cache contract or atomic publication could not be completed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_pitch_line_mask_contract(*, frame_sha256: str) -> dict[str, Any]:
    frame_digest = str(frame_sha256).strip().lower()
    if len(frame_digest) != 64 or any(
        character not in "0123456789abcdef" for character in frame_digest
    ):
        raise PitchLineMaskCacheError("frame_sha256 must be a SHA-256 hex digest")
    return {
        "schemaVersion": PITCH_LINE_MASK_CACHE_SCHEMA_VERSION,
        "frameContentSha256": frame_digest,
        "algorithm": PITCH_LINE_MASK_ALGORITHM,
    }


def pitch_line_mask_cache_path(
    cache_directory: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    key = sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()
    return Path(cache_directory).expanduser().resolve() / key[:2] / f"{key}.json"


@dataclass(frozen=True, slots=True)
class PitchLineMaskLookup:
    mask: np.ndarray | None
    status: str


def lookup_pitch_line_mask(
    cache_directory: str | Path,
    *,
    frame_sha256: str,
) -> PitchLineMaskLookup:
    """Return the cached binary mask, or an ordinary miss on any fault."""

    contract = build_pitch_line_mask_contract(frame_sha256=frame_sha256)
    path = pitch_line_mask_cache_path(cache_directory, contract)
    if not path.is_file():
        return PitchLineMaskLookup(None, "absent")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PitchLineMaskLookup(None, "corrupt")
    if not isinstance(envelope, Mapping) or envelope.get("contract") != contract:
        return PitchLineMaskLookup(None, "corrupt")
    encoded = envelope.get("maskPngBase64")
    if not isinstance(encoded, str):
        return PitchLineMaskLookup(None, "corrupt")
    try:
        png_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        return PitchLineMaskLookup(None, "corrupt")
    if envelope.get("maskSha256") != sha256(png_bytes).hexdigest():
        return PitchLineMaskLookup(None, "corrupt")
    mask = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if mask is None:
        return PitchLineMaskLookup(None, "corrupt")
    return PitchLineMaskLookup(mask, "hit")


def store_pitch_line_mask(
    cache_directory: str | Path,
    *,
    frame_sha256: str,
    mask: np.ndarray,
) -> None:
    """Atomically publish one binary mask; IO faults never propagate."""

    contract = build_pitch_line_mask_contract(frame_sha256=frame_sha256)
    encoded_ok, png = cv2.imencode(".png", mask)
    if not encoded_ok:
        raise PitchLineMaskCacheError("pitch line mask could not be encoded")
    png_bytes = png.tobytes()
    envelope = {
        "contract": contract,
        "maskSha256": sha256(png_bytes).hexdigest(),
        "maskPngBase64": base64.b64encode(png_bytes).decode("ascii"),
    }
    path = pitch_line_mask_cache_path(cache_directory, contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(envelope))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise PitchLineMaskCacheError(
            f"could not publish pitch line mask {path}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def cached_pitch_line_mask_loader(
    cache_directory: str | Path,
    *,
    enabled: bool,
):
    """Build a mask loader that reads through the content cache.

    A path whose bytes cannot be hashed (synthetic test paths, races) falls
    back to decode-and-compute transparently without touching the cache.
    """

    from .person_detection_cache import frame_content_sha256
    from .pitch_image_evidence import pitch_line_mask

    def load(path: str | Path) -> np.ndarray | None:
        digest: str | None = None
        if enabled:
            try:
                digest = frame_content_sha256(path)
                cached = lookup_pitch_line_mask(
                    cache_directory, frame_sha256=digest
                )
            except (PitchLineMaskCacheError, OSError):
                digest = None
            else:
                if cached.status == "hit":
                    return cached.mask
        image = cv2.imread(str(path))
        if image is None:
            return None
        mask = pitch_line_mask(image)
        if digest is not None:
            try:
                store_pitch_line_mask(
                    cache_directory, frame_sha256=digest, mask=mask
                )
            except (PitchLineMaskCacheError, OSError):
                pass
        return mask

    return load


__all__ = (
    "cached_pitch_line_mask_loader",
    "PITCH_LINE_MASK_ALGORITHM",
    "PITCH_LINE_MASK_CACHE_SCHEMA_VERSION",
    "PitchLineMaskCacheError",
    "PitchLineMaskLookup",
    "build_pitch_line_mask_contract",
    "lookup_pitch_line_mask",
    "pitch_line_mask_cache_path",
    "store_pitch_line_mask",
)
