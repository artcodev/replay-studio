from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .ball_detection_contract import (
    BallDetectionError,
    BallDetectorUnavailable,
    FrameInput,
)
from .wasb_ball_protocol import WasbMultipartRequest


def wasb_http_transport(
    url: str,
    request: WasbMultipartRequest,
    timeout: float,
) -> Mapping[str, Any]:
    """Send the canonical WASB multipart request without base64 expansion."""

    files = [
        ("frames", _multipart_frame(frame, index=index))
        for index, frame in enumerate(request.uploads)
    ]
    try:
        response = httpx.post(
            url,
            files=files,
            data={
                "manifest": json.dumps(
                    request.manifest,
                    separators=(",", ":"),
                )
            },
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        decoded = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise BallDetectorUnavailable(
            f"WASB service request failed: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise BallDetectorUnavailable(
            "WASB service returned a non-object JSON response"
        )
    return decoded


def _multipart_frame(
    frame: FrameInput,
    *,
    index: int,
) -> tuple[str, bytes, str]:
    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            raise BallDetectionError(
                "WASB multipart frames must use uint8 pixels"
            )
        if frame.ndim not in (2, 3) or (
            frame.ndim == 3 and frame.shape[2] not in (1, 3, 4)
        ):
            raise BallDetectionError(
                "WASB multipart frames must have 1, 3, or 4 channels"
            )
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - broken runtime only
            raise BallDetectorUnavailable(
                "OpenCV is required to encode WASB multipart frames"
            ) from exc
        encoded, buffer = cv2.imencode(".png", np.ascontiguousarray(frame))
        if not encoded:
            raise BallDetectionError("WASB frame could not be encoded as PNG")
        return f"frame-{index:03d}.png", buffer.tobytes(), "image/png"

    path = Path(frame).expanduser().resolve()
    if not path.is_file():
        raise BallDetectionError(f"frame does not exist: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.name, path.read_bytes(), media_type
