from __future__ import annotations

from pathlib import Path

import numpy as np

from .ball_detection_contract import (
    BallDetectionError,
    BallDetectorUnavailable,
    FrameInput,
)


def frame_array(frame: FrameInput) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        if frame.ndim < 2:
            raise BallDetectionError(
                "frame array must have at least two dimensions"
            )
        return frame
    path = Path(frame).expanduser()
    if not path.is_file():
        raise BallDetectionError(f"frame does not exist: {path}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - broken runtime only
        raise BallDetectorUnavailable(
            "OpenCV is required to decode frame files"
        ) from exc
    image = cv2.imread(str(path))
    if image is None:
        raise BallDetectionError(f"frame could not be decoded: {path}")
    return image
