"""Camera context accumulated up to a selected sampled frame."""

from pathlib import Path

import cv2
import numpy as np

from .reconstruction_motion import camera_motion_estimate


def camera_transform_to_frame(
    frames: list[tuple[Path, float]],
    target_index: int,
    target_image: np.ndarray,
) -> np.ndarray:
    previous_image: np.ndarray | None = None
    transform = np.eye(3, dtype=np.float64)
    for index, (path, _) in enumerate(frames[: target_index + 1]):
        image = target_image if index == target_index else cv2.imread(str(path))
        if image is None:
            continue
        if previous_image is not None:
            motion = camera_motion_estimate(previous_image, image)
            transform = (
                transform @ motion.matrix
                if motion.reliable
                else np.eye(3, dtype=np.float64)
            )
        previous_image = image
    return transform
