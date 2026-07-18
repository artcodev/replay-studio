from __future__ import annotations

"""Resolve one sampled video frame and its accumulated camera transform."""

import cv2
import numpy as np

from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import frame_paths
from .reconstruction_motion import camera_motion_estimate


def calibration_frame_context(
    scene: dict,
    scene_time: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_index = min(
        range(len(frames)),
        key=lambda index: abs(frames[index][1] - scene_time),
    )
    previous_image: np.ndarray | None = None
    target_image: np.ndarray | None = None
    camera_transform = np.eye(3, dtype=np.float64)
    for index, (path, _) in enumerate(frames[: target_index + 1]):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if previous_image is not None:
            motion = camera_motion_estimate(previous_image, image)
            camera_transform = (
                camera_transform @ motion.matrix
                if motion.reliable
                else np.eye(3, dtype=np.float64)
            )
        previous_image = image
        if index == target_index:
            target_image = image
    if target_image is None:
        raise ReconstructionError("The sampled frame could not be read")
    return (
        target_index,
        float(frames[target_index][1]),
        target_image,
        camera_transform,
    )
