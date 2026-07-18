from __future__ import annotations

"""Sampled-frame lookup and image-space target normalization for annotations."""

import cv2

from .reconstruction_errors import ReconstructionError
from .reconstruction_frame_annotation_contract import FrameAnnotationTarget
from .reconstruction_inputs import frame_paths


def resolve_frame_annotation_target(
    scene: dict,
    *,
    scene_time: float,
    bbox: dict,
) -> FrameAnnotationTarget:
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_path, frame_time = min(
        frames,
        key=lambda item: abs(item[1] - float(scene_time)),
    )
    image = cv2.imread(str(target_path))
    if image is None:
        raise ReconstructionError("The sampled frame could not be read")
    frame_height, frame_width = image.shape[:2]
    x = min(max(0.0, float(bbox["x"])), frame_width - 4.0)
    y = min(max(0.0, float(bbox["y"])), frame_height - 4.0)
    width = min(float(bbox["width"]), frame_width - x)
    height = min(float(bbox["height"]), frame_height - y)
    if width < 4 or height < 4:
        raise ReconstructionError("The person box is outside the video frame")
    return FrameAnnotationTarget(
        path=target_path,
        scene_time=float(frame_time),
        frame_index=int(target_path.stem.split("_")[-1]),
        x=x,
        y=y,
        width=width,
        height=height,
    )
