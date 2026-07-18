from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ball_detection_contract import BallCandidate


@dataclass(frozen=True, slots=True)
class ImageTile:
    index: int
    x: int
    y: int
    width: int
    height: int


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(result: Any, class_id: int) -> str | None:
    names = getattr(result, "names", None)
    if isinstance(names, Mapping):
        value = names.get(class_id)
        return str(value) if value is not None else None
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return None


def parse_ultralytics_ball_candidates(
    result: Any,
    *,
    backend_name: str,
    class_ids: Sequence[int] | None,
    offset: tuple[int, int] = (0, 0),
    full_image_size: tuple[int, int] | None = None,
    tile: ImageTile | None = None,
    frame_index: int | None = None,
    timestamp: float | None = None,
) -> list[BallCandidate]:
    """Parse an Ultralytics result without importing the runtime package."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
    confidences = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
    count = min(len(xyxy), len(confidences), len(classes))
    accepted_classes = (
        None if class_ids is None else {int(value) for value in class_ids}
    )
    offset_x, offset_y = offset
    candidates: list[BallCandidate] = []

    if full_image_size is None:
        image = getattr(result, "orig_img", None)
        if image is not None and getattr(image, "ndim", 0) >= 2:
            full_image_size = (int(image.shape[1]), int(image.shape[0]))

    for detection_index in range(count):
        class_id = int(classes[detection_index])
        if accepted_classes is not None and class_id not in accepted_classes:
            continue
        coordinates = xyxy[detection_index].astype(float)
        confidence = float(confidences[detection_index])
        if not np.isfinite(coordinates).all() or not np.isfinite(confidence):
            continue
        x1, y1, x2, y2 = coordinates
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y
        if full_image_size is not None:
            width, height = full_image_size
            x1, x2 = np.clip((x1, x2), 0.0, float(width))
            y1, y2 = np.clip((y1, y2), 0.0, float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        metadata: dict[str, Any] = {
            "detectionIndex": detection_index,
            "className": _class_name(result, class_id),
        }
        if frame_index is not None:
            metadata["frameIndex"] = frame_index
        if timestamp is not None:
            metadata["timestamp"] = timestamp
        if tile is not None:
            metadata["tile"] = {
                "index": tile.index,
                "x": tile.x,
                "y": tile.y,
                "width": tile.width,
                "height": tile.height,
            }
        candidates.append(
            BallCandidate(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=confidence,
                backend=backend_name,
                class_id=class_id,
                metadata=metadata,
            )
        )
    return candidates


def _intersection_over_union(
    left: BallCandidate,
    right: BallCandidate,
) -> float:
    x1 = max(left.bbox[0], right.bbox[0])
    y1 = max(left.bbox[1], right.bbox[1])
    x2 = min(left.bbox[2], right.bbox[2])
    y2 = min(left.bbox[3], right.bbox[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = left.width * left.height
    right_area = right.width * right.height
    return intersection / max(left_area + right_area - intersection, 1e-9)


def select_ball_candidates(
    candidates: Sequence[BallCandidate],
    *,
    max_candidates: int,
    nms_iou: float,
) -> tuple[BallCandidate, ...]:
    """Apply confidence ordering, class-agnostic NMS, and a global cap."""

    selected: list[BallCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        if any(
            _intersection_over_union(candidate, kept) > nms_iou
            for kept in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return tuple(selected)
