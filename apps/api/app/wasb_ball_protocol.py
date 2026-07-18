from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ball_candidate_selection import select_ball_candidates
from .ball_detection_contract import (
    BallCandidate,
    BallDetectorUnavailable,
    FrameInput,
)


CONTRACT_VERSION = 1
WASB_WORKER_BACKEND = "wasb-sbdt-soccer"


@dataclass(frozen=True, slots=True)
class WasbMultipartRequest:
    """Provider request before HTTP wire encoding.

    ``uploads`` contains each image once. ``manifest.frames`` may reference an
    upload more than once to express temporal edge padding without copying the
    image bytes.
    """

    uploads: tuple[FrameInput, ...]
    manifest: Mapping[str, Any]
    target_index: int
    context_mode: str


@dataclass(frozen=True, slots=True)
class WasbTargetResponse:
    image_size: tuple[int, int]
    candidates: tuple[BallCandidate, ...]
    worker_metadata: Mapping[str, Any]


def build_wasb_multipart_request(
    frame: FrameInput,
    context_frames: Sequence[FrameInput],
    *,
    frame_index: int | None,
    timestamp: float | None,
    max_candidates: int,
) -> WasbMultipartRequest:
    """Build the exact three-frame window expected by WASB-SBDT."""

    if len(context_frames) >= 2:
        uploads = (context_frames[0], frame, context_frames[-1])
        file_indices = (0, 1, 2)
        target_index = 1
        context_mode = "centered"
    elif context_frames:
        uploads = (context_frames[0], frame)
        file_indices = (0, 0, 1)
        target_index = 2
        context_mode = "causal"
    else:
        uploads = (frame,)
        file_indices = (0, 0, 0)
        target_index = 1
        context_mode = "edge-repeat"

    target_frame_index = frame_index if frame_index is not None else target_index
    if target_frame_index < 0:
        raise BallDetectorUnavailable("WASB frame_index must not be negative")
    logical_frame_indices = _logical_frame_indices(
        target_frame_index,
        context_mode=context_mode,
    )
    manifest_frames: list[dict[str, Any]] = []
    for logical_index, (file_index, source_frame_index) in enumerate(
        zip(file_indices, logical_frame_indices, strict=True)
    ):
        item: dict[str, Any] = {
            "fileIndex": file_index,
            "frameIndex": source_frame_index,
        }
        if logical_index == target_index and timestamp is not None:
            item["timestamp"] = float(timestamp)
        manifest_frames.append(item)

    return WasbMultipartRequest(
        uploads=uploads,
        manifest={
            "contractVersion": CONTRACT_VERSION,
            "maxCandidates": max_candidates,
            "targetIndex": target_index,
            "frames": manifest_frames,
        },
        target_index=target_index,
        context_mode=context_mode,
    )


def _logical_frame_indices(
    target_frame_index: int,
    *,
    context_mode: str,
) -> tuple[int, int, int]:
    previous = max(0, target_frame_index - 1)
    if context_mode == "centered":
        return previous, target_frame_index, target_frame_index + 1
    if context_mode == "causal":
        return previous, previous, target_frame_index
    return target_frame_index, target_frame_index, target_frame_index


def parse_wasb_target_response(
    response: Mapping[str, Any],
    *,
    request: WasbMultipartRequest,
    backend_name: str,
    max_candidates: int,
    nms_iou: float,
    frame_index: int | None,
    timestamp: float | None,
) -> WasbTargetResponse:
    """Validate the canonical batch response and select the target frame."""

    if response.get("contractVersion") != CONTRACT_VERSION:
        raise BallDetectorUnavailable(
            "WASB service returned an unsupported contractVersion"
        )
    if response.get("backend") != WASB_WORKER_BACKEND:
        raise BallDetectorUnavailable(
            "WASB service returned an unexpected backend"
        )
    model_version = response.get("modelVersion")
    if not isinstance(model_version, str) or not model_version:
        raise BallDetectorUnavailable(
            "WASB service response has no modelVersion"
        )
    raw_frames = response.get("frames")
    expected_frames = request.manifest["frames"]
    if (
        not isinstance(raw_frames, Sequence)
        or isinstance(raw_frames, (str, bytes))
        or len(raw_frames) != len(expected_frames)
    ):
        raise BallDetectorUnavailable(
            "WASB service returned an invalid frames array"
        )
    raw_target = raw_frames[request.target_index]
    if not isinstance(raw_target, Mapping):
        raise BallDetectorUnavailable(
            "WASB service target frame must be an object"
        )
    expected_target = expected_frames[request.target_index]
    if raw_target.get("fileIndex") != expected_target["fileIndex"]:
        raise BallDetectorUnavailable(
            "WASB service target fileIndex does not match the request"
        )
    if raw_target.get("frameIndex") != expected_target["frameIndex"]:
        raise BallDetectorUnavailable(
            "WASB service target frameIndex does not match the request"
        )
    image_size = _strict_image_size(raw_target.get("imageSize"))
    candidates = _parse_wasb_candidates(
        raw_target.get("candidates"),
        backend_name=backend_name,
        image_size=image_size,
        max_candidates=max_candidates,
        nms_iou=nms_iou,
        frame_index=frame_index,
        timestamp=timestamp,
        model_version=model_version,
    )
    metadata = response.get("metadata")
    if not isinstance(metadata, Mapping):
        raise BallDetectorUnavailable(
            "WASB service response metadata must be an object"
        )
    target_padding = raw_target.get("temporalPadding")
    if not isinstance(target_padding, bool):
        raise BallDetectorUnavailable(
            "WASB service target temporalPadding must be a boolean"
        )
    return WasbTargetResponse(
        image_size=image_size,
        candidates=candidates,
        worker_metadata={
            **dict(metadata),
            "backend": WASB_WORKER_BACKEND,
            "modelVersion": model_version,
            "targetIndex": request.target_index,
            "temporalPadding": target_padding,
        },
    )


def _strict_image_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise BallDetectorUnavailable(
            "WASB service target imageSize must contain two integers"
        )
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise BallDetectorUnavailable(
            "WASB service target imageSize must be positive"
        )
    return width, height


def _parse_wasb_candidates(
    raw_items: Any,
    *,
    backend_name: str,
    image_size: tuple[int, int],
    max_candidates: int,
    nms_iou: float,
    frame_index: int | None,
    timestamp: float | None,
    model_version: str,
) -> tuple[BallCandidate, ...]:
    if not isinstance(raw_items, list):
        raise BallDetectorUnavailable(
            "WASB service target response has no candidates array"
        )
    if len(raw_items) > max_candidates:
        raise BallDetectorUnavailable(
            "WASB service returned more candidates than requested"
        )
    width, height = image_size
    candidates: list[BallCandidate] = []
    for index, value in enumerate(raw_items):
        if not isinstance(value, Mapping):
            raise BallDetectorUnavailable(
                f"WASB service candidate {index} must be an object"
            )
        try:
            x1, y1, x2, y2 = _external_bbox(value)
            confidence = float(value["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BallDetectorUnavailable(
                f"WASB service candidate {index} is invalid"
            ) from exc
        values = np.asarray((x1, y1, x2, y2, confidence), dtype=np.float64)
        if not np.isfinite(values).all() or not 0.0 <= confidence <= 1.0:
            raise BallDetectorUnavailable(
                f"WASB service candidate {index} has non-finite values"
            )
        x1, x2 = np.clip((x1, x2), 0.0, float(width))
        y1, y2 = np.clip((y1, y2), 0.0, float(height))
        if x2 <= x1 or y2 <= y1:
            raise BallDetectorUnavailable(
                f"WASB service candidate {index} has an empty bounding box"
            )
        candidate_model = value.get("modelVersion")
        if candidate_model is not None and candidate_model != model_version:
            raise BallDetectorUnavailable(
                f"WASB service candidate {index} modelVersion is inconsistent"
            )
        metadata: dict[str, Any] = {
            "detectionIndex": index,
            "observed": bool(value.get("observed", True)),
        }
        if frame_index is not None:
            metadata["frameIndex"] = frame_index
        if timestamp is not None:
            metadata["timestamp"] = timestamp
        for key in (
            "heatmapPeak",
            "temporalScore",
            "componentScore",
            "componentArea",
            "occluded",
            "sourceFrameIndex",
        ):
            if key in value:
                metadata[key] = value[key]
        candidates.append(
            BallCandidate(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=confidence,
                backend=backend_name,
                metadata=metadata,
            )
        )
    return select_ball_candidates(
        candidates,
        max_candidates=max_candidates,
        nms_iou=nms_iou,
    )


def _external_bbox(
    item: Mapping[str, Any],
    default_radius: float = 4.0,
) -> tuple[float, float, float, float]:
    bbox = item.get("bbox", item.get("box"))
    if (
        isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes))
        and len(bbox) == 4
    ):
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    position = item.get("position")
    if (
        isinstance(position, Sequence)
        and not isinstance(position, (str, bytes))
        and len(position) == 2
    ):
        x, y = float(position[0]), float(position[1])
    else:
        x, y = float(item["x"]), float(item["y"])
    radius = float(item.get("radius", default_radius))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    return x - radius, y - radius, x + radius, y + radius
