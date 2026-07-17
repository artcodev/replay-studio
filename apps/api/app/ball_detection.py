"""Pluggable football-ball detectors with a small, stable integration contract.

The module deliberately does not download model weights.  An Ultralytics model
must either be injected by the caller or loaded from an existing local
checkpoint.  WASB integration is isolated behind a JSON service/subprocess
contract so its legacy PyTorch runtime can live outside the API process.

External WASB workers receive JSON with ``frames`` and ``targetIndex``.  Each
frame is base64 encoded either as the original file bytes or as a contiguous
NumPy array.  They return, at minimum::

    {
      "imageSize": [1920, 1080],
      "candidates": [
        {"bbox": [100.0, 200.0, 110.0, 210.0], "confidence": 0.91}
      ]
    }

``position: [x, y]`` (or scalar ``x``/``y``) is also accepted when the worker
does not estimate a box.  The adapter then creates a small box around the
position.  This keeps the reconstruction-side contract identical for
single-frame YOLO and temporal heatmap detectors.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias
from urllib import request as urllib_request

import numpy as np


FrameInput: TypeAlias = str | Path | np.ndarray
FailurePolicy: TypeAlias = Literal["raise", "fallback"]
BackendName: TypeAlias = Literal[
    "generic-ultralytics",
    "dedicated-ultralytics",
    "wasb-service",
    "wasb-subprocess",
]


class BallDetectionError(RuntimeError):
    """Base error for the ball detector boundary."""


class BallDetectorConfigurationError(BallDetectionError):
    """Raised when a detector cannot be constructed safely."""


class BallDetectorUnavailable(BallDetectionError):
    """Raised when an external detector cannot produce a result."""


@dataclass(frozen=True, slots=True)
class BallCandidate:
    """One image-space ball hypothesis.

    Coordinates use the original, full-frame pixel coordinate system even
    when inference was performed on overlapping tiles.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    backend: str
    class_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def as_reconstruction_detection(self) -> dict[str, Any]:
        """Return the shape consumed by the existing reconstruction pipeline."""

        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "detectorBackend": self.backend,
            "detectorClassId": self.class_id,
            "detectorMetadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BallDetectionBatch:
    candidates: tuple[BallCandidate, ...]
    image_size: tuple[int, int]
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_reconstruction_detections(self) -> list[dict[str, Any]]:
        return [candidate.as_reconstruction_detection() for candidate in self.candidates]


class BallDetector(Protocol):
    backend_name: str

    def detect(
        self,
        frame: FrameInput,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch: ...


@dataclass(frozen=True, slots=True)
class UltralyticsBallDetectorConfig:
    backend_name: str = "generic-ultralytics"
    class_ids: tuple[int, ...] | None = (32,)
    confidence: float = 0.035
    image_size: int | None = 1280
    device: str | int | None = "cpu"
    max_candidates: int = 12
    tile_size: tuple[int, int] | None = None
    tile_overlap: float = 0.2
    inference_batch_size: int = 4
    nms_iou: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise BallDetectorConfigurationError("confidence must be between 0 and 1")
        if self.image_size is not None and self.image_size <= 0:
            raise BallDetectorConfigurationError("image_size must be positive")
        if self.max_candidates <= 0:
            raise BallDetectorConfigurationError("max_candidates must be positive")
        if self.tile_size is not None and any(value <= 0 for value in self.tile_size):
            raise BallDetectorConfigurationError("tile dimensions must be positive")
        if not 0.0 <= self.tile_overlap < 1.0:
            raise BallDetectorConfigurationError("tile_overlap must be in [0, 1)")
        if self.inference_batch_size <= 0:
            raise BallDetectorConfigurationError("inference_batch_size must be positive")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise BallDetectorConfigurationError("nms_iou must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class BallDetectorConfig:
    """Factory settings that can be populated from the API's environment."""

    backend: BackendName = "generic-ultralytics"
    checkpoint_path: str | Path | None = None
    device: str | int | None = "cpu"
    confidence: float = 0.035
    # None chooses the backend's reference resolution: 1280 for the current
    # generic COCO pass and 640 for Roboflow's tiled one-class checkpoint.
    image_size: int | None = None
    max_candidates: int = 12
    tile_size: tuple[int, int] = (640, 640)
    tile_overlap: float = 0.2
    inference_batch_size: int = 4
    nms_iou: float = 0.1
    wasb_service_url: str | None = None
    wasb_command: tuple[str, ...] = ()
    wasb_timeout: float = 30.0
    failure_policy: FailurePolicy = "raise"


@dataclass(frozen=True, slots=True)
class _Tile:
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
    tile: _Tile | None = None,
    frame_index: int | None = None,
    timestamp: float | None = None,
) -> list[BallCandidate]:
    """Parse an Ultralytics ``Results`` object without importing Ultralytics."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
    confidences = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
    count = min(len(xyxy), len(confidences), len(classes))
    accepted_classes = None if class_ids is None else {int(value) for value in class_ids}
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


def _intersection_over_union(left: BallCandidate, right: BallCandidate) -> float:
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
    """Apply confidence ordering, class-agnostic NMS, and a global top-K cap."""

    selected: list[BallCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(_intersection_over_union(candidate, kept) > nms_iou for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return tuple(selected)


def _axis_origins(length: int, tile_length: int, overlap: float) -> tuple[int, ...]:
    tile_length = min(tile_length, length)
    if tile_length == length:
        return (0,)
    step = max(1, int(round(tile_length * (1.0 - overlap))))
    origins = list(range(0, length - tile_length + 1, step))
    final_origin = length - tile_length
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return tuple(origins)


def _tiles(image_size: tuple[int, int], tile_size: tuple[int, int], overlap: float) -> tuple[_Tile, ...]:
    image_width, image_height = image_size
    tile_width, tile_height = tile_size
    tiles: list[_Tile] = []
    for y in _axis_origins(image_height, tile_height, overlap):
        for x in _axis_origins(image_width, tile_width, overlap):
            width = min(tile_width, image_width - x)
            height = min(tile_height, image_height - y)
            tiles.append(_Tile(index=len(tiles), x=x, y=y, width=width, height=height))
    return tuple(tiles)


def _frame_array(frame: FrameInput) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        if frame.ndim < 2:
            raise BallDetectionError("frame array must have at least two dimensions")
        return frame
    path = Path(frame).expanduser()
    if not path.is_file():
        raise BallDetectionError(f"frame does not exist: {path}")
    try:
        import cv2  # Ultralytics already depends on OpenCV; keep import optional here.
    except ImportError as exc:  # pragma: no cover - exercised only in a broken runtime
        raise BallDetectorUnavailable("OpenCV is required to tile frame files") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise BallDetectionError(f"frame could not be decoded: {path}")
    return image


def _result_image_size(result: Any, frame: FrameInput) -> tuple[int, int]:
    image = getattr(result, "orig_img", None)
    if image is not None and getattr(image, "ndim", 0) >= 2:
        return int(image.shape[1]), int(image.shape[0])
    if isinstance(frame, np.ndarray) and frame.ndim >= 2:
        return int(frame.shape[1]), int(frame.shape[0])
    image = _frame_array(frame)
    return int(image.shape[1]), int(image.shape[0])


def _default_model_loader(checkpoint: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - project dependency in production
        raise BallDetectorUnavailable("Ultralytics is not installed") from exc
    return YOLO(checkpoint)


class UltralyticsBallDetector:
    """Single-frame or tiled local Ultralytics detector."""

    def __init__(
        self,
        config: UltralyticsBallDetectorConfig,
        *,
        model: Any | None = None,
        checkpoint_path: str | Path | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.backend_name = config.backend_name
        if model is None:
            if checkpoint_path is None:
                raise BallDetectorConfigurationError(
                    "inject an Ultralytics model or provide an existing local checkpoint"
                )
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise BallDetectorConfigurationError(
                    f"local ball checkpoint does not exist: {checkpoint}"
                )
            model = (model_loader or _default_model_loader)(str(checkpoint))
        self._model = model

    def _prediction_arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "conf": self.config.confidence,
            "verbose": False,
        }
        if self.config.image_size is not None:
            arguments["imgsz"] = self.config.image_size
        if self.config.device is not None:
            arguments["device"] = self.config.device
        if self.config.class_ids is not None:
            arguments["classes"] = list(self.config.class_ids)
        return arguments

    def _predict(self, source: Any) -> Any:
        output = self._model.predict(source, **self._prediction_arguments())
        if not output:
            raise BallDetectorUnavailable(f"{self.backend_name} returned no Results object")
        return output[0]

    def _predict_batch(self, sources: Sequence[np.ndarray]) -> Sequence[Any]:
        output = self._model.predict(list(sources), **self._prediction_arguments())
        if len(output) != len(sources):
            raise BallDetectorUnavailable(
                f"{self.backend_name} returned {len(output)} results for {len(sources)} tiles"
            )
        return output

    def detect(
        self,
        frame: FrameInput,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch:
        del context_frames  # Temporal context is consumed only by temporal backends.
        if self.config.tile_size is None:
            source = str(frame) if isinstance(frame, Path) else frame
            result = self._predict(source)
            image_size = _result_image_size(result, frame)
            raw = parse_ultralytics_ball_candidates(
                result,
                backend_name=self.backend_name,
                class_ids=self.config.class_ids,
                full_image_size=image_size,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            candidates = select_ball_candidates(
                raw,
                max_candidates=self.config.max_candidates,
                nms_iou=self.config.nms_iou,
            )
            return BallDetectionBatch(
                candidates=candidates,
                image_size=image_size,
                backend=self.backend_name,
                metadata={"rawCandidateCount": len(raw), "tileCount": 1},
            )

        image = _frame_array(frame)
        image_size = (int(image.shape[1]), int(image.shape[0]))
        tiles = _tiles(image_size, self.config.tile_size, self.config.tile_overlap)
        raw: list[BallCandidate] = []
        batch_size = self.config.inference_batch_size
        inference_batch_count = 0
        for start in range(0, len(tiles), batch_size):
            tile_batch = tiles[start : start + batch_size]
            crops = [
                image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
                for tile in tile_batch
            ]
            results = self._predict_batch(crops)
            inference_batch_count += 1
            for tile, result in zip(tile_batch, results, strict=True):
                raw.extend(
                    parse_ultralytics_ball_candidates(
                        result,
                        backend_name=self.backend_name,
                        class_ids=self.config.class_ids,
                        offset=(tile.x, tile.y),
                        full_image_size=image_size,
                        tile=tile,
                        frame_index=frame_index,
                        timestamp=timestamp,
                    )
                )
        candidates = select_ball_candidates(
            raw,
            max_candidates=self.config.max_candidates,
            nms_iou=self.config.nms_iou,
        )
        return BallDetectionBatch(
            candidates=candidates,
            image_size=image_size,
            backend=self.backend_name,
            metadata={
                "rawCandidateCount": len(raw),
                "tileCount": len(tiles),
                "tileSize": list(self.config.tile_size),
                "tileOverlap": self.config.tile_overlap,
                "inferenceBatchSize": batch_size,
                "inferenceBatchCount": inference_batch_count,
            },
        )


def _encoded_frame(frame: FrameInput) -> dict[str, Any]:
    if isinstance(frame, np.ndarray):
        contiguous = np.ascontiguousarray(frame)
        return {
            "encoding": "numpy-base64",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "colorSpace": "BGR" if contiguous.ndim == 3 else "GRAY",
            "dataBase64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    path = Path(frame).expanduser().resolve()
    if not path.is_file():
        raise BallDetectionError(f"frame does not exist: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "encoding": "file-base64",
        "filename": path.name,
        "mediaType": media_type,
        "dataBase64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def _frame_size_from_input(frame: FrameInput) -> tuple[int, int]:
    image = _frame_array(frame)
    return int(image.shape[1]), int(image.shape[0])


def _response_image_size(response: Mapping[str, Any], frame: FrameInput) -> tuple[int, int]:
    value = response.get("imageSize")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    width = response.get("imageWidth")
    height = response.get("imageHeight")
    if width is not None and height is not None:
        return int(width), int(height)
    return _frame_size_from_input(frame)


def _external_bbox(item: Mapping[str, Any], default_radius: float = 4.0) -> tuple[float, float, float, float]:
    bbox = item.get("bbox", item.get("box"))
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    position = item.get("position")
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) >= 2:
        x, y = float(position[0]), float(position[1])
    else:
        x, y = float(item["x"]), float(item["y"])
    radius = float(item.get("radius", default_radius))
    return x - radius, y - radius, x + radius, y + radius


def _parse_external_candidates(
    response: Mapping[str, Any],
    *,
    backend_name: str,
    image_size: tuple[int, int],
    max_candidates: int,
    nms_iou: float,
    frame_index: int | None,
    timestamp: float | None,
) -> tuple[BallCandidate, ...]:
    raw_items = response.get("candidates")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise BallDetectorUnavailable(f"{backend_name} response has no candidates array")
    width, height = image_size
    candidates: list[BallCandidate] = []
    for index, value in enumerate(raw_items):
        if not isinstance(value, Mapping):
            continue
        try:
            x1, y1, x2, y2 = _external_bbox(value)
            confidence = float(value.get("confidence", value.get("score", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = np.clip((x1, x2), 0.0, float(width))
        y1, y2 = np.clip((y1, y2), 0.0, float(height))
        if not np.isfinite((x1, y1, x2, y2, confidence)).all() or x2 <= x1 or y2 <= y1:
            continue
        metadata: dict[str, Any] = {
            "detectionIndex": index,
            "observed": bool(value.get("observed", True)),
        }
        if frame_index is not None:
            metadata["frameIndex"] = frame_index
        if timestamp is not None:
            metadata["timestamp"] = timestamp
        for key in ("heatmapPeak", "temporalScore", "occluded", "sourceFrameIndex"):
            if key in value:
                metadata[key] = value[key]
        candidates.append(
            BallCandidate(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=confidence,
                backend=backend_name,
                class_id=None,
                metadata=metadata,
            )
        )
    return select_ball_candidates(
        candidates,
        max_candidates=max_candidates,
        nms_iou=nms_iou,
    )


def _http_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise BallDetectorUnavailable("WASB service returned a non-object JSON response")
    return decoded


def _subprocess_transport(
    command: Sequence[str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(command),
        input=json.dumps(payload, separators=(",", ":")),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-500:] or f"exit code {completed.returncode}"
        raise BallDetectorUnavailable(f"WASB subprocess failed: {detail}")
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BallDetectorUnavailable("WASB subprocess returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise BallDetectorUnavailable("WASB subprocess returned a non-object JSON response")
    return decoded


ExternalTransport: TypeAlias = Callable[[Mapping[str, Any], float], Mapping[str, Any]]


class _WasbExternalBallDetector:
    def __init__(
        self,
        *,
        backend_name: str,
        transport: ExternalTransport,
        timeout: float,
        max_candidates: int,
        nms_iou: float,
        failure_policy: FailurePolicy,
        fallback: BallDetector | None,
    ) -> None:
        if timeout <= 0:
            raise BallDetectorConfigurationError("WASB timeout must be positive")
        if max_candidates <= 0:
            raise BallDetectorConfigurationError("max_candidates must be positive")
        if not 0.0 <= nms_iou <= 1.0:
            raise BallDetectorConfigurationError("nms_iou must be between 0 and 1")
        if failure_policy not in ("raise", "fallback"):
            raise BallDetectorConfigurationError("failure_policy must be raise or fallback")
        if failure_policy == "fallback" and fallback is None:
            raise BallDetectorConfigurationError(
                "failure_policy=fallback requires an explicit fallback detector"
            )
        self.backend_name = backend_name
        self._transport = transport
        self._timeout = timeout
        self._max_candidates = max_candidates
        self._nms_iou = nms_iou
        self._failure_policy = failure_policy
        self._fallback = fallback

    def detect(
        self,
        frame: FrameInput,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch:
        try:
            # Offline reconstruction has both neighbours available.  With two
            # context frames the contract is (previous, next), so the worker
            # evaluates [previous, current, next] and returns its centred
            # output channel.  A one-frame context retains the legacy causal
            # (previous, current) behavior for external callers.
            if len(context_frames) >= 2:
                frames = [context_frames[0], frame, context_frames[-1]]
                target_index = 1
                context_mode = "centered"
            elif context_frames:
                frames = [context_frames[0], frame]
                target_index = 1
                context_mode = "causal"
            else:
                frames = [frame]
                target_index = 0
                context_mode = "edge-repeat"
            payload: dict[str, Any] = {
                "contractVersion": 1,
                "frames": [_encoded_frame(value) for value in frames],
                "targetIndex": target_index,
                "maxCandidates": self._max_candidates,
            }
            if frame_index is not None:
                payload["frameIndex"] = frame_index
            if timestamp is not None:
                payload["timestamp"] = timestamp
            response = self._transport(payload, self._timeout)
            image_size = _response_image_size(response, frame)
            candidates = _parse_external_candidates(
                response,
                backend_name=self.backend_name,
                image_size=image_size,
                max_candidates=self._max_candidates,
                nms_iou=self._nms_iou,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            response_metadata = response.get("metadata")
            return BallDetectionBatch(
                candidates=candidates,
                image_size=image_size,
                backend=self.backend_name,
                metadata={
                    "temporalContextFrames": len(context_frames),
                    "temporalContextMode": context_mode,
                    "worker": dict(response_metadata) if isinstance(response_metadata, Mapping) else {},
                },
            )
        except Exception as exc:
            if self._failure_policy == "fallback" and self._fallback is not None:
                fallback_result = self._fallback.detect(
                    frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    context_frames=context_frames,
                )
                return BallDetectionBatch(
                    candidates=fallback_result.candidates,
                    image_size=fallback_result.image_size,
                    backend=fallback_result.backend,
                    metadata={
                        **dict(fallback_result.metadata),
                        "requestedBackend": self.backend_name,
                        "fallbackReason": f"{type(exc).__name__}: {exc}",
                    },
                )
            if isinstance(exc, BallDetectionError):
                raise
            raise BallDetectorUnavailable(f"{self.backend_name} failed: {exc}") from exc


class WasbServiceBallDetector(_WasbExternalBallDetector):
    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        max_candidates: int = 12,
        nms_iou: float = 0.1,
        failure_policy: FailurePolicy = "raise",
        fallback: BallDetector | None = None,
        transport: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise BallDetectorConfigurationError("WASB service URL must use http or https")
        service_transport = transport or _http_transport
        super().__init__(
            backend_name="wasb-service",
            transport=lambda payload, call_timeout: service_transport(url, payload, call_timeout),
            timeout=timeout,
            max_candidates=max_candidates,
            nms_iou=nms_iou,
            failure_policy=failure_policy,
            fallback=fallback,
        )


class WasbSubprocessBallDetector(_WasbExternalBallDetector):
    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 30.0,
        max_candidates: int = 12,
        nms_iou: float = 0.1,
        failure_policy: FailurePolicy = "raise",
        fallback: BallDetector | None = None,
        transport: Callable[
            [Sequence[str], Mapping[str, Any], float], Mapping[str, Any]
        ]
        | None = None,
    ) -> None:
        if not command or any(not value for value in command):
            raise BallDetectorConfigurationError("WASB command must not be empty")
        process_transport = transport or _subprocess_transport
        frozen_command = tuple(command)
        super().__init__(
            backend_name="wasb-subprocess",
            transport=lambda payload, call_timeout: process_transport(
                frozen_command, payload, call_timeout
            ),
            timeout=timeout,
            max_candidates=max_candidates,
            nms_iou=nms_iou,
            failure_policy=failure_policy,
            fallback=fallback,
        )


def build_ball_detector(
    config: BallDetectorConfig,
    *,
    model: Any | None = None,
    fallback: BallDetector | None = None,
    model_loader: Callable[[str], Any] | None = None,
    service_transport: Callable[
        [str, Mapping[str, Any], float], Mapping[str, Any]
    ]
    | None = None,
    subprocess_transport: Callable[
        [Sequence[str], Mapping[str, Any], float], Mapping[str, Any]
    ]
    | None = None,
) -> BallDetector:
    """Build one configured backend without implicit downloads or fallback."""

    if config.backend in ("generic-ultralytics", "dedicated-ultralytics"):
        dedicated = config.backend == "dedicated-ultralytics"
        detector_config = UltralyticsBallDetectorConfig(
            backend_name=config.backend,
            class_ids=(0,) if dedicated else (32,),
            confidence=config.confidence,
            image_size=(
                config.image_size
                if config.image_size is not None
                else (640 if dedicated else 1280)
            ),
            device=config.device,
            max_candidates=config.max_candidates,
            tile_size=config.tile_size if dedicated else None,
            tile_overlap=config.tile_overlap,
            inference_batch_size=config.inference_batch_size,
            nms_iou=config.nms_iou,
        )
        return UltralyticsBallDetector(
            detector_config,
            model=model,
            checkpoint_path=config.checkpoint_path,
            model_loader=model_loader,
        )
    if config.backend == "wasb-service":
        if config.wasb_service_url is None:
            raise BallDetectorConfigurationError("wasb_service_url is required")
        return WasbServiceBallDetector(
            config.wasb_service_url,
            timeout=config.wasb_timeout,
            max_candidates=config.max_candidates,
            nms_iou=config.nms_iou,
            failure_policy=config.failure_policy,
            fallback=fallback,
            transport=service_transport,
        )
    if config.backend == "wasb-subprocess":
        return WasbSubprocessBallDetector(
            config.wasb_command,
            timeout=config.wasb_timeout,
            max_candidates=config.max_candidates,
            nms_iou=config.nms_iou,
            failure_policy=config.failure_policy,
            fallback=fallback,
            transport=subprocess_transport,
        )
    raise BallDetectorConfigurationError(f"unsupported ball detector backend: {config.backend}")
