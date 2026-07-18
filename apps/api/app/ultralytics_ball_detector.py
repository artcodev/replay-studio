from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ball_candidate_selection import (
    ImageTile,
    parse_ultralytics_ball_candidates,
    select_ball_candidates,
)
from .ball_detection_contract import (
    BallDetectionBatch,
    BallDetectionError,
    BallDetectorConfigurationError,
    BallDetectorUnavailable,
    FrameInput,
    UltralyticsBallDetectorConfig,
)
from .ball_frame_input import frame_array


def _axis_origins(
    length: int,
    tile_length: int,
    overlap: float,
) -> tuple[int, ...]:
    tile_length = min(tile_length, length)
    if tile_length == length:
        return (0,)
    step = max(1, int(round(tile_length * (1.0 - overlap))))
    origins = list(range(0, length - tile_length + 1, step))
    final_origin = length - tile_length
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return tuple(origins)


def _tiles(
    image_size: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: float,
) -> tuple[ImageTile, ...]:
    image_width, image_height = image_size
    tile_width, tile_height = tile_size
    tiles: list[ImageTile] = []
    for y in _axis_origins(image_height, tile_height, overlap):
        for x in _axis_origins(image_width, tile_width, overlap):
            width = min(tile_width, image_width - x)
            height = min(tile_height, image_height - y)
            tiles.append(
                ImageTile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            )
    return tuple(tiles)


def _result_image_size(result: Any, frame: FrameInput) -> tuple[int, int]:
    image = getattr(result, "orig_img", None)
    if image is not None and getattr(image, "ndim", 0) >= 2:
        return int(image.shape[1]), int(image.shape[0])
    if isinstance(frame, np.ndarray) and frame.ndim >= 2:
        return int(frame.shape[1]), int(frame.shape[0])
    image = frame_array(frame)
    return int(image.shape[1]), int(image.shape[0])


def _default_model_loader(checkpoint: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - production dependency
        raise BallDetectorUnavailable("Ultralytics is not installed") from exc
    return YOLO(checkpoint)


class UltralyticsBallDetector:
    """Single-frame, tiled, and explicit-region local detector."""

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
                    "inject an Ultralytics model or provide an existing local "
                    "checkpoint"
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
            raise BallDetectorUnavailable(
                f"{self.backend_name} returned no Results object"
            )
        return output[0]

    def _predict_batch(self, sources: Sequence[np.ndarray]) -> Sequence[Any]:
        output = self._model.predict(list(sources), **self._prediction_arguments())
        if len(output) != len(sources):
            raise BallDetectorUnavailable(
                f"{self.backend_name} returned {len(output)} results for "
                f"{len(sources)} tiles"
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
        del context_frames
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
                metadata={
                    "rawCandidateCount": len(raw),
                    "tileCount": 1,
                    "scanMode": "global",
                },
            )

        image = frame_array(frame)
        image_size = (int(image.shape[1]), int(image.shape[0]))
        tiles = _tiles(
            image_size,
            self.config.tile_size,
            self.config.tile_overlap,
        )
        return self._detect_tiles(
            image,
            image_size,
            tiles,
            frame_index=frame_index,
            timestamp=timestamp,
            scan_mode="global",
        )

    def _detect_tiles(
        self,
        image: np.ndarray,
        image_size: tuple[int, int],
        tiles: Sequence[ImageTile],
        *,
        frame_index: int | None,
        timestamp: float | None,
        scan_mode: str,
    ) -> BallDetectionBatch:
        raw = []
        inference_batch_count = 0
        batch_size = self.config.inference_batch_size
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
        metadata: dict[str, Any] = {
            "rawCandidateCount": len(raw),
            "tileCount": len(tiles),
            "inferenceBatchSize": batch_size,
            "inferenceBatchCount": inference_batch_count,
            "scanMode": scan_mode,
        }
        if scan_mode == "global" and self.config.tile_size is not None:
            metadata.update(
                {
                    "tileSize": list(self.config.tile_size),
                    "tileOverlap": self.config.tile_overlap,
                }
            )
        if scan_mode == "roi":
            metadata.update(
                {
                    "roiRegionCount": len(tiles),
                    "roiRegions": [
                        [
                            tile.x,
                            tile.y,
                            tile.x + tile.width,
                            tile.y + tile.height,
                        ]
                        for tile in tiles
                    ],
                }
            )
        return BallDetectionBatch(
            candidates=select_ball_candidates(
                raw,
                max_candidates=self.config.max_candidates,
                nms_iou=self.config.nms_iou,
            ),
            image_size=image_size,
            backend=self.backend_name,
            metadata=metadata,
        )

    def detect_regions(
        self,
        frame: FrameInput,
        regions: Sequence[tuple[float, float, float, float]],
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch:
        """Detect full-image pixel regions while preserving coordinates."""

        del context_frames
        image = frame_array(frame)
        image_size = (int(image.shape[1]), int(image.shape[0]))
        image_width, image_height = image_size
        tiles: list[ImageTile] = []
        for region in regions:
            if len(region) != 4:
                raise BallDetectionError(
                    "ROI regions must contain four coordinates"
                )
            x1, y1, x2, y2 = (float(value) for value in region)
            if not np.isfinite((x1, y1, x2, y2)).all():
                raise BallDetectionError(
                    "ROI regions must contain finite coordinates"
                )
            left = max(0, min(image_width, int(np.floor(x1))))
            top = max(0, min(image_height, int(np.floor(y1))))
            right = max(0, min(image_width, int(np.ceil(x2))))
            bottom = max(0, min(image_height, int(np.ceil(y2))))
            if right <= left or bottom <= top:
                continue
            tiles.append(
                ImageTile(
                    index=len(tiles),
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                )
            )
        return self._detect_tiles(
            image,
            image_size,
            tiles,
            frame_index=frame_index,
            timestamp=timestamp,
            scan_mode="roi",
        )
