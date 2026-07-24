from __future__ import annotations

from time import perf_counter

import torch

from .calibration_contract import DecodedFrame, FrameCalibration, InferenceTimings
from .calibration_projector import CalibrationProjector
from .pnlcalib_constants import KEYPOINT_THRESHOLD, LINE_THRESHOLD
from .pnlcalib_runtime import LoadedPnLCalibModels


class PnLCalibInference:
    """Own model execution and decoding, but not caching or request batching."""

    def __init__(self, models: LoadedPnLCalibModels) -> None:
        self._models = models
        self._projector = CalibrationProjector(models.runtime)

    def _synchronize(self) -> None:
        """Make asynchronous accelerator timings and failures observable."""

        if self._models.device.type == "mps":
            torch.mps.synchronize()
        elif self._models.device.type == "cuda":
            torch.cuda.synchronize(self._models.device)

    def infer(
        self,
        frames: list[DecodedFrame],
        timings: InferenceTimings,
    ) -> list[FrameCalibration | None]:
        started = perf_counter()
        batch = torch.stack([frame.tensor for frame in frames])
        if self._models.device.type != "cpu":
            batch = batch.to(self._models.device)
        self._synchronize()
        timings.tensor_assembly_seconds += perf_counter() - started

        started = perf_counter()
        heatmaps = self._models.keypoint_model(batch)
        self._synchronize()
        timings.keypoint_inference_seconds += perf_counter() - started

        started = perf_counter()
        line_heatmaps = self._models.line_model(batch)
        self._synchronize()
        timings.line_inference_seconds += perf_counter() - started

        started = perf_counter()
        self._synchronize()
        runtime = self._models.runtime
        keypoint_coords = runtime.decode_keypoints(heatmaps[:, :-1])
        line_coords = runtime.decode_lines(line_heatmaps[:, :-1])
        keypoint_items = runtime.coords_to_dict(
            keypoint_coords,
            threshold=KEYPOINT_THRESHOLD,
            ground_plane_only=True,
        )
        line_items = runtime.coords_to_dict(
            line_coords,
            threshold=LINE_THRESHOLD,
            ground_plane_only=False,
        )
        self._synchronize()
        timings.heatmap_decode_seconds += perf_counter() - started

        started = perf_counter()
        output = [
            self._projector.project(frame, keypoints, detected_lines)
            for frame, keypoints, detected_lines in zip(
                frames,
                keypoint_items,
                line_items,
            )
        ]
        timings.geometry_seconds += perf_counter() - started
        return output
