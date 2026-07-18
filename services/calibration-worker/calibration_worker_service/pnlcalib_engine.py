from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic, perf_counter

import torch

from .calibration_cache import CalibrationResultCache
from .calibration_contract import (
    CalibrationBatchInference,
    CalibrationBatchResult,
    CalibrationDiagnostics,
    CalibrationReadiness,
    DecodedFrame,
    FrameCalibration,
    InferenceTimings,
)


class PnLCalibEngine:
    """Serialize model access and coordinate content-cache-aware batches."""

    def __init__(
        self,
        *,
        inference: CalibrationBatchInference,
        model_version: str,
        device: str,
        batch_size: int,
        cache: CalibrationResultCache,
        model_load_seconds: float,
    ) -> None:
        self._inference = inference
        self._model_version = model_version
        self._device = device
        self._batch_size = max(1, batch_size)
        self._cache = cache
        self._model_load_seconds = model_load_seconds
        self._lock = Lock()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_load_seconds(self) -> float:
        return self._model_load_seconds

    def calibrate(self, frames: list[DecodedFrame]) -> CalibrationBatchResult:
        engine_started = perf_counter()
        timings = InferenceTimings()
        lock_started = perf_counter()
        with self._lock, torch.inference_mode():
            lock_wait = perf_counter() - lock_started
            resolved: list[FrameCalibration | None] = [None] * len(frames)
            misses: OrderedDict[str, list[int]] = OrderedDict()
            cache_hit_count = 0
            now = monotonic()
            for index, frame in enumerate(frames):
                key = self._cache.key(self._model_version, frame.content_sha256)
                lookup = self._cache.get(key, now)
                if lookup.hit:
                    cache_hit_count += 1
                    resolved[index] = (
                        lookup.result.for_frame(frame.frame_index)
                        if lookup.result is not None
                        else None
                    )
                else:
                    misses.setdefault(key, []).append(index)

            miss_items = list(misses.items())
            for start in range(0, len(miss_items), self._batch_size):
                batch_items = miss_items[start : start + self._batch_size]
                batch_frames = [frames[indices[0]] for _, indices in batch_items]
                batch_results = self._inference.infer(batch_frames, timings)
                if len(batch_results) != len(batch_items):
                    raise RuntimeError(
                        "PnLCalib inference returned a different number of results"
                    )
                for (key, indices), result in zip(batch_items, batch_results):
                    self._cache.put(key, result, monotonic())
                    for index in indices:
                        resolved[index] = (
                            result.for_frame(frames[index].frame_index)
                            if result is not None
                            else None
                        )

            diagnostics = CalibrationDiagnostics(
                model_version=self._model_version,
                requested_frame_count=len(frames),
                unique_frame_count=len(
                    {
                        self._cache.key(self._model_version, frame.content_sha256)
                        for frame in frames
                    }
                ),
                cache_hit_count=cache_hit_count,
                cache_miss_count=len(miss_items),
                deduplicated_frame_count=sum(
                    max(0, len(indices) - 1) for _, indices in miss_items
                ),
                inference_batch_count=(
                    (len(miss_items) + self._batch_size - 1) // self._batch_size
                    if miss_items
                    else 0
                ),
                cache_entry_count=len(self._cache),
                lock_wait_seconds=lock_wait,
                inference_timings=timings.snapshot(),
                engine_seconds=perf_counter() - engine_started,
            )
        return CalibrationBatchResult(
            frames=tuple(item for item in resolved if item is not None),
            diagnostics=diagnostics,
        )

    def readiness(self) -> CalibrationReadiness:
        return CalibrationReadiness(
            device=self._device,
            batch_size=self._batch_size,
            model_version=self._model_version,
            model_load_seconds=self._model_load_seconds,
            cache_max_entries=self._cache.max_entries,
            cache_ttl_seconds=self._cache.ttl_seconds,
            cache_entry_count=len(self._cache),
        )
