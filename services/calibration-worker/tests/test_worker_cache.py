from __future__ import annotations

from threading import Thread

import torch

from calibration_worker_service.calibration_cache import CalibrationResultCache
from calibration_worker_service.calibration_contract import (
    CalibrationReadiness,
    DecodedFrame,
    FrameCalibration,
)
from calibration_worker_service.pnlcalib_engine import PnLCalibEngine
from calibration_worker_service.runtime import CalibrationEngineRuntime


def _frame(frame_index: int, content_hash: str) -> DecodedFrame:
    return DecodedFrame(
        frame_index=frame_index,
        width=1920,
        height=1080,
        tensor=torch.zeros(1),
        content_sha256=content_hash,
    )


def _calibration(frame_index: int) -> FrameCalibration:
    return FrameCalibration(
        frame_index=frame_index,
        confidence=0.9,
        detected_keypoint_count=8,
        completed_keypoint_count=10,
        inlier_count=8,
        inlier_ratio=1.0,
        line_count=4,
        detected_line_count=4,
        raw_lines=(),
        matched_curves=0,
        completed_curve_count=0,
        reprojection_error=1.0,
        ground_error_p50_metres=0.1,
        ground_error_p95_metres=0.2,
        pitch_side="left",
        raw_keypoints=(),
        image_to_pitch=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


class FakeInference:
    def __init__(self, results_by_hash: dict[str, FrameCalibration | None]) -> None:
        self.results_by_hash = results_by_hash
        self.calls: list[list[str]] = []

    def infer(self, frames, _timings):
        self.calls.append([frame.content_sha256 for frame in frames])
        return [self.results_by_hash[frame.content_sha256] for frame in frames]


def _engine(
    results_by_hash: dict[str, FrameCalibration | None],
    *,
    model_version: str = "test-model-v1",
    cache: CalibrationResultCache | None = None,
) -> tuple[PnLCalibEngine, FakeInference]:
    inference = FakeInference(results_by_hash)
    return (
        PnLCalibEngine(
            inference=inference,
            model_version=model_version,
            device="cpu",
            batch_size=2,
            cache=cache or CalibrationResultCache(max_entries=8, ttl_seconds=3600.0),
            model_load_seconds=0.0,
        ),
        inference,
    )


def test_identical_content_is_deduplicated_then_served_from_cache() -> None:
    engine, inference = _engine(
        {"same": _calibration(1), "other": _calibration(3)}
    )

    first = engine.calibrate(
        [_frame(1, "same"), _frame(2, "same"), _frame(3, "other")]
    )
    diagnostics = first.diagnostics.to_wire()

    assert inference.calls == [["same", "other"]]
    assert [item.frame_index for item in first.frames] == [1, 2, 3]
    assert diagnostics["requestedFrameCount"] == 3
    assert diagnostics["uniqueFrameCount"] == 2
    assert diagnostics["cacheHitCount"] == 0
    assert diagnostics["cacheMissCount"] == 2
    assert diagnostics["deduplicatedFrameCount"] == 1
    assert diagnostics["inferenceBatchCount"] == 1
    assert diagnostics["cacheEntryCount"] == 2

    warm = engine.calibrate([_frame(11, "other"), _frame(12, "same")])
    warm_diagnostics = warm.diagnostics.to_wire()

    assert inference.calls == [["same", "other"]]
    assert [item.frame_index for item in warm.frames] == [11, 12]
    assert warm_diagnostics["cacheHitCount"] == 2
    assert warm_diagnostics["cacheMissCount"] == 0
    assert warm_diagnostics["inferenceBatchCount"] == 0
    assert warm_diagnostics["modelInferenceSeconds"] == 0.0


def test_cache_key_includes_model_version() -> None:
    cache = CalibrationResultCache(max_entries=8, ttl_seconds=3600.0)
    first_engine, first_inference = _engine(
        {"same": _calibration(1)},
        model_version="test-model-v1",
        cache=cache,
    )
    second_engine, second_inference = _engine(
        {"same": _calibration(2)},
        model_version="test-model-v2",
        cache=cache,
    )

    first_engine.calibrate([_frame(1, "same")])
    result = second_engine.calibrate([_frame(2, "same")])

    assert first_inference.calls == [["same"]]
    assert second_inference.calls == [["same"]]
    assert result.diagnostics.cache_hit_count == 0
    assert result.diagnostics.cache_miss_count == 1


def test_failed_calibration_is_cached_too() -> None:
    engine, inference = _engine({"unusable": None})

    assert engine.calibrate([_frame(1, "unusable")]).frames == ()
    second = engine.calibrate([_frame(2, "unusable")])

    assert second.frames == ()
    assert inference.calls == [["unusable"]]
    assert second.diagnostics.cache_hit_count == 1
    assert second.diagnostics.cache_miss_count == 0


def test_cache_expires_entries_and_evicts_the_least_recently_used() -> None:
    calibration = _calibration(1)
    cache = CalibrationResultCache(max_entries=2, ttl_seconds=5.0)
    cache.put("a", calibration, now=0.0)
    cache.put("b", calibration, now=1.0)

    assert cache.get("a", now=2.0).hit is True
    cache.put("c", calibration, now=3.0)

    assert cache.get("b", now=3.0).hit is False
    assert cache.get("a", now=6.0).hit is False
    assert cache.get("c", now=6.0).hit is True


def test_runtime_constructs_exactly_one_engine_under_concurrency() -> None:
    created: list[object] = []

    class FakeEngine:
        def readiness(self):
            return CalibrationReadiness("cpu", 1, "test", 0.0, 0, 0.0, 0)

    def factory():
        engine = FakeEngine()
        created.append(engine)
        return engine

    runtime = CalibrationEngineRuntime(factory)
    resolved: list[object] = []
    threads = [Thread(target=lambda: resolved.append(runtime.get_engine())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert all(engine is created[0] for engine in resolved)
