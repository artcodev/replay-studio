import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ball_detection_contract import BallCandidate, BallDetectionBatch
from app.ball_frames import DenseBallFrameSet
from app.reconstruction_errors import ReconstructionError
from app.reconstruction_frame_analysis import analyze_scene_frame
from app.ball_detection_configuration import (
    verify_queued_ball_checkpoint as _verify_queued_ball_checkpoint,
)
from app.reconstruction_ball_detector_selection import (
    configured_ball_detectors as _configured_ball_detectors,
)
from app.reconstruction_ball_detection import detect_ball_frames as _detect_ball_frames
from app.reconstruction_ball_projection_status import (
    ball_world_projection_status as _ball_world_projection_status,
)
from app.reconstruction_ball_roi import ball_roi_regions as _ball_roi_regions


class RecordingDetector:
    def __init__(
        self,
        backend_name: str,
        *,
        fail: Exception | None = None,
        candidate: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.fail = fail
        self.candidate = candidate
        self.calls: list[dict] = []

    def detect(
        self,
        frame,
        *,
        frame_index=None,
        timestamp=None,
        context_frames=(),
    ) -> BallDetectionBatch:
        self.calls.append(
            {
                "frame": Path(frame),
                "frameIndex": frame_index,
                "timestamp": timestamp,
                "contextFrames": tuple(Path(item) for item in context_frames),
            }
        )
        if self.fail is not None:
            raise self.fail
        candidates = (
            (
                BallCandidate(
                    bbox=(10.0, 20.0, 14.0, 24.0),
                    confidence=0.8,
                    backend=self.backend_name,
                ),
            )
            if self.candidate
            else ()
        )
        return BallDetectionBatch(
            candidates=candidates,
            image_size=(1920, 1080),
            backend=self.backend_name,
        )


class AdaptiveRecordingDetector(RecordingDetector):
    def __init__(
        self,
        backend_name: str = "dedicated-ultralytics",
        *,
        global_empty_frames: set[int] | None = None,
        roi_empty_frames: set[int] | None = None,
        global_error_frames: set[int] | None = None,
    ) -> None:
        super().__init__(backend_name)
        self.global_empty_frames = global_empty_frames or set()
        self.roi_empty_frames = roi_empty_frames or set()
        self.global_error_frames = global_error_frames or set()
        self.global_calls: list[dict] = []
        self.roi_calls: list[dict] = []

    def _batch(self, frame_index: int, *, empty: bool, metadata: dict):
        candidate = BallCandidate(
            bbox=(120.0 + frame_index, 620.0, 130.0 + frame_index, 630.0),
            confidence=0.8,
            backend=self.backend_name,
            metadata={
                "tile": {
                    "index": 0,
                    "x": 0,
                    "y": 440,
                    "width": 640,
                    "height": 640,
                }
            },
        )
        return BallDetectionBatch(
            candidates=() if empty else (candidate,),
            image_size=(1920, 1080),
            backend=self.backend_name,
            metadata=metadata,
        )

    def detect(
        self,
        frame,
        *,
        frame_index=None,
        timestamp=None,
        context_frames=(),
    ) -> BallDetectionBatch:
        call = {
            "frame": Path(frame),
            "frameIndex": int(frame_index),
            "timestamp": timestamp,
            "contextFrames": tuple(Path(item) for item in context_frames),
        }
        self.calls.append(call)
        self.global_calls.append(call)
        if int(frame_index) in self.global_error_frames:
            raise RuntimeError("global detector failure")
        return self._batch(
            int(frame_index),
            empty=int(frame_index) in self.global_empty_frames,
            metadata={"tileCount": 8, "scanMode": "global"},
        )

    def detect_regions(
        self,
        frame,
        regions,
        *,
        frame_index=None,
        timestamp=None,
        context_frames=(),
    ) -> BallDetectionBatch:
        call = {
            "frame": Path(frame),
            "frameIndex": int(frame_index),
            "timestamp": timestamp,
            "contextFrames": tuple(Path(item) for item in context_frames),
            "regions": tuple(tuple(float(value) for value in item) for item in regions),
        }
        self.roi_calls.append(call)
        return self._batch(
            int(frame_index),
            empty=int(frame_index) in self.roi_empty_frames,
            metadata={
                "tileCount": len(regions),
                "roiRegionCount": len(regions),
                "scanMode": "roi",
            },
        )


class FlakyPrimaryDetector(RecordingDetector):
    """Primary detector that fails only on the requested frame indexes."""

    def __init__(self, fail_frames: set[int]) -> None:
        super().__init__("dedicated-ultralytics", candidate=True)
        self.fail_frames = fail_frames

    def detect(self, frame, *, frame_index=None, timestamp=None, context_frames=()):
        if int(frame_index) in self.fail_frames:
            self.calls.append(
                {
                    "frame": Path(frame),
                    "frameIndex": frame_index,
                    "timestamp": timestamp,
                    "contextFrames": tuple(Path(item) for item in context_frames),
                }
            )
            raise RuntimeError("primary detector outage")
        return super().detect(
            frame,
            frame_index=frame_index,
            timestamp=timestamp,
            context_frames=context_frames,
        )


def _scene() -> dict:
    return {
        "id": "ball-detection-test",
        "payload": {"videoAsset": {"id": "asset-1", "analysisFps": 10}},
    }


def _dense_frames(tmp_path, count=3, frame_rate=25.0) -> DenseBallFrameSet:
    frames = tuple(
        (tmp_path / f"frame_{index:06d}.jpg", index / frame_rate)
        for index in range(count)
    )
    return DenseBallFrameSet(
        frames=frames,
        frame_rate=frame_rate,
        source_start=0.0,
        source_end=count / frame_rate,
        cache_key="test-cache",
        cache_hit=True,
    )


def _patch_runtime(monkeypatch, dense, *, failure_policy="fallback") -> None:
    monkeypatch.setattr(
        "app.reconstruction_ball_detection_source.dense_ball_frame_paths", lambda _: dense
    )
    def settings():
        return SimpleNamespace(
            ball_detection_failure_policy=failure_policy,
            reconstruction_frame_rate=10.0,
            ball_detection_checkpoint_interval=1,
            ball_detection_circuit_retry_interval=2,
            media_root=str(dense.frames[0][0].parent) if dense.frames else ".",
        )

    monkeypatch.setattr("app.reconstruction_ball_detection.get_settings", settings)
    monkeypatch.setattr("app.reconstruction_ball_detection_source.get_settings", settings)


def _adaptive_input(*, interval: int = 5, max_regions: int = 3, padding: int = 320):
    return {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {"name": "ball.pt", "size": 123, "mtimeNs": 456},
        "confidence": 0.05,
        "tileSize": 640,
        "adaptiveRoi": {
            "enabled": True,
            "algorithmVersion": "adaptive-roi-v1",
            "fullScanIntervalFrames": interval,
            "maxRegions": max_regions,
            "paddingPixels": padding,
            "reacquirePolicy": "same-frame-global-on-miss",
        },
    }


def test_dense_detection_receives_centered_temporal_context_with_edge_repeats(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector = RecordingDetector("offline-temporal")

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(), detector, None, [], []
    )

    paths = [path for path, _ in dense.frames]
    assert [call["frame"] for call in detector.calls] == paths
    assert [call["contextFrames"] for call in detector.calls] == [
        (paths[0], paths[1]),
        (paths[0], paths[2]),
        (paths[1], paths[2]),
    ]
    assert [call["frameIndex"] for call in detector.calls] == [0, 1, 2]
    assert [call["timestamp"] for call in detector.calls] == pytest.approx(
        [0.0, 0.04, 0.08]
    )
    assert len(resolved) == len(batches) == 3
    assert metadata["failedFrameCount"] == 0
    assert metadata["fallbackFrameCount"] == 0
    assert warnings == []


def test_adaptive_dedicated_detector_uses_initial_periodic_and_final_global_scans(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path, count=8)
    _patch_runtime(monkeypatch, dense)
    detector = AdaptiveRecordingDetector()

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(),
        detector,
        None,
        [],
        [],
        detector_input=_adaptive_input(),
    )

    assert len(resolved) == 8
    assert [call["frameIndex"] for call in detector.global_calls] == [0, 5, 7]
    assert [call["frameIndex"] for call in detector.roi_calls] == [1, 2, 3, 4, 6]
    assert [item["scanMode"] for item in batches] == [
        "global",
        "roi",
        "roi",
        "roi",
        "roi",
        "global",
        "roi",
        "global",
    ]
    assert batches[0]["metadata"]["globalScanReason"] == "initial-frame"
    assert batches[5]["metadata"]["globalScanReason"] == "periodic"
    assert batches[7]["metadata"]["globalScanReason"] == "final-frame"
    # The exact source tile which produced the global candidate is reused;
    # a naive centered crop is not equivalent for this checkpoint.
    assert detector.roi_calls[0]["regions"] == ((0.0, 440.0, 640.0, 1080.0),)
    assert metadata["adaptiveRoi"] == {
        "enabled": True,
        "algorithmVersion": "adaptive-roi-v1",
        "fullScanIntervalFrames": 5,
        "maxRegions": 3,
        "paddingPixels": 320,
        "globalScanFrameCount": 3,
        "roiScanFrameCount": 5,
        "roiReacquireFrameCount": 0,
        "globalInferenceFrameCount": 3,
        "roiInferenceFrameCount": 5,
        "globalCropCount": 24,
        "roiCropCount": 5,
        "totalModelCropCount": 29,
        "referenceFullScanCropCount": 8,
        "estimatedFullScanBaselineCropCount": 64,
        "estimatedCropReductionRatio": 0.5469,
    }
    assert warnings == []


def test_adaptive_roi_miss_reacquires_globally_on_the_same_timestamp(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path, count=4)
    _patch_runtime(monkeypatch, dense)
    detector = AdaptiveRecordingDetector(roi_empty_frames={1})

    _, metadata, batches, _ = _detect_ball_frames(
        _scene(), detector, None, [], [], detector_input=_adaptive_input()
    )

    assert [call["frameIndex"] for call in detector.global_calls] == [0, 1, 3]
    assert [call["frameIndex"] for call in detector.roi_calls] == [1, 2]
    assert batches[1]["scanMode"] == "global-reacquire"
    assert batches[1]["metadata"]["globalScanReason"] == "roi-miss"
    assert batches[1]["metadata"]["roiAttempt"]["roiRegionCount"] == 1
    assert metadata["adaptiveRoi"]["roiReacquireFrameCount"] == 1
    assert metadata["adaptiveRoi"]["globalCropCount"] == 24
    assert metadata["adaptiveRoi"]["roiCropCount"] == 2
    assert metadata["adaptiveRoi"]["totalModelCropCount"] == 26


def test_camera_cut_discards_roi_seed_and_forces_global_scan(monkeypatch, tmp_path):
    dense = _dense_frames(tmp_path, count=4)
    _patch_runtime(monkeypatch, dense)
    detector = AdaptiveRecordingDetector()
    scene = _scene()
    scene["payload"]["cameraCuts"] = [{"t": 0.08, "preset": "broadcast"}]

    _, _, batches, _ = _detect_ball_frames(
        scene, detector, None, [], [], detector_input=_adaptive_input()
    )

    assert [call["frameIndex"] for call in detector.global_calls] == [0, 2, 3]
    assert [call["frameIndex"] for call in detector.roi_calls] == [1]
    assert batches[2]["metadata"]["globalScanReason"] == "camera-cut"


def test_roi_builder_keeps_multiple_seeds_deduplicates_tiles_and_shifts_border():
    seeds = [
        {
            "x": 110,
            "y": 620,
            "confidence": 0.9,
            "detectorMetadata": {
                "tile": {"x": 0, "y": 440, "width": 640, "height": 640}
            },
        },
        {
            "x": 120,
            "y": 630,
            "confidence": 0.8,
            "detectorMetadata": {
                "tile": {"x": 0, "y": 440, "width": 640, "height": 640}
            },
        },
        {
            "x": 900,
            "y": 400,
            "confidence": 0.7,
            "detectorMetadata": {
                "tile": {"x": 640, "y": 0, "width": 640, "height": 640}
            },
        },
        # A manual/unscaled seed without tile provenance uses a shifted 640px
        # window instead of shrinking at the right/bottom border.
        {"x": 1910, "y": 1070, "confidence": 0.6},
    ]

    regions = _ball_roi_regions(
        seeds,
        (1920, 1080),
        max_regions=3,
        padding_pixels=320,
    )

    assert regions == [
        (0.0, 440.0, 640.0, 1080.0),
        (640.0, 0.0, 1280.0, 640.0),
        (1280.0, 440.0, 1920.0, 1080.0),
    ]


def test_clean_dense_detection_is_reused_from_raw_candidate_cache(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    monkeypatch.setattr(
        "app.reconstruction_ball_detection_source.dense_ball_frame_paths", lambda _: dense
    )
    def settings():
        return SimpleNamespace(
            ball_detection_failure_policy="fallback",
            ball_detection_checkpoint_interval=4,
            ball_detection_circuit_retry_interval=2,
            reconstruction_frame_rate=10.0,
            media_root=str(tmp_path),
        )

    monkeypatch.setattr("app.reconstruction_ball_detection.get_settings", settings)
    monkeypatch.setattr("app.reconstruction_ball_detection_source.get_settings", settings)
    detector_input = {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {"name": "ball.pt", "size": 123, "mtimeNs": 456},
        "confidence": 0.05,
        "tileSize": 640,
    }
    first_detector = RecordingDetector("dedicated-ultralytics", candidate=True)

    first_resolved, first_metadata, first_batches, first_warnings = _detect_ball_frames(
        _scene(),
        first_detector,
        None,
        [],
        [],
        detector_input=detector_input,
    )

    assert len(first_detector.calls) == 3
    assert first_metadata["detectionCacheHit"] is False
    assert first_metadata["detectionCacheStored"] is True
    assert first_warnings == []

    second_detector = RecordingDetector(
        "dedicated-ultralytics",
        fail=AssertionError("detector must not run on a cache hit"),
    )
    second_resolved, second_metadata, second_batches, second_warnings = _detect_ball_frames(
        _scene(),
        second_detector,
        None,
        [],
        [],
        detector_input=detector_input,
    )

    assert second_detector.calls == []
    assert second_metadata["detectionCacheHit"] is True
    assert second_metadata["backendCounts"] == {"dedicated-ultralytics": 3}
    assert second_resolved == first_resolved
    assert second_batches == first_batches
    assert second_warnings == []


class SequenceRecordingDetector(RecordingDetector):
    """WASB-style detector with a batched sequence transport."""

    def __init__(self, *, fail_sequence: bool = False) -> None:
        super().__init__("wasb-service", candidate=True)
        self.fail_sequence = fail_sequence
        self.sequence_calls: list[list[int]] = []

    def detect_sequence(self, frames):
        self.sequence_calls.append([frame_index for _, frame_index, _ in frames])
        if self.fail_sequence:
            raise RuntimeError("batched transport offline")
        return [
            BallDetectionBatch(
                candidates=(
                    BallCandidate(
                        bbox=(10.0, 20.0, 14.0, 24.0),
                        confidence=0.8,
                        backend=self.backend_name,
                    ),
                ),
                image_size=(1920, 1080),
                backend=self.backend_name,
            )
            for _ in frames
        ]


def _wasb_batched_input(chunk_size: int = 3) -> dict:
    return {
        "schemaVersion": 1,
        "backend": "wasb-service",
        "wasbTransport": "batched-sequence",
        "wasbBatchSize": chunk_size,
    }


def test_batched_transport_sends_one_request_per_chunk(monkeypatch, tmp_path):
    dense = _dense_frames(tmp_path, count=7)
    _patch_runtime(monkeypatch, dense)
    detector = SequenceRecordingDetector()

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(),
        detector,
        None,
        [],
        [],
        detector_input=_wasb_batched_input(chunk_size=3),
    )

    # 7 frames at 3 per request → 3 requests, zero per-frame calls.
    assert detector.sequence_calls == [[0, 1, 2], [3, 4, 5], [6]]
    assert detector.calls == []
    assert len(resolved) == len(batches) == 7
    assert metadata["fallbackFrameCount"] == 0
    assert metadata["batchedTransport"] == {
        "requested": True,
        "requestCount": 3,
        "framesPerRequest": 3,
    }
    assert warnings == []


def test_batched_transport_failure_degrades_to_the_per_frame_path(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector = SequenceRecordingDetector(fail_sequence=True)

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(),
        detector,
        None,
        [],
        [],
        detector_input=_wasb_batched_input(),
    )

    # The batched request failed once; every frame still resolves through the
    # ordinary per-frame path, so only the batching is lost — not quality.
    assert detector.sequence_calls == [[0, 1, 2]]
    assert [call["frameIndex"] for call in detector.calls] == [0, 1, 2]
    assert len(resolved) == len(batches) == 3
    assert metadata["batchedTransport"]["requestCount"] == 0
    assert "batched transport offline" in metadata["batchedTransport"]["fallbackReason"]
    assert metadata["fallbackFrameCount"] == 0
    assert metadata["backendCounts"] == {"wasb-service": 3}
    assert warnings == []


def test_degraded_dense_detection_is_cached_and_reused_with_markers(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector_input = {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {"name": "ball.pt", "size": 123, "mtimeNs": 456},
        "confidence": 0.05,
        "tileSize": 640,
    }
    flaky = FlakyPrimaryDetector(fail_frames={2})
    fallback = RecordingDetector("generic-ultralytics", candidate=True)

    first_resolved, first_metadata, first_batches, first_warnings = _detect_ball_frames(
        _scene(),
        flaky,
        fallback,
        [],
        [],
        detector_input=detector_input,
    )

    assert first_metadata["fallbackFrameCount"] == 1
    assert first_metadata["failedFrameCount"] == 0
    # One degraded frame must not void the cache for the whole run.
    assert first_metadata["detectionCacheStored"] is True
    assert any("explicit fallback on 1/3 frames" in item for item in first_warnings)

    second_primary = RecordingDetector(
        "dedicated-ultralytics",
        fail=AssertionError("primary must not run on a cache hit"),
    )
    second_fallback = RecordingDetector(
        "generic-ultralytics",
        fail=AssertionError("fallback must not run on a cache hit"),
    )
    second_resolved, second_metadata, second_batches, second_warnings = (
        _detect_ball_frames(
            _scene(),
            second_primary,
            second_fallback,
            [],
            [],
            detector_input=detector_input,
        )
    )

    assert second_primary.calls == []
    assert second_fallback.calls == []
    assert second_metadata["detectionCacheHit"] is True
    assert second_metadata["fallbackFrameCount"] == 1
    assert second_metadata["failedFrameCount"] == 0
    assert second_resolved == first_resolved
    assert second_batches == first_batches
    assert any("explicit fallback on 1/3 frames" in item for item in second_warnings)


def test_checkpoints_continue_after_a_degraded_frame_and_resume_restores_state(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector_input = {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {"name": "ball.pt", "size": 123, "mtimeNs": 456},
        "confidence": 0.05,
        "tileSize": 640,
    }
    flaky = FlakyPrimaryDetector(fail_frames={0})
    fallback = RecordingDetector("generic-ultralytics", candidate=True)

    def stop_after_two(completed, _total, _detail):
        if completed == 2:
            raise RuntimeError("cancel checkpoint")

    with pytest.raises(RuntimeError, match="cancel checkpoint"):
        _detect_ball_frames(
            _scene(),
            flaky,
            fallback,
            [],
            [],
            on_progress=stop_after_two,
            detector_input=detector_input,
        )

    resumed_primary = RecordingDetector("dedicated-ultralytics", candidate=True)
    resumed_fallback = RecordingDetector(
        "generic-ultralytics",
        fail=AssertionError("fallback must not run after the primary recovered"),
    )
    resolved, metadata, batches, _warnings = _detect_ball_frames(
        _scene(),
        resumed_primary,
        resumed_fallback,
        [],
        [],
        detector_input=detector_input,
    )

    # The degraded prefix was checkpointed, so only the last frame is computed.
    assert metadata["detectionCheckpointHit"] is True
    assert metadata["resumedFrameCount"] == 2
    assert [call["frameIndex"] for call in resumed_primary.calls] == [2]
    assert len(resolved) == len(batches) == 3
    assert metadata["fallbackFrameCount"] == 2
    assert metadata["detectionCacheStored"] is True


def test_interrupted_clean_detection_resumes_from_periodic_checkpoint(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector_input = {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {"name": "ball.pt", "size": 123, "mtimeNs": 456},
        "confidence": 0.05,
        "tileSize": 640,
    }
    interrupted = RecordingDetector("dedicated-ultralytics", candidate=True)

    def stop_after_two(completed, _total, _detail):
        if completed == 2:
            raise RuntimeError("cancel checkpoint")

    with pytest.raises(RuntimeError, match="cancel checkpoint"):
        _detect_ball_frames(
            _scene(),
            interrupted,
            None,
            [],
            [],
            on_progress=stop_after_two,
            detector_input=detector_input,
        )
    assert len(interrupted.calls) == 2

    resumed_detector = RecordingDetector("dedicated-ultralytics", candidate=True)
    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(),
        resumed_detector,
        None,
        [],
        [],
        detector_input=detector_input,
    )

    assert len(resumed_detector.calls) == 1
    assert len(resolved) == len(batches) == 3
    assert metadata["detectionCheckpointHit"] is True
    assert metadata["resumedFrameCount"] == 2
    assert metadata["detectionCacheStored"] is True
    assert warnings == []


def test_adaptive_checkpoint_resume_reconstructs_identical_seed_and_schedule(
    monkeypatch, tmp_path
):
    uninterrupted_dense = _dense_frames(tmp_path / "uninterrupted", count=7)
    _patch_runtime(monkeypatch, uninterrupted_dense)
    uninterrupted = AdaptiveRecordingDetector()
    full_resolved, _, full_batches, _ = _detect_ball_frames(
        _scene(),
        uninterrupted,
        None,
        [],
        [],
        detector_input=_adaptive_input(),
    )

    resumed_dense = _dense_frames(tmp_path / "resumed", count=7)
    _patch_runtime(monkeypatch, resumed_dense)
    interrupted = AdaptiveRecordingDetector()

    def stop_after_four(completed, _total, _detail):
        if completed == 4:
            raise RuntimeError("cancel adaptive checkpoint")

    with pytest.raises(RuntimeError, match="cancel adaptive checkpoint"):
        _detect_ball_frames(
            _scene(),
            interrupted,
            None,
            [],
            [],
            on_progress=stop_after_four,
            detector_input=_adaptive_input(),
        )

    resumed = AdaptiveRecordingDetector()
    resumed_resolved, metadata, resumed_batches, warnings = _detect_ball_frames(
        _scene(),
        resumed,
        None,
        [],
        [],
        detector_input=_adaptive_input(),
    )

    assert metadata["detectionCheckpointHit"] is True
    assert metadata["resumedFrameCount"] == 4
    assert resumed_resolved == full_resolved
    assert [item["scanMode"] for item in resumed_batches] == [
        item["scanMode"] for item in full_batches
    ]
    assert [call["frameIndex"] for call in resumed.roi_calls] == [4]
    assert resumed.roi_calls[0]["regions"] == uninterrupted.roi_calls[3]["regions"]
    assert [call["frameIndex"] for call in resumed.global_calls] == [5, 6]
    assert warnings == []


def test_missing_seed_and_previous_failure_force_global_reacquisition(
    monkeypatch, tmp_path
):
    no_seed_dense = _dense_frames(tmp_path / "no-seed", count=4)
    _patch_runtime(monkeypatch, no_seed_dense)
    no_seed_detector = AdaptiveRecordingDetector(global_empty_frames={0})
    _, _, no_seed_batches, _ = _detect_ball_frames(
        _scene(),
        no_seed_detector,
        None,
        [],
        [],
        detector_input=_adaptive_input(),
    )
    assert [call["frameIndex"] for call in no_seed_detector.global_calls] == [0, 1, 3]
    assert no_seed_batches[1]["metadata"]["globalScanReason"] == "no-seed"

    failed_dense = _dense_frames(tmp_path / "failure", count=4)
    _patch_runtime(monkeypatch, failed_dense)
    failed_detector = AdaptiveRecordingDetector(global_error_frames={0})
    _, _, failed_batches, warnings = _detect_ball_frames(
        _scene(),
        failed_detector,
        None,
        [],
        [],
        detector_input=_adaptive_input(),
    )
    assert [call["frameIndex"] for call in failed_detector.global_calls] == [0, 1, 3]
    assert failed_batches[1]["metadata"]["globalScanReason"] == (
        "previous-frame-fallback"
    )
    assert any("failed on 1/4 frames" in warning for warning in warnings)


@pytest.mark.parametrize("backend", ["generic-ultralytics", "wasb-service"])
def test_adaptive_roi_is_never_applied_to_other_backends(
    monkeypatch, tmp_path, backend
):
    dense = _dense_frames(tmp_path / backend, count=3)
    _patch_runtime(monkeypatch, dense)
    detector = AdaptiveRecordingDetector(backend)
    detector_input = {**_adaptive_input(), "backend": backend}

    _, metadata, _, _ = _detect_ball_frames(
        _scene(), detector, None, [], [], detector_input=detector_input
    )

    assert [call["frameIndex"] for call in detector.global_calls] == [0, 1, 2]
    assert detector.roi_calls == []
    assert metadata["adaptiveRoi"] == {"enabled": False}


def test_first_primary_failure_opens_circuit_until_the_half_open_probe(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    primary = RecordingDetector("wasb-service", fail=RuntimeError("worker offline"))
    fallback = RecordingDetector("dedicated-ultralytics", candidate=True)

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(), primary, fallback, [], []
    )

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 3
    assert all(len(candidates) == 1 for candidates, _ in resolved)
    assert metadata["failedFrameCount"] == 0
    assert metadata["fallbackFrameCount"] == 3
    assert metadata["circuitBreaker"] == {
        "opened": True,
        "reason": "RuntimeError: worker offline",
        "retryIntervalFrames": 2,
        "openedCount": 1,
        "transitions": [
            {
                "frameIndex": 0,
                "event": "opened",
                "reason": "RuntimeError: worker offline",
            }
        ],
    }
    assert metadata["backendCounts"] == {"dedicated-ultralytics": 3}
    assert batches[0]["fallbackReason"] == "RuntimeError: worker offline"
    assert batches[1]["fallbackReason"] == (
        "circuit-open after RuntimeError: worker offline"
    )
    assert batches[2]["fallbackReason"] == batches[1]["fallbackReason"]
    assert any("explicit fallback on 3/3 frames" in warning for warning in warnings)


def test_half_open_probe_restores_the_primary_detector_after_recovery(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path, count=5)
    _patch_runtime(monkeypatch, dense)
    # The primary fails only on frame 0; the circuit serves the fallback for
    # retryIntervalFrames=2 frames, then the half-open probe succeeds.
    primary = FlakyPrimaryDetector(fail_frames={0})
    fallback = RecordingDetector("generic-ultralytics", candidate=True)

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(), primary, fallback, [], []
    )

    assert [call["frameIndex"] for call in primary.calls] == [0, 3, 4]
    assert [call["frameIndex"] for call in fallback.calls] == [0, 1, 2]
    assert len(resolved) == 5
    assert metadata["fallbackFrameCount"] == 3
    assert metadata["failedFrameCount"] == 0
    assert metadata["backendCounts"] == {
        "dedicated-ultralytics": 2,
        "generic-ultralytics": 3,
    }
    assert metadata["circuitBreaker"]["opened"] is False
    assert metadata["circuitBreaker"]["reason"] is None
    assert metadata["circuitBreaker"]["openedCount"] == 1
    assert metadata["circuitBreaker"]["transitions"] == [
        {
            "frameIndex": 0,
            "event": "opened",
            "reason": "RuntimeError: primary detector outage",
        },
        {"frameIndex": 3, "event": "closed", "reason": None},
    ]
    assert batches[3]["fallbackReason"] is None
    assert batches[4]["fallbackReason"] is None
    assert any("explicit fallback on 3/5 frames" in warning for warning in warnings)


def test_failed_detector_maps_one_generic_fallback_observation_to_one_dense_frame(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector = RecordingDetector("broken", fail=RuntimeError("no model"))
    generic_fallback = [
        (
            [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "confidence": 0.6,
                    "bbox": [98.0, 198.0, 102.0, 202.0],
                }
            ],
            0.04,
        )
    ]

    resolved, metadata, batches, warnings = _detect_ball_frames(
        _scene(), detector, None, [], generic_fallback
    )

    assert [len(candidates) for candidates, _ in resolved] == [0, 1, 0]
    assert sum(len(candidates) for candidates, _ in resolved) == 1
    assert resolved[1][0][0]["candidateId"] == "ball-f00001-generic-01"
    assert [
        batch["metadata"]["genericFallbackCandidateAccepted"] for batch in batches
    ] == [
        False,
        True,
        False,
    ]
    assert metadata["failedFrameCount"] == 3
    assert metadata["backendCounts"] == {"generic-coco-fallback": 3}
    assert any("failed on 3/3 frames" in warning for warning in warnings)


def test_raise_policy_propagates_primary_failure_without_calling_fallback(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense, failure_policy="raise")
    primary = RecordingDetector("wasb-service", fail=TimeoutError("timed out"))
    fallback = RecordingDetector("dedicated-ultralytics", candidate=True)

    with pytest.raises(
        ReconstructionError,
        match=r"Ball detector wasb-service failed on dense frame 1/3: TimeoutError: timed out",
    ):
        _detect_ball_frames(_scene(), primary, fallback, [], [])

    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_strict_wasb_does_not_load_unrelated_dedicated_checkpoint(monkeypatch):
    settings = SimpleNamespace(
        ball_detection_failure_policy="raise",
        reconstruction_device="cpu",
        ball_detection_max_candidates=12,
        ball_detection_nms_iou=0.1,
        ball_wasb_worker_url="http://ball-worker:8092/v1/detections",
        ball_wasb_timeout=30.0,
    )
    built: list[str] = []

    def build(config, **_kwargs):
        built.append(config.backend)
        return RecordingDetector(config.backend)

    monkeypatch.setattr(
        "app.reconstruction_ball_detector_selection.get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "app.reconstruction_ball_detector_selection.build_ball_detector",
        build,
    )
    monkeypatch.setattr(
        "app.reconstruction_ball_detector_selection.load_model",
        lambda *_: (_ for _ in ()).throw(AssertionError("dedicated model loaded")),
    )

    detector, fallback = _configured_ball_detectors(
        object(),
        "wasb-service",
        {
            "schemaVersion": 1,
            "backend": "wasb-service",
            "failurePolicy": "raise",
            "maxCandidates": 12,
            "nmsIou": 0.1,
            "workerEndpoint": "http://ball-worker:8092/v1/detections",
            "timeoutSeconds": 30.0,
        },
    )

    assert detector.backend_name == "wasb-service"
    assert fallback is None
    assert built == ["generic-ultralytics", "wasb-service"]


def test_queued_checkpoint_identity_ignores_mtime_only_changes(tmp_path):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"same-weights")
    expected = {
        "name": "ball.pt",
        "size": checkpoint.stat().st_size,
        "sha256": "5a73f55361f1162bd6cd93ba97de41c42e77c6a9539a29174fec90ef5265fd20",
    }

    stat = checkpoint.stat()
    os.utime(
        checkpoint,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )

    _verify_queued_ball_checkpoint(checkpoint, expected)


def test_queued_checkpoint_identity_fails_closed_when_local_weights_changed(tmp_path):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"old-weights")
    expected = {
        "name": "ball.pt",
        "size": checkpoint.stat().st_size,
        "sha256": "48aab6e93d3a7db5733fc8aaf39bbab1b6354f7a775010f4d36fc79dabc65264",
    }
    checkpoint.write_bytes(b"new-weights")

    with pytest.raises(ReconstructionError, match="no longer matches"):
        _verify_queued_ball_checkpoint(checkpoint, expected)


def test_queued_checkpoint_without_content_hash_requires_a_new_run(tmp_path):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(ReconstructionError, match="has no content hash"):
        _verify_queued_ball_checkpoint(
            checkpoint,
            {
                "name": "ball.pt",
                "size": checkpoint.stat().st_size,
                "mtimeNs": checkpoint.stat().st_mtime_ns,
            },
        )


@pytest.mark.parametrize("expected", [None, "ball.pt", []])
def test_missing_queued_checkpoint_identity_requires_a_new_run(tmp_path, expected):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(ReconstructionError, match="identity is missing"):
        _verify_queued_ball_checkpoint(checkpoint, expected)


def test_world_projection_is_published_only_for_real_keyframes():
    assert _ball_world_projection_status("metric", [{"t": 0.0}]) == "published"
    assert _ball_world_projection_status("metric", []) == "no-stable-trajectory"
    assert _ball_world_projection_status("unavailable", [{"t": 0.0}]) == "calibration-rejected"


def test_frame_analysis_uses_one_sampled_frame_for_video_people_and_ball(
    monkeypatch,
    tmp_path,
):
    frames = [
        (tmp_path / "frame_00101.jpg", 0.0),
        (tmp_path / "frame_00102.jpg", 0.1),
    ]
    detector = RecordingDetector("dedicated-ultralytics", candidate=True)
    result = SimpleNamespace(orig_img=np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "frame-sync",
        "duration": 1.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "tracks": [],
            "canonicalPeople": [],
            "ball": {"keyframes": []},
            "videoAsset": {
                "sourceStart": 0.0,
                "reconstruction": {
                    "status": "ready",
                    "model": "test-model",
                    "ballBackend": "dedicated-ultralytics",
                    "ballDetectionInput": {
                        "backend": "dedicated-ultralytics",
                        "failurePolicy": "raise",
                    },
                    "pitchCalibration": {"status": "fallback"},
                },
            },
        },
    }
    monkeypatch.setattr("app.reconstruction_frame_analysis.frame_paths", lambda _: frames)
    monkeypatch.setattr("app.reconstruction_frame_analysis.load_model", lambda _: object())
    monkeypatch.setattr("app.ultralytics_person_inference.predict_frame", lambda *_: result)
    monkeypatch.setattr(
        "app.ultralytics_person_inference.parse_person_detections",
        lambda _: ([], []),
    )
    monkeypatch.setattr(
            "app.reconstruction_frame_ball_analysis.configured_ball_detectors",
        lambda *_: (detector, None),
    )
    monkeypatch.setattr("app.reconstruction_frame_context.cv2.imread", lambda *_: None)

    analysis = analyze_scene_frame(scene, 0.08)

    assert detector.calls[0]["frame"] == frames[1][0]
    assert detector.calls[0]["timestamp"] == pytest.approx(0.1)
    assert analysis["sceneTime"] == analysis["ballSceneTime"] == 0.1
    assert analysis["frameIndex"] == 102
    assert analysis["ballCandidates"][0]["primary"] is True
