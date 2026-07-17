from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ball_detection import BallCandidate, BallDetectionBatch
from app.ball_frames import DenseBallFrameSet
from app.reconstruction import (
    ReconstructionError,
    _ball_world_projection_status,
    _configured_ball_detectors,
    _detect_ball_frames,
    _verify_queued_ball_checkpoint,
    analyze_scene_frame,
)


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
    monkeypatch.setattr("app.reconstruction.dense_ball_frame_paths", lambda _: dense)
    monkeypatch.setattr(
        "app.reconstruction.get_settings",
        lambda: SimpleNamespace(
            ball_detection_failure_policy=failure_policy,
            reconstruction_frame_rate=10.0,
        ),
    )


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


def test_clean_dense_detection_is_reused_from_raw_candidate_cache(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    monkeypatch.setattr("app.reconstruction.dense_ball_frame_paths", lambda _: dense)
    monkeypatch.setattr(
        "app.reconstruction.get_settings",
        lambda: SimpleNamespace(
            ball_detection_failure_policy="fallback",
            reconstruction_frame_rate=10.0,
            media_root=str(tmp_path),
        ),
    )
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


def test_first_primary_failure_opens_circuit_and_all_frames_use_fallback(
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
    }
    assert metadata["backendCounts"] == {"dedicated-ultralytics": 3}
    assert batches[0]["fallbackReason"] == "RuntimeError: worker offline"
    assert batches[1]["fallbackReason"] == (
        "circuit-open after RuntimeError: worker offline"
    )
    assert batches[2]["fallbackReason"] == batches[1]["fallbackReason"]
    assert any("explicit fallback on 3/3 frames" in warning for warning in warnings)


def test_failed_detector_maps_one_legacy_observation_to_only_one_dense_frame(
    monkeypatch, tmp_path
):
    dense = _dense_frames(tmp_path)
    _patch_runtime(monkeypatch, dense)
    detector = RecordingDetector("broken", fail=RuntimeError("no model"))
    legacy = [
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
        _scene(), detector, None, [], legacy
    )

    assert [len(candidates) for candidates, _ in resolved] == [0, 1, 0]
    assert sum(len(candidates) for candidates, _ in resolved) == 1
    assert resolved[1][0][0]["candidateId"] == "ball-f00001-legacy-01"
    assert [batch["metadata"]["legacyCandidateAccepted"] for batch in batches] == [
        False,
        True,
        False,
    ]
    assert metadata["failedFrameCount"] == 3
    assert metadata["backendCounts"] == {"legacy-coco-fallback": 3}
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
        ball_wasb_worker_url="http://ball-worker:8092/detect",
        ball_wasb_timeout=30.0,
    )
    built: list[str] = []

    def build(config, **_kwargs):
        built.append(config.backend)
        return RecordingDetector(config.backend)

    monkeypatch.setattr("app.reconstruction.get_settings", lambda: settings)
    monkeypatch.setattr("app.reconstruction.build_ball_detector", build)
    monkeypatch.setattr(
        "app.reconstruction._load_model",
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
            "workerEndpoint": "http://ball-worker:8092/detect",
            "timeoutSeconds": 30.0,
        },
    )

    assert detector.backend_name == "wasb-service"
    assert fallback is None
    assert built == ["generic-ultralytics", "wasb-service"]


def test_queued_checkpoint_identity_fails_closed_when_local_weights_changed(tmp_path):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"new-weights")

    with pytest.raises(ReconstructionError, match="no longer matches"):
        _verify_queued_ball_checkpoint(
            checkpoint,
            {"name": "ball.pt", "size": 2, "mtimeNs": checkpoint.stat().st_mtime_ns},
        )


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
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: frames)
    monkeypatch.setattr("app.reconstruction._load_model", lambda _: object())
    monkeypatch.setattr("app.reconstruction._predict_frame", lambda *_: result)
    monkeypatch.setattr("app.reconstruction._person_detections", lambda _: ([], []))
    monkeypatch.setattr(
        "app.reconstruction._configured_ball_detectors",
        lambda *_: (detector, None),
    )
    monkeypatch.setattr("app.reconstruction.cv2.imread", lambda *_: None)
    monkeypatch.setattr(
        "app.reconstruction.dense_ball_frame_paths",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("interactive frame analysis mixed in a dense frame")
        ),
    )

    analysis = analyze_scene_frame(scene, 0.08)

    assert detector.calls[0]["frame"] == frames[1][0]
    assert detector.calls[0]["timestamp"] == pytest.approx(0.1)
    assert analysis["sceneTime"] == analysis["ballSceneTime"] == 0.1
    assert analysis["frameIndex"] == 102
    assert analysis["ballCandidates"][0]["primary"] is True
