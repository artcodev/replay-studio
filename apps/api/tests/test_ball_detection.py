from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ball_detection import (
    BallCandidate,
    BallDetectionBatch,
    BallDetectorConfig,
    BallDetectorConfigurationError,
    BallDetectorUnavailable,
    UltralyticsBallDetector,
    UltralyticsBallDetectorConfig,
    WasbServiceBallDetector,
    WasbSubprocessBallDetector,
    build_ball_detector,
    parse_ultralytics_ball_candidates,
)


def _result(
    image: np.ndarray,
    boxes: list[list[float]],
    confidences: list[float],
    classes: list[int],
):
    return SimpleNamespace(
        orig_img=image,
        boxes=SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(confidences, dtype=np.float32),
            cls=np.asarray(classes, dtype=np.float32),
        ),
        names={0: "person", 32: "sports ball"},
    )


class _FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        batch_size = len(source) if isinstance(source, list) else 1
        return [self.outputs.pop(0) for _ in range(batch_size)]


def test_generic_parser_keeps_only_requested_class_and_preserves_metadata():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    result = _result(
        image,
        [[10, 10, 20, 30], [40, 20, 48, 28]],
        [0.99, 0.76],
        [0, 32],
    )

    candidates = parse_ultralytics_ball_candidates(
        result,
        backend_name="generic-ultralytics",
        class_ids=(32,),
        frame_index=7,
        timestamp=1.4,
    )

    assert len(candidates) == 1
    assert candidates[0].bbox == (40.0, 20.0, 48.0, 28.0)
    assert candidates[0].x == 44.0
    assert candidates[0].metadata == {
        "detectionIndex": 1,
        "className": "sports ball",
        "frameIndex": 7,
        "timestamp": 1.4,
    }
    assert candidates[0].as_reconstruction_detection()["detectorBackend"] == (
        "generic-ultralytics"
    )


def test_dedicated_detector_offsets_tiles_suppresses_duplicates_and_applies_top_k():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    first_tile = image[:, :60]
    second_tile = image[:, 40:]
    model = _FakeModel(
        [
            _result(first_tile, [[50, 20, 58, 28], [5, 5, 9, 9]], [0.80, 0.20], [0, 0]),
            _result(second_tile, [[10, 20, 18, 28], [40, 40, 46, 46]], [0.95, 0.70], [0, 0]),
        ]
    )
    detector = UltralyticsBallDetector(
        UltralyticsBallDetectorConfig(
            backend_name="dedicated-ultralytics",
            class_ids=(0,),
            image_size=640,
            max_candidates=2,
            tile_size=(60, 60),
            tile_overlap=1 / 3,
            nms_iou=0.1,
        ),
        model=model,
    )

    batch = detector.detect(image, frame_index=3)

    assert batch.image_size == (100, 60)
    assert batch.metadata["tileCount"] == 2
    assert batch.metadata["rawCandidateCount"] == 4
    assert [candidate.bbox for candidate in batch.candidates] == [
        (50.0, 20.0, 58.0, 28.0),
        (80.0, 40.0, 86.0, 46.0),
    ]
    assert batch.candidates[0].confidence == pytest.approx(0.95)
    assert batch.candidates[0].metadata["tile"]["x"] == 40
    assert batch.metadata["inferenceBatchCount"] == 1
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 2
    assert all(call[1]["classes"] == [0] for call in model.calls)


def test_tiled_detector_respects_configured_inference_batch_size():
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    model = _FakeModel(
        [
            _result(np.zeros((40, 40, 3), dtype=np.uint8), [], [], [])
            for _ in range(5)
        ]
    )
    detector = UltralyticsBallDetector(
        UltralyticsBallDetectorConfig(
            backend_name="dedicated-ultralytics",
            class_ids=(0,),
            tile_size=(40, 40),
            tile_overlap=0.5,
            inference_batch_size=2,
        ),
        model=model,
    )

    batch = detector.detect(image)

    assert batch.metadata["tileCount"] == 5
    assert batch.metadata["inferenceBatchSize"] == 2
    assert batch.metadata["inferenceBatchCount"] == 3
    assert [len(call[0]) for call in model.calls] == [2, 2, 1]


def test_detector_refuses_missing_checkpoint_instead_of_triggering_download(tmp_path: Path):
    loader_called = False

    def loader(_checkpoint: str):
        nonlocal loader_called
        loader_called = True

    with pytest.raises(BallDetectorConfigurationError, match="does not exist"):
        UltralyticsBallDetector(
            UltralyticsBallDetectorConfig(),
            checkpoint_path=tmp_path / "football-ball-detection.pt",
            model_loader=loader,
        )

    assert loader_called is False


def test_factory_builds_current_generic_and_dedicated_profiles():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    generic_model = _FakeModel([_result(image, [[1, 1, 5, 5]], [0.8], [32])])
    generic = build_ball_detector(
        BallDetectorConfig(backend="generic-ultralytics"),
        model=generic_model,
    )
    assert generic.detect(image).candidates[0].class_id == 32
    assert generic_model.calls[0][1]["classes"] == [32]
    assert generic_model.calls[0][1]["imgsz"] == 1280

    dedicated_model = _FakeModel([_result(image, [[1, 1, 5, 5]], [0.8], [0])])
    dedicated = build_ball_detector(
        BallDetectorConfig(
            backend="dedicated-ultralytics",
            tile_size=(640, 640),
        ),
        model=dedicated_model,
    )
    assert dedicated.detect(image).candidates[0].class_id == 0
    assert dedicated_model.calls[0][1]["classes"] == [0]
    assert dedicated_model.calls[0][1]["imgsz"] == 640


class _FallbackDetector:
    backend_name = "fallback-test"

    def detect(self, frame, **_kwargs):
        return BallDetectionBatch(
            candidates=(
                BallCandidate(
                    bbox=(1.0, 2.0, 5.0, 6.0),
                    confidence=0.6,
                    backend=self.backend_name,
                ),
            ),
            image_size=(20, 10),
            backend=self.backend_name,
            metadata={"fallback": True},
        )


def test_wasb_service_sends_temporal_context_and_parses_candidates():
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {
            "imageSize": [100, 60],
            "candidates": [
                {
                    "position": [50, 30],
                    "radius": 3,
                    "confidence": 0.91,
                    "temporalScore": 0.88,
                }
            ],
            "metadata": {"model": "wasb_soccer_best"},
        }

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/detect",
        timeout=12.0,
        transport=transport,
    )
    context = np.zeros((60, 100, 3), dtype=np.uint8)
    frame = np.ones((60, 100, 3), dtype=np.uint8)

    batch = detector.detect(frame, frame_index=4, timestamp=0.4, context_frames=(context,))

    assert captured["url"] == "http://wasb-worker:8092/detect"
    assert captured["timeout"] == 12.0
    assert captured["payload"]["targetIndex"] == 1
    assert len(captured["payload"]["frames"]) == 2
    assert batch.candidates[0].bbox == (47.0, 27.0, 53.0, 33.0)
    assert batch.candidates[0].metadata["temporalScore"] == 0.88
    assert batch.metadata["worker"] == {"model": "wasb_soccer_best"}


def test_wasb_service_centres_offline_previous_and_next_context():
    captured = {}

    def transport(_url, payload, _timeout):
        captured["payload"] = payload
        return {"imageSize": [20, 10], "candidates": []}

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/detect",
        transport=transport,
    )
    previous = np.full((10, 20, 3), 1, dtype=np.uint8)
    current = np.full((10, 20, 3), 2, dtype=np.uint8)
    following = np.full((10, 20, 3), 3, dtype=np.uint8)

    batch = detector.detect(current, context_frames=(previous, following))

    assert captured["payload"]["targetIndex"] == 1
    assert len(captured["payload"]["frames"]) == 3
    assert batch.metadata["temporalContextMode"] == "centered"


def test_wasb_failure_is_explicit_or_uses_configured_fallback():
    def broken_transport(_url, _payload, _timeout):
        raise TimeoutError("worker timeout")

    strict = WasbServiceBallDetector(
        "http://wasb-worker:8092/detect",
        transport=broken_transport,
    )
    with pytest.raises(BallDetectorUnavailable, match="worker timeout"):
        strict.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    fallback = WasbServiceBallDetector(
        "http://wasb-worker:8092/detect",
        failure_policy="fallback",
        fallback=_FallbackDetector(),
        transport=broken_transport,
    )
    batch = fallback.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    assert batch.backend == "fallback-test"
    assert batch.metadata["fallback"] is True
    assert batch.metadata["requestedBackend"] == "wasb-service"
    assert "worker timeout" in batch.metadata["fallbackReason"]


def test_wasb_fallback_policy_requires_a_detector():
    with pytest.raises(BallDetectorConfigurationError, match="explicit fallback"):
        WasbServiceBallDetector(
            "http://wasb-worker:8092/detect",
            failure_policy="fallback",
        )


def test_wasb_subprocess_uses_argument_vector_without_shell():
    captured = {}

    def transport(command, payload, timeout):
        captured.update(command=command, payload=payload, timeout=timeout)
        return {
            "imageSize": [20, 10],
            "candidates": [{"x": 8, "y": 4, "confidence": 0.7}],
        }

    detector = WasbSubprocessBallDetector(
        ("python", "wasb_adapter.py"),
        transport=transport,
    )
    batch = detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    assert captured["command"] == ("python", "wasb_adapter.py")
    assert captured["payload"]["contractVersion"] == 1
    assert batch.candidates[0].x == 8.0
    assert batch.candidates[0].y == 4.0
