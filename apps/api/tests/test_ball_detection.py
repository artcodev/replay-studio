import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ball_candidate_selection import parse_ultralytics_ball_candidates
from app.ball_detection_contract import (
    BallCandidate,
    BallDetectionBatch,
    BallDetectorConfig,
    BallDetectorConfigurationError,
    BallDetectorUnavailable,
    UltralyticsBallDetectorConfig,
)
from app.ball_detector_factory import build_ball_detector
from app.ultralytics_ball_detector import UltralyticsBallDetector
from app.wasb_ball_detector import (
    WasbServiceBallDetector,
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


def test_region_detector_preserves_full_frame_coordinates_at_border_and_nms():
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    left_crop = image[0:60, 0:60]
    border_crop = image[20:60, 40:100]
    model = _FakeModel(
        [
            _result(left_crop, [[50, 25, 58, 33]], [0.80], [0]),
            # Same full-frame box after the x/y region offset. Cross-region
            # NMS must collapse it, even though the second crop touches both
            # the right and bottom image borders.
            _result(border_crop, [[10, 5, 18, 13]], [0.95], [0]),
        ]
    )
    detector = UltralyticsBallDetector(
        UltralyticsBallDetectorConfig(
            backend_name="dedicated-ultralytics",
            class_ids=(0,),
            tile_size=(60, 60),
            inference_batch_size=8,
            nms_iou=0.1,
        ),
        model=model,
    )

    batch = detector.detect_regions(
        image,
        [(0, 0, 60, 60), (40, 20, 120, 90)],
        frame_index=9,
        timestamp=0.36,
    )

    assert batch.image_size == (100, 60)
    assert batch.metadata == {
        "rawCandidateCount": 2,
        "tileCount": 2,
        "roiRegionCount": 2,
        "roiRegions": [[0, 0, 60, 60], [40, 20, 100, 60]],
        "inferenceBatchSize": 8,
        "inferenceBatchCount": 1,
        "scanMode": "roi",
    }
    assert len(batch.candidates) == 1
    assert batch.candidates[0].bbox == (50.0, 25.0, 58.0, 33.0)
    assert batch.candidates[0].confidence == pytest.approx(0.95)
    assert batch.candidates[0].metadata["tile"] == {
        "index": 1,
        "x": 40,
        "y": 20,
        "width": 60,
        "height": 40,
    }


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


def _wasb_response(request, candidates=None):
    frames = []
    for index, item in enumerate(request.manifest["frames"]):
        frames.append(
            {
                **item,
                "imageSize": [100, 60],
                "temporalPadding": len(set(
                    frame["fileIndex"] for frame in request.manifest["frames"]
                )) < 3,
                "candidates": (
                    list(candidates or [])
                    if index == request.target_index
                    else []
                ),
            }
        )
    return {
        "contractVersion": 1,
        "backend": "wasb-sbdt-soccer",
        "modelVersion": "wasb-soccer@sha256:test",
        "frames": frames,
        "metadata": {"model": "wasb_soccer_best"},
    }


def test_wasb_service_sends_temporal_context_and_parses_candidates():
    captured = {}

    def transport(url, request, timeout):
        captured.update(url=url, request=request, timeout=timeout)
        return _wasb_response(
            request,
            [
                {
                    "position": [50, 30],
                    "radius": 3,
                    "confidence": 0.91,
                    "temporalScore": 0.88,
                    "modelVersion": "wasb-soccer@sha256:test",
                }
            ],
        )

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        timeout=12.0,
        transport=transport,
    )
    context = np.zeros((60, 100, 3), dtype=np.uint8)
    frame = np.ones((60, 100, 3), dtype=np.uint8)

    batch = detector.detect(frame, frame_index=4, timestamp=0.4, context_frames=(context,))

    assert captured["url"] == "http://wasb-worker:8092/v1/detections"
    assert captured["timeout"] == 12.0
    request = captured["request"]
    assert request.target_index == 2
    assert request.manifest["targetIndex"] == 2
    assert [item["fileIndex"] for item in request.manifest["frames"]] == [0, 0, 1]
    assert len(request.uploads) == 2
    assert batch.candidates[0].bbox == (47.0, 27.0, 53.0, 33.0)
    assert batch.candidates[0].metadata["temporalScore"] == 0.88
    assert batch.metadata["temporalContextMode"] == "causal"
    assert batch.metadata["worker"]["model"] == "wasb_soccer_best"
    assert batch.metadata["worker"]["modelVersion"] == "wasb-soccer@sha256:test"


def test_wasb_detect_sequence_uploads_each_frame_once_and_parses_all(monkeypatch):
    captured = {}

    def transport(url, request, timeout):
        captured.update(url=url, request=request, timeout=timeout)
        return _wasb_response_all_frames(request)

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        timeout=12.0,
        transport=transport,
    )
    frames = [
        (np.full((60, 100, 3), index, dtype=np.uint8), 10 + index, index / 25.0)
        for index in range(5)
    ]

    batches = detector.detect_sequence(frames)

    request = captured["request"]
    assert len(request.uploads) == 5
    assert [item["fileIndex"] for item in request.manifest["frames"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [item["frameIndex"] for item in request.manifest["frames"]] == [
        10,
        11,
        12,
        13,
        14,
    ]
    assert "targetIndex" not in request.manifest
    assert len(batches) == 5
    assert all(batch.backend == "wasb-service" for batch in batches)
    assert batches[2].candidates[0].metadata["frameIndex"] == 12
    assert batches[0].metadata["temporalContextMode"] == "tiled-window-sequence"
    assert batches[3].metadata["worker"]["sequenceIndex"] == 3


def _wasb_response_all_frames(request):
    frames = []
    for item in request.manifest["frames"]:
        frames.append(
            {
                **item,
                "imageSize": [100, 60],
                "temporalPadding": False,
                "candidates": [
                    {
                        "position": [50, 30],
                        "radius": 3,
                        "confidence": 0.9,
                    }
                ],
            }
        )
    return {
        "contractVersion": 1,
        "backend": "wasb-sbdt-soccer",
        "modelVersion": "wasb-soccer@sha256:test",
        "frames": frames,
        "metadata": {"model": "wasb_soccer_best"},
    }


def test_wasb_detect_sequence_failure_raises_without_internal_fallback():
    def broken_transport(_url, _request, _timeout):
        raise ConnectionError("worker offline")

    fallback = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        transport=lambda *_: (_ for _ in ()).throw(
            AssertionError("fallback must not be consulted by detect_sequence")
        ),
    )
    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        transport=broken_transport,
        failure_policy="fallback",
        fallback=fallback,
    )

    with pytest.raises(BallDetectorUnavailable, match="worker offline"):
        detector.detect_sequence([(np.zeros((60, 100, 3), dtype=np.uint8), 0, 0.0)])


def test_wasb_service_centres_offline_previous_and_next_context():
    captured = {}

    def transport(_url, request, _timeout):
        captured["request"] = request
        return _wasb_response(request)

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        transport=transport,
    )
    previous = np.full((10, 20, 3), 1, dtype=np.uint8)
    current = np.full((10, 20, 3), 2, dtype=np.uint8)
    following = np.full((10, 20, 3), 3, dtype=np.uint8)

    batch = detector.detect(current, context_frames=(previous, following))

    request = captured["request"]
    assert request.target_index == 1
    assert [item["fileIndex"] for item in request.manifest["frames"]] == [0, 1, 2]
    assert len(request.uploads) == 3
    assert batch.metadata["temporalContextMode"] == "centered"


def test_wasb_failure_is_explicit_or_uses_configured_fallback():
    def broken_transport(_url, _request, _timeout):
        raise TimeoutError("worker timeout")

    strict = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        transport=broken_transport,
    )
    with pytest.raises(BallDetectorUnavailable, match="worker timeout"):
        strict.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    fallback = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
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
            "http://wasb-worker:8092/v1/detections",
            failure_policy="fallback",
        )


def test_wasb_service_rejects_retired_detection_endpoint():
    with pytest.raises(
        BallDetectorConfigurationError,
        match="must target /v1/detections",
    ):
        WasbServiceBallDetector("http://wasb-worker:8092/detect")


def test_wasb_service_rejects_partial_or_wrong_backend_responses():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    def wrong_backend(_url, request, _timeout):
        payload = _wasb_response(request)
        payload["backend"] = "some-other-model"
        return payload

    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections",
        transport=wrong_backend,
    )
    with pytest.raises(BallDetectorUnavailable, match="unexpected backend"):
        detector.detect(frame)


def test_wasb_http_transport_serializes_multipart_without_base64(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            manifest = json.loads(captured["data"]["manifest"])
            request = SimpleNamespace(
                manifest=manifest,
                target_index=manifest["targetIndex"],
            )
            return _wasb_response(request)

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("app.wasb_ball_transport.httpx.post", post)
    detector = WasbServiceBallDetector(
        "http://wasb-worker:8092/v1/detections"
    )
    detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    assert captured["url"].endswith("/v1/detections")
    assert len(captured["files"]) == 1
    field_name, (filename, image_bytes, media_type) = captured["files"][0]
    assert field_name == "frames"
    assert filename.endswith(".png")
    assert image_bytes.startswith(b"\x89PNG")
    assert media_type == "image/png"
    manifest = captured["data"]["manifest"]
    assert "dataBase64" not in manifest
    assert '"targetIndex":1' in manifest
