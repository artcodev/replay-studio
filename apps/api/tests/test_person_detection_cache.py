from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import person_detection_cache as cache
from app import person_base_detection_cache as base_cache
from app import person_detector_provenance as provenance
from app import reconstruction_reid_evidence as identity_evidence
from app import reconstruction_person_annotations as person_annotations
from app import ultralytics_person_inference as person_inference


def _detector_input(model: str = "players-v1.pt") -> dict:
    return {
        "schemaVersion": 1,
        "provider": {"backend": "ultralytics-yolo", "version": "9.0.0"},
        "checkpoint": {"name": model, "sha256": f"sha-{model}"},
        "classes": {"person": 0, "genericBallFallback": 32},
        "inference": {
            "imageSize": 1280,
            "confidence": 0.035,
            "providerNmsIou": 0.7,
        },
        "personFilter": {"version": "pitch-person-v3", "localNmsIou": 0.48},
        "genericBallFallbackFilter": {"version": "generic-coco-ball-v2"},
    }


def _people_payload() -> list[dict]:
    return [
        {
            "x": 30.0,
            "y": 60.0,
            "width": 20.0,
            "height": 50.0,
            "confidence": 0.91,
            "feature": [0.0] * 11 + [1.0],
        }
    ]


def _ball_payload() -> list[dict]:
    return [{"x": 62.0, "y": 42.0, "confidence": 0.73}]


def test_detector_provenance_fingerprints_checkpoint_and_processing_policy(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "players.pt"
    checkpoint.write_bytes(b"detector-weights")

    detector_input = provenance.person_detection_input(str(checkpoint))

    assert detector_input["checkpoint"]["sha256"] == sha256(
        b"detector-weights"
    ).hexdigest()
    assert detector_input["checkpoint"]["contentAvailable"] is True
    assert detector_input["inference"]["providerNmsIou"] == 0.70
    assert detector_input["personFilter"] == {
        "version": "pitch-person-v3",
        "localNmsIou": 0.48,
        "minimumFootYRatio": 0.18,
        "shallowFootYRatio": 0.34,
        "shallowConfidence": 0.12,
        "shallowGrassRatio": 0.52,
        "appearanceFeatureSchema": "hsv-histogram-v1",
    }
    assert detector_input["genericBallFallbackFilter"] == {
        "version": "generic-coco-ball-v2",
        "minimumCenterYRatio": 0.30,
        "maximumBoxSizePixels": 24.0,
        "minimumGrassRatio": 0.24,
        "deduplicationRadiusPixels": 10.0,
    }


def test_contract_covers_exact_frame_model_nms_and_filter_policy() -> None:
    first = cache.build_person_detection_cache_contract(
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
    )
    reordered = cache.build_person_detection_cache_contract(
        frame_sha256="a" * 64,
        detector_input=dict(reversed(list(_detector_input().items()))),
    )
    changed_frame = cache.build_person_detection_cache_contract(
        frame_sha256="b" * 64,
        detector_input=_detector_input(),
    )
    changed_model_input = _detector_input("players-v2.pt")
    changed_model = cache.build_person_detection_cache_contract(
        frame_sha256="a" * 64,
        detector_input=changed_model_input,
    )
    changed_policy_input = _detector_input()
    changed_policy_input["personFilter"]["localNmsIou"] = 0.42
    changed_policy = cache.build_person_detection_cache_contract(
        frame_sha256="a" * 64,
        detector_input=changed_policy_input,
    )

    assert cache.person_detection_cache_key(first) == cache.person_detection_cache_key(
        reordered
    )
    assert len(
        {
            cache.person_detection_cache_key(first),
            cache.person_detection_cache_key(changed_frame),
            cache.person_detection_cache_key(changed_model),
            cache.person_detection_cache_key(changed_policy),
        }
    ) == 4


def test_atomic_round_trip_returns_detached_base_evidence(tmp_path: Path) -> None:
    stored = cache.store_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
        image_size=(100, 80),
        people=_people_payload(),
        generic_ball_candidates=_ball_payload(),
    )
    assert stored is not None
    assert stored.path.is_file()

    lookup = cache.lookup_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
    )
    assert lookup.status == "hit"
    assert lookup.entry is not None
    first_people, first_balls, image_size = lookup.entry.as_pipeline_data()
    first_people[0]["x"] = -1
    first_people[0]["feature"][0] = 99
    first_balls[0]["x"] = -1
    second_people, second_balls, _ = lookup.entry.as_pipeline_data()
    assert image_size == (100, 80)
    assert second_people == _people_payload()
    assert second_balls == _ball_payload()


def test_corrupt_or_tampered_artifact_is_a_safe_miss(tmp_path: Path) -> None:
    stored = cache.store_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
        image_size=(100, 80),
        people=_people_payload(),
        generic_ball_candidates=_ball_payload(),
    )
    assert stored is not None
    envelope = json.loads(stored.path.read_text())
    envelope["payload"]["people"][0]["x"] = 71.0
    stored.path.write_text(json.dumps(envelope))

    tampered = cache.lookup_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
    )
    assert tampered.entry is None
    assert tampered.status == "corrupt"

    stored.path.write_text("not json")
    corrupt = cache.lookup_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
    )
    assert corrupt.entry is None
    assert corrupt.status == "corrupt"


def test_partial_or_fallback_provider_output_is_not_published(tmp_path: Path) -> None:
    for provider_status in ("partial", "fallback", "failed"):
        assert (
            cache.store_person_detection_cache(
                tmp_path,
                frame_sha256="a" * 64,
                detector_input=_detector_input(),
                image_size=(100, 80),
                people=_people_payload(),
                generic_ball_candidates=_ball_payload(),
                provider_status=provider_status,
            )
            is None
        )
    assert not list(tmp_path.rglob("*.json"))


def test_failed_atomic_replace_preserves_previous_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stored = cache.store_person_detection_cache(
        tmp_path,
        frame_sha256="a" * 64,
        detector_input=_detector_input(),
        image_size=(100, 80),
        people=_people_payload(),
        generic_ball_candidates=_ball_payload(),
    )
    assert stored is not None
    original = stored.path.read_bytes()

    def fail_replace(_: Path, __: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    with pytest.raises(cache.PersonDetectionCacheError, match="simulated replace failure"):
        cache.store_person_detection_cache(
            tmp_path,
            frame_sha256="a" * 64,
            detector_input=_detector_input(),
            image_size=(100, 80),
            people=_people_payload(),
            generic_ball_candidates=_ball_payload(),
        )
    assert stored.path.read_bytes() == original
    assert not list(stored.path.parent.glob("*.tmp"))


class _Tensor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor(
            np.asarray(
                [
                    [20.0, 10.0, 40.0, 60.0],
                    [60.0, 40.0, 64.0, 44.0],
                ],
                dtype=np.float32,
            )
        )
        self.cls = _Tensor(np.asarray([0, 32], dtype=np.float32))
        self.conf = _Tensor(np.asarray([0.91, 0.73], dtype=np.float32))


class _Result:
    def __init__(self, image: np.ndarray) -> None:
        self.orig_img = image
        self.boxes = _Boxes()


def _write_frame(path: Path, green: int) -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :, 1] = green
    assert cv2.imwrite(str(path), image)


def _predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], object]:
    calls: list[str] = []

    class Provider:
        def predict(self, path: Path):
            calls.append(path.name)
            image = cv2.imread(str(path))
            assert image is not None
            return person_inference.prediction_from_ultralytics_result(
                _Result(image)
            )

    return calls, Provider()


def _diagnostics(detector_input: dict, frame_count: int = 1) -> dict:
    return base_cache.base_detection_cache_diagnostics(
        frame_count,
        detector_input,
    )


def test_pipeline_helper_second_identical_run_has_no_provider_call_and_is_detached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame_00001.jpg"
    _write_frame(frame, 150)
    calls, provider = _predictor(monkeypatch)
    detector_input = _detector_input()
    first_diagnostics = _diagnostics(detector_input)
    _, first_people, first_balls = base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, first_diagnostics
    )
    identity_evidence.capture_detection_observations(first_people, 1)
    first_observation_id = first_people[0].observation_id
    first_people[0].x = -500
    first_people[0].feature[0] = 99
    first_balls[0]["x"] = -500

    second_diagnostics = _diagnostics(detector_input)
    _, second_people, second_balls = base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, second_diagnostics
    )
    identity_evidence.capture_detection_observations(second_people, 1)

    assert calls == [frame.name]
    assert first_diagnostics["providerCalls"] == 1
    assert first_diagnostics["writes"] == 1
    assert second_diagnostics["providerCalls"] == 0
    assert second_diagnostics["hits"] == 1
    assert second_people[0].x == pytest.approx(30.0)
    assert second_people[0].feature[0] != 99
    assert second_people[0].observation_id == first_observation_id
    assert second_balls[0]["x"] == pytest.approx(62.0)


def test_one_changed_frame_content_causes_exactly_one_miss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [tmp_path / "frame_00001.jpg", tmp_path / "frame_00002.jpg"]
    _write_frame(frames[0], 140)
    _write_frame(frames[1], 160)
    calls, provider = _predictor(monkeypatch)
    detector_input = _detector_input()
    for path in frames:
        base_cache.cached_base_frame_detections(
            provider, path, tmp_path, detector_input, _diagnostics(detector_input)
        )
    assert len(calls) == 2

    _write_frame(frames[1], 180)
    diagnostics = _diagnostics(detector_input, frame_count=2)
    for path in frames:
        base_cache.cached_base_frame_detections(
            provider, path, tmp_path, detector_input, diagnostics
        )

    assert len(calls) == 3
    assert diagnostics["hits"] == 1
    assert diagnostics["misses"] == 1
    assert diagnostics["providerCalls"] == 1


def test_model_or_filter_policy_change_invalidates_pipeline_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame_00001.jpg"
    _write_frame(frame, 150)
    calls, provider = _predictor(monkeypatch)
    first = _detector_input()
    base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, first, _diagnostics(first)
    )

    changed_model = _detector_input("players-v2.pt")
    base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, changed_model, _diagnostics(changed_model)
    )
    changed_policy = _detector_input()
    changed_policy["personFilter"]["localNmsIou"] = 0.40
    base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, changed_policy, _diagnostics(changed_policy)
    )

    assert calls == [frame.name, frame.name, frame.name]


def test_corrupt_pipeline_artifact_recomputes_and_replaces_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame_00001.jpg"
    _write_frame(frame, 150)
    calls, provider = _predictor(monkeypatch)
    detector_input = _detector_input()
    base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, _diagnostics(detector_input)
    )
    frame_digest = cache.frame_content_sha256(frame)
    lookup = cache.lookup_person_detection_cache(
        tmp_path,
        frame_sha256=frame_digest,
        detector_input=detector_input,
    )
    assert lookup.entry is not None
    lookup.entry.path.write_text("broken")

    diagnostics = _diagnostics(detector_input)
    base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, diagnostics
    )

    assert calls == [frame.name, frame.name]
    assert diagnostics["misses"] == 1
    assert diagnostics["corruptArtifacts"] == 1
    assert diagnostics["errors"] == 1
    assert diagnostics["writes"] == 1
    assert (
        cache.lookup_person_detection_cache(
            tmp_path,
            frame_sha256=frame_digest,
            detector_input=detector_input,
        ).status
        == "hit"
    )


def test_manual_bbox_is_applied_after_cache_and_does_not_invalidate_base_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame_00001.jpg"
    _write_frame(frame, 150)
    calls, provider = _predictor(monkeypatch)
    detector_input = _detector_input()
    image, base_people, _ = base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, _diagnostics(detector_input)
    )
    annotated = person_annotations.apply_person_annotations(
        image,
        base_people,
        [
            {
                "id": "manual-person",
                "action": "confirm",
                "scope": "observation",
                "kind": "home-player",
                "bbox": {"x": 70.0, "y": 10.0, "width": 12.0, "height": 45.0},
            }
        ],
    )
    _, rebuilt_base_people, _ = base_cache.cached_base_frame_detections(
        provider, frame, tmp_path, detector_input, _diagnostics(detector_input)
    )

    assert calls == [frame.name]
    assert len(annotated) == 2
    assert any(item.annotation_id == "manual-person" for item in annotated)
    assert len(rebuilt_base_people) == 1
    assert rebuilt_base_people[0].annotation_id is None
