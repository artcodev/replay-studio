from pathlib import Path
from types import SimpleNamespace

import numpy as np

import app.reconstruction_sampled_frame_detection as sampled_detection
from app.reconstruction_sampled_frame_contract import (
    SampledCalibrationAnalysis,
    SampledCalibrationInputs,
)


class _Progress:
    def update(self, *args, **kwargs) -> None:
        return None


def test_streaming_detection_passes_one_decoded_image_to_calibration(
    monkeypatch,
) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    raw_people = [object()]
    annotated_people = [object()]
    balls = [{"confidence": 0.4}]
    decode_calls: list[Path] = []
    captured_source_indices: list[int] = []
    accumulated: list[tuple[np.ndarray, list[object]]] = []

    def cached_detections(model, path, directory, detector_input, diagnostics):
        decode_calls.append(path)
        return image, raw_people, balls

    def apply_annotations(received_image, received_people, annotations):
        assert received_image is image
        assert received_people is raw_people
        assert annotations == []
        return annotated_people

    class Accumulator:
        def __init__(self, scene, inputs) -> None:
            assert inputs is calibration_inputs

        def add_frame(self, **values) -> None:
            accumulated.append((values["image"], values["people"]))
            assert values["source_index"] == 7

        def result(self) -> SampledCalibrationAnalysis:
            return SampledCalibrationAnalysis(
                frame_size=(32, 24),
                frame_sizes={0: (32, 24)},
                camera_motion_edges={},
                camera_transforms={},
                accepted_frame_calibrations={},
                accepted_automatic_direct_by_sample={},
                accepted_manual_direct_by_sample={},
                frame_evidence=[],
                rejected_frame_count=0,
            )

    monkeypatch.setattr(
        sampled_detection.person_base_detection_cache,
        "cached_base_frame_detections",
        cached_detections,
    )
    monkeypatch.setattr(sampled_detection, "apply_person_annotations", apply_annotations)
    monkeypatch.setattr(sampled_detection, "frame_annotations", lambda *_: [])
    monkeypatch.setattr(
        sampled_detection,
        "capture_detection_observations",
        lambda people, source_index: captured_source_indices.append(source_index),
    )
    monkeypatch.setattr(sampled_detection, "SampledCalibrationAccumulator", Accumulator)

    def unreadable_frame(_path):
        raise OSError("synthetic frame path has no bytes")

    monkeypatch.setattr(sampled_detection, "frame_content_sha256", unreadable_frame)

    calibration_inputs = SampledCalibrationInputs(
        manual_reference={},
        frame_calibrations={},
        calibration_warnings=[],
        manual_stabilized_by_sample={},
        manual_override_by_sample={},
    )
    runtime = SimpleNamespace(
        model=object(),
        person_cache_directory=Path("/tmp/cache"),
        person_detector_input={"fingerprint": "detector-v1"},
        person_cache_diagnostics={"errors": [], "hits": 1},
    )
    frame = Path("/tmp/frame_000007.jpg")
    result = sampled_detection.analyze_sampled_frames(
        {"payload": {"videoAsset": {}}},
        [(frame, 1.25)],
        runtime,
        calibration_inputs,
        _Progress(),
    )

    assert decode_calls == [frame]
    assert len(accumulated) == 1
    assert accumulated[0][0] is image
    assert accumulated[0][1] is annotated_people
    assert captured_source_indices == [7]
    assert result.person_frames == [(annotated_people, 1.25)]
    assert result.generic_ball_frames == [(balls, 1.25)]
    assert result.person_counts == [1]
    assert result.ball_counts == [1]
    assert runtime.person_cache_diagnostics == {
        "errors": [],
        "hits": 1,
        "status": "ready",
        "hitRatio": 1.0,
        "baseBoundary": (
            "pre-annotation/pre-calibration/pre-tracking/pre-reid/pre-ocr"
        ),
        # An unreadable frame disables crop extraction transparently.
        "personCropStore": {"hits": 0, "stores": 0, "storeErrors": 0},
    }
