from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.reconstruction_dense_ball_phase import detect_dense_ball_phase
from app.reconstruction_temporal_calibration_phase import TemporalCalibrationResult


class RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[tuple[tuple, dict]] = []

    def update(self, *args, **kwargs) -> None:
        self.updates.append((args, kwargs))


def _temporal_result() -> TemporalCalibrationResult:
    return TemporalCalibrationResult(
        resolved_by_sample={},
        anchor_by_sample={},
        uncertainty_by_sample={},
        recovered_frame_count=0,
        metric_person_sample_count=0,
    )


def test_dense_ball_phase_uses_immutable_queued_rate_and_failure_policy(
    monkeypatch,
) -> None:
    progress = RecordingProgress()
    detector_input = {
        "analysisFrameRate": 17.0,
        "failurePolicy": "raise",
    }
    captured: dict = {}

    def detect_frames(
        _scene,
        _detector,
        _fallback_detector,
        _frames,
        _generic_fallback_frames,
        progress_callback,
        *,
        failure_policy,
        detector_input,
    ):
        captured["failurePolicy"] = failure_policy
        captured["detectorInput"] = detector_input
        progress_callback(1, 2, "queued-detector")
        return [], {"source": "test"}, [], []

    monkeypatch.setattr(
        "app.reconstruction_dense_ball_phase.detect_ball_frames",
        detect_frames,
    )

    result = detect_dense_ball_phase(
        {"duration": 2.0, "payload": {"pitch": {"length": 105, "width": 68}}},
        detector=SimpleNamespace(),
        fallback_detector=None,
        sampled_frames=[],
        generic_fallback_ball_frames=[],
        detector_input=detector_input,
        backend="dedicated-ultralytics",
        frame_sizes={},
        temporal_calibration=_temporal_result(),
        frame_evidence=[],
        camera_transforms={},
        progress=progress,
    )

    assert captured == {
        "failurePolicy": "raise",
        "detectorInput": detector_input,
    }
    assert progress.updates[0][1]["total"] == 34
    assert "17 FPS" in progress.updates[0][0][3]
    assert progress.updates[1][1]["completed"] == 1
    assert result.frame_metadata == {"source": "test"}


@pytest.mark.parametrize(
    ("detector_input", "message"),
    [
        (
            {"analysisFrameRate": 0.0, "failurePolicy": "raise"},
            "analysisFrameRate",
        ),
        (
            {"analysisFrameRate": 17.0, "failurePolicy": "ignore"},
            "failurePolicy",
        ),
    ],
)
def test_dense_ball_phase_rejects_invalid_queued_policy(
    detector_input,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_dense_ball_phase(
            {
                "duration": 2.0,
                "payload": {"pitch": {"length": 105, "width": 68}},
            },
            detector=SimpleNamespace(),
            fallback_detector=None,
            sampled_frames=[],
            generic_fallback_ball_frames=[],
            detector_input=detector_input,
            backend="dedicated-ultralytics",
            frame_sizes={},
            temporal_calibration=_temporal_result(),
            frame_evidence=[],
            camera_transforms={},
            progress=RecordingProgress(),
        )
