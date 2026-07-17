from __future__ import annotations

import numpy as np
import pytest

from app.pitch_calibration import PitchCalibration
from app.reconstruction import (
    _apply_dense_ball_projection,
    _dense_ball_projection_context,
)


PITCH = {"length": 105.0, "width": 68.0}
FRAME_SIZE = (100, 100)


def _calibration(matrix: np.ndarray, frame_index: int) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.asarray(matrix, dtype=np.float64),
        confidence=0.9,
        supported_lines=8,
        mean_line_score=0.8,
        rectangle="field-keypoints-right",
        method="test-calibration",
        frame_index=frame_index,
        reprojection_error=1.0,
    )


def _pitch_matrix(x_offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [0.5, 0.0, -25.0 + x_offset],
            [0.0, 0.5, -25.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _translation(x_offset: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, x_offset],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _evidence(*, upper_status: str = "accepted", motion_status: str = "estimated") -> list[dict]:
    return [
        {
            "sourceFrameIndex": 100,
            "status": "accepted",
            "projectionSource": "direct",
            "cameraMotion": {"status": "first-frame", "confidence": 1.0},
        },
        {
            "sourceFrameIndex": 101,
            "status": upper_status,
            "projectionSource": "temporal-forward",
            "cameraMotion": {"status": motion_status, "confidence": 0.9},
        },
    ]


def _context(
    scene_time: float,
    *,
    calibrations: dict[int, PitchCalibration] | None = None,
    evidence: list[dict] | None = None,
    transforms: dict[int, np.ndarray] | None = None,
    sampled_times: list[float] | None = None,
):
    return _dense_ball_projection_context(
        scene_time,
        sampled_times or [0.0, 0.1],
        {0: FRAME_SIZE, 1: FRAME_SIZE},
        calibrations
        or {
            0: _calibration(_pitch_matrix(), 100),
            1: _calibration(_pitch_matrix(10.0), 101),
        },
        {0: 1000, 1: 1001},
        {0: 1.0, 1: 3.0},
        evidence or _evidence(),
        transforms or {100: np.eye(3), 101: _translation(10.0)},
    )


def test_dense_projection_interpolates_qa_bracket_and_preserves_provenance():
    context = _context(0.04)

    assert context.provenance["method"] == "bounded-bracketing-homography-interpolation"
    assert context.provenance["sampleIndices"] == [0, 1]
    assert context.provenance["sourceFrameIndices"] == [100, 101]
    assert context.provenance["alpha"] == pytest.approx(0.4)
    assert context.provenance["fallback"] is False
    assert context.projection_source == "dense-bracket-interpolated"
    assert context.position_uncertainty_metres is not None
    assert context.position_uncertainty_metres > 1.8
    assert context.calibration is not None
    assert context.calibration.image_to_pitch[0, 2] == pytest.approx(-21.0)
    assert context.camera_transform[0, 2] == pytest.approx(4.0)

    balls = [
        {
            "x": 100.0,
            "y": 160.0,
            "imageWidth": 200,
            "imageHeight": 200,
            "provenance": {"backend": "dedicated-ultralytics"},
        }
    ]
    metric_count = _apply_dense_ball_projection(balls, context, PITCH, 7)

    assert metric_count == 1
    assert balls[0]["x"] == pytest.approx(50.0)
    assert balls[0]["y"] == pytest.approx(80.0)
    assert balls[0]["pitchX"] == pytest.approx(4.0)
    assert balls[0]["pitchZ"] == pytest.approx(15.0)
    assert balls[0]["stabilizedX"] == pytest.approx(54.0)
    assert balls[0]["calibrationSampleIndices"] == [0, 1]
    assert balls[0]["calibrationInterpolationAlpha"] == pytest.approx(0.4)
    assert balls[0]["provenance"]["backend"] == "dedicated-ultralytics"
    assert (
        balls[0]["provenance"]["projection"]["method"]
        == "bounded-bracketing-homography-interpolation"
    )


def test_dense_projection_falls_back_when_bracket_is_not_qa_accepted():
    context = _context(
        0.04,
        calibrations={0: _calibration(_pitch_matrix(), 100)},
        evidence=_evidence(upper_status="rejected"),
    )

    assert context.provenance["method"] == "nearest-qa-sample-fallback"
    assert context.provenance["fallback"] is True
    assert context.provenance["fallbackReason"] == "bracket-calibration-not-qa-accepted"
    assert context.nearest_sample_index == 0
    assert context.calibration is not None
    assert np.allclose(context.calibration.image_to_pitch, _pitch_matrix())


def test_dense_projection_rejects_degenerate_interpolated_matrix():
    lower = np.diag([0.5, 0.5, 1.0])
    upper = np.diag([-0.5, -0.5, 1.0])
    context = _context(
        0.05,
        calibrations={
            0: _calibration(lower, 100),
            1: _calibration(upper, 101),
        },
    )

    assert context.provenance["method"] == "nearest-qa-sample-fallback"
    assert context.provenance["fallback"] is True
    assert context.provenance["fallbackReason"].startswith(
        "calibration-interpolation-interpolated-matrix-near-singular"
    )
    assert context.nearest_sample_index == 0
    assert context.calibration is not None
    assert np.allclose(context.calibration.image_to_pitch, lower)


def test_dense_projection_does_not_interpolate_across_unreliable_camera_edge():
    context = _context(0.04, evidence=_evidence(motion_status="cut"))

    assert context.provenance["method"] == "nearest-qa-sample-fallback"
    assert (
        context.provenance["fallbackReason"]
        == "bracket-camera-motion-edge-not-reliable"
    )
    assert context.nearest_sample_index == 0


def test_dense_projection_enforces_bracket_time_bound():
    context = _context(0.1, sampled_times=[0.0, 0.3])

    assert context.provenance["method"] == "nearest-qa-sample-fallback"
    assert (
        context.provenance["fallbackReason"]
        == "sample-bracket-exceeds-interpolation-bound"
    )
    assert context.provenance["alpha"] == pytest.approx(1.0 / 3.0)

