import cv2
import numpy as np
import pytest
from types import SimpleNamespace

import app.pitch_calibration_quality as calibration_quality
from app.pitch_anchor_calibration import calibration_from_anchors
from app.pitch_calibration_contract import PitchCalibration
from app.pitch_calibration_quality import (
    calibration_alignment_error,
    calibration_alignment_metrics,
    semantic_line_evidence,
)
from app.pitch_geometry import projected_pitch_markings


def _synthetic_calibration() -> PitchCalibration:
    # Pitch X/Z -> image x/y maps the full 105x68m field inside 960x540.
    pitch_to_image = np.array(
        [[8.0, 0.0, 480.0], [0.0, 6.0, 270.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PitchCalibration(
        image_to_pitch=np.linalg.inv(pitch_to_image),
        confidence=0.9,
        supported_lines=10,
        mean_line_score=0.9,
        rectangle="field-keypoints",
    )


def _synthetic_pitch_image(calibration: PitchCalibration) -> np.ndarray:
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    image[:, :] = (48, 128, 48)
    for marking in projected_pitch_markings(calibration, 960, 540):
        points = np.array(
            [[round(point["x"]), round(point["y"])] for point in marking["points"]],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(image, [points], False, (245, 245, 245), 3, cv2.LINE_AA)
    return image


def test_bidirectional_alignment_rewards_complete_matching_overlay():
    calibration = _synthetic_calibration()
    image = _synthetic_pitch_image(calibration)

    metrics = calibration_alignment_metrics(image, calibration)

    assert metrics is not None
    assert metrics.precision > 0.9
    assert metrics.recall > 0.7
    assert metrics.f1 > 0.75
    assert metrics.residual_p50 <= 1.0
    assert calibration_alignment_error(image, calibration) == round(metrics.residual_p50, 2)


def test_bidirectional_alignment_exposes_shifted_camera_fit():
    calibration = _synthetic_calibration()
    image = _synthetic_pitch_image(calibration)
    shifted_matrix = calibration.image_to_pitch.copy()
    shifted_matrix[0, 2] += 10.0
    shifted = PitchCalibration(
        image_to_pitch=shifted_matrix,
        confidence=0.9,
        supported_lines=10,
        mean_line_score=0.9,
        rectangle="field-keypoints",
    )

    good = calibration_alignment_metrics(image, calibration)
    bad = calibration_alignment_metrics(image, shifted)

    assert good is not None and bad is not None
    assert bad.f1 < good.f1 * 0.65
    assert bad.residual_p95 > good.residual_p95


def test_alignment_residuals_are_normalized_to_the_960x540_reference_grid(
    monkeypatch,
):
    calibration = _synthetic_calibration()

    def residuals(mask, _calibration):
        scale = mask.shape[1] / 960
        values = np.asarray([2.0, 3.0, 4.0]) * scale
        return SimpleNamespace(
            model_to_observed=values,
            observed_to_model=values,
            model_sample_count=3,
            observed_sample_count=3,
        )

    monkeypatch.setattr(
        calibration_quality,
        "alignment_residuals_from_mask",
        residuals,
    )
    reference = calibration_quality.calibration_alignment_metrics_from_mask(
        np.zeros((540, 960), dtype=np.uint8), calibration
    )
    full_hd = calibration_quality.calibration_alignment_metrics_from_mask(
        np.zeros((1080, 1920), dtype=np.uint8), calibration
    )

    assert reference is not None and full_hd is not None
    assert full_hd.residual_p50 == pytest.approx(reference.residual_p50)
    assert full_hd.residual_p95 == pytest.approx(reference.residual_p95)
    assert full_hd.f1 == pytest.approx(reference.f1)
    assert full_hd.as_dict()["residualUnit"] == "reference-pixel"
    assert full_hd.as_dict()["referenceWidth"] == 960


def test_semantic_line_evidence_reports_per_class_residual_and_skips_goal_frame():
    pitch_to_image = np.array(
        [[8.0, 0.0, 480.0], [0.0, 6.0, 270.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    calibration = PitchCalibration(
        image_to_pitch=np.linalg.inv(pitch_to_image),
        confidence=0.9,
        supported_lines=2,
        mean_line_score=0.9,
        rectangle="field-keypoints-right",
        raw_lines=(
            {
                "id": 17,
                "name": "Side line top",
                "start": {"x": 60.0, "y": 68.0},
                "end": {"x": 900.0, "y": 64.0},
                "confidence": 0.94,
                "groundPlane": True,
            },
            {
                "id": 10,
                "name": "Goal right crossbar",
                "start": {"x": 800.0, "y": 120.0},
                "end": {"x": 900.0, "y": 130.0},
                "confidence": 0.8,
                "groundPlane": False,
            },
        ),
    )

    lines = semantic_line_evidence(calibration)

    assert lines[0]["residualStatus"] == "scored"
    assert lines[0]["residualP50"] == pytest.approx(2.0)
    assert lines[0]["residualP95"] == pytest.approx(2.0)
    assert lines[1]["residualStatus"] == "not-scored-3d"
    assert lines[1]["residualP50"] is None
    assert lines[1]["residualP95"] is None


def test_manual_anchors_must_cover_a_stable_pitch_area():
    anchors = [
        {
            "image": {"x": x, "y": y},
            "pitch": {"x": pitch_x, "z": 0.0},
        }
        for (x, y), pitch_x in zip(
            [(100, 100), (500, 100), (100, 400), (500, 400)],
            [-20.0, -10.0, 10.0, 20.0],
        )
    ]

    with pytest.raises(ValueError, match="stable area"):
        calibration_from_anchors(anchors, "center-circle")
