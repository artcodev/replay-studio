from __future__ import annotations

import numpy as np

from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_calibration_draft import calibration_draft
from app.reconstruction_frame_calibration_quality import direct_calibration_qa


def _manual_calibration() -> PitchCalibration:
    # Low line-derived confidence, exactly the value a line-poor frame produces.
    return PitchCalibration(
        image_to_pitch=np.array(
            [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.25,
        supported_lines=4,
        mean_line_score=0.0,
        rectangle="center-circle",
        method="manual-pitch-anchors",
    )


def _black_frame() -> np.ndarray:
    # No visible white markings, so there is nothing to score an overlay against.
    return np.zeros((540, 960, 3), dtype=np.uint8)


def test_manual_calibration_bypasses_line_derived_gates():
    image = _black_frame()
    pitch = {"length": 105.0, "width": 68.0}
    calibration = _manual_calibration()

    manual = direct_calibration_qa(image, calibration, people=[], pitch=pitch, manual=True)
    automatic = direct_calibration_qa(image, calibration, people=[], pitch=pitch, manual=False)

    # The operator's anchor is authoritative: neither the line-derived metric
    # confidence nor the absence of a line score may reject it. Acceptance is the
    # absence of rejection reasons.
    assert "confidence-below-metric-threshold" not in manual["rejectionReasons"]
    assert "semantic-line-alignment-unscored" not in manual["rejectionReasons"]
    assert manual["rejectionReasons"] == []
    # The very same calibration is rejected on the automatic path.
    assert "confidence-below-metric-threshold" in automatic["rejectionReasons"]
    assert "semantic-line-alignment-unscored" in automatic["rejectionReasons"]


def test_manual_draft_stays_usable_when_no_markings_are_scorable():
    draft = calibration_draft(
        {"id": "scene-1"},
        frame_index=0,
        frame_time=0.0,
        image=_black_frame(),
        calibration=_manual_calibration(),
        preset="center-circle",
        source="manual",
        anchors=[{"id": "a", "label": "a", "image": {"x": 1.0, "y": 2.0}, "pitch": {"x": 0.0, "z": 0.0}}],
    )
    # A line-poor manual overlay is a usable review candidate, not "poor" — so
    # apply no longer rejects it.
    assert draft["alignmentMetrics"] is None
    assert draft["quality"] == "review"


def test_automatic_keypoint_tail_over_one_metre_is_rejected(monkeypatch):
    calibration = PitchCalibration(
        image_to_pitch=np.array(
            [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.92,
        supported_lines=10,
        mean_line_score=1.0,
        rectangle="field-keypoints-right",
        method="pnlcalib-points-lines",
        keypoint_count=17,
        detected_keypoint_count=17,
        inlier_count=17,
        inlier_ratio=1.0,
        reprojection_error=1.0,
        reprojection_p95=6.0,
        ground_error_p50=0.32,
        ground_error_p95=1.0885,
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_calibration_quality.calibration_alignment_metrics",
        lambda *_: type(
            "Alignment",
            (),
            {
                "residual_p50": 0.955,
                "residual_p95": 6.018,
                "f1": 0.35,
                "as_dict": lambda self: {
                    "residualP50": self.residual_p50,
                    "residualP95": self.residual_p95,
                    "f1": self.f1,
                },
            },
        )(),
    )

    qa = direct_calibration_qa(
        _black_frame(),
        calibration,
        people=[],
        pitch={"length": 105.0, "width": 68.0},
    )

    assert "semantic-keypoint-ground-error-too-high" in qa["rejectionReasons"]
    gate = next(
        item
        for item in qa["qualityGates"]
        if item["id"] == "semantic-keypoint-ground-error-p95"
    )
    assert gate["status"] == "fail"
    assert gate["value"] == 1.0885
