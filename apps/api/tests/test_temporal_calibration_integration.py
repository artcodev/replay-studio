from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_calibration_resolution import (
    resolve_temporal_frame_calibrations as _resolve_temporal_frame_calibrations,
)
from app.reconstruction_person_detection_contract import Detection
from app.camera_motion_contract import CameraMotionEstimate
from app.temporal_homography import normalize_homography


PITCH = {"length": 105, "width": 68}


def _frames(count: int) -> list[tuple[Path, float]]:
    return [
        (Path(f"sample_{100 + index}.jpg"), index * 0.2)
        for index in range(count)
    ]


def _calibration(
    matrix: np.ndarray,
    *,
    confidence: float = 0.92,
    frame_index: int = 100,
    side: str = "right",
) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=normalize_homography(matrix),
        confidence=confidence,
        supported_lines=12,
        mean_line_score=0.86,
        rectangle=f"field-keypoints-{side}",
        method="pnlcalib-points-lines",
        keypoint_count=12,
        inlier_count=11,
        reprojection_error=1.2,
        frame_index=frame_index,
        detected_keypoint_count=12,
        inlier_ratio=11 / 12,
        reprojection_p95=2.0,
    )


def _motion(matrix: np.ndarray | None = None) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        matrix=np.eye(3, dtype=np.float64) if matrix is None else matrix,
        status="estimated",
        confidence=0.96,
        tracked_count=80,
        inlier_count=72,
        inlier_ratio=0.9,
        residual_p50=0.4,
        residual_p95=0.8,
        forward_backward_p95=0.6,
        coverage_ratio=0.45,
    )


def _detection(x: float = 480.0, y: float = 270.0) -> Detection:
    return Detection(
        x=x,
        y=y,
        width=18.0,
        height=44.0,
        confidence=0.8,
        feature=np.zeros(12, dtype=np.float32),
    )


def _person_frames(count: int) -> list[tuple[list[Detection], float]]:
    # Fewer than four detections deliberately leaves person support unscored,
    # matching a sparse partial-field shot without bypassing temporal QA.
    return [([_detection()], index * 0.2) for index in range(count)]


def _evidence(count: int, direct_samples: set[int]) -> list[dict]:
    result = []
    for index in range(count):
        direct = index in direct_samples
        result.append(
            {
                "sourceFrameIndex": 100 + index,
                "sampleIndex": index,
                "sceneTime": round(index * 0.2, 3),
                "status": "accepted" if direct else "missing",
                "source": "pnlcalib-points-lines" if direct else "none",
                "projectionSource": "direct" if direct else "none",
                "backend": "pnlcalib-points-lines" if direct else None,
                "confidence": 0.92 if direct else None,
                "imageToPitch": None,
                "visiblePitchSide": "right" if direct else None,
                "rejectionReasons": [],
            }
        )
    return result


def _base_homography() -> np.ndarray:
    return np.asarray(
        [
            [0.1, 0.0, -48.0],
            [0.0, 0.1, -27.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_resolver_uses_future_anchor_and_publishes_temporal_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _frames(3)
    evidence = _evidence(3, {2})
    h0 = _base_homography()
    frame_1_to_0 = np.asarray(
        [[1.0, 0.0, -4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frame_2_to_1 = np.asarray(
        [[1.0, 0.0, -6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h2 = normalize_homography(h0 @ frame_1_to_0 @ frame_2_to_1)
    evidence[2]["imageToPitch"] = h2.tolist()
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.cv2.imread",
        lambda _path: np.zeros((540, 960, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.calibration_alignment_metrics",
        lambda _image, _calibration: None,
    )

    resolved, anchor_frames, uncertainties, recovered_count = (
        _resolve_temporal_frame_calibrations(
            frames,
            {index: (960, 540) for index in range(3)},
            {2: _calibration(h2, frame_index=102)},
            {1: _motion(frame_1_to_0), 2: _motion(frame_2_to_1)},
            evidence,
            _person_frames(3),
            PITCH,
        )
    )

    assert set(resolved) == {0, 1, 2}
    np.testing.assert_allclose(resolved[0].image_to_pitch, h0, atol=1e-9)
    assert recovered_count == 2
    assert anchor_frames[0] == 102
    assert uncertainties[0] > uncertainties[2]

    recovered = evidence[0]
    assert recovered["observationStatus"] == "missing"
    assert recovered["observation"]["status"] == "missing"
    assert recovered["status"] == "accepted"
    assert recovered["solutionStatus"] == "temporal-accepted"
    assert recovered["projectionSource"] == "temporal-backward"
    assert recovered["selectedHypothesisId"] == "temporal-backward-s2-to-s0"
    assert recovered["temporal"]["direction"] == "backward"
    assert recovered["temporal"]["anchorFrameIndices"] == [102]
    assert recovered["temporal"]["anchorSampleIndices"] == [2]
    assert recovered["temporal"]["anchorSceneTimes"] == [0.4]
    assert recovered["temporal"]["motionEdgeIndices"] == [1, 2]
    assert recovered["temporal"]["temporalDistanceSeconds"] == 0.4
    assert 0.0 < recovered["temporal"]["motionConfidence"] <= 1.0
    assert recovered["uncertainty"]["kind"] == "engineering-p95"
    assert recovered["uncertainty"]["p95Metres"] == pytest.approx(
        recovered["positionUncertaintyMetres"]
    )
    assert len(recovered["hypotheses"]) == 1
    assert recovered["hypotheses"][0]["selected"] is True
    assert recovered["hypotheses"][0]["origin"] == "temporal-backward"
    assert recovered["hypotheses"][0]["anchorFrameIndices"] == [102]
    assert recovered["hypotheses"][0]["uncertaintyP95Metres"] > 0


def test_resolver_keeps_conflicting_temporal_candidates_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _frames(3)
    evidence = _evidence(3, {0, 2})
    left = _base_homography()
    right = left.copy()
    right[0, 2] += 20.0
    evidence[0]["imageToPitch"] = left.tolist()
    evidence[2]["imageToPitch"] = right.tolist()
    # No target image validation should run for an ambiguity with no selection.
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.cv2.imread",
        lambda _path: pytest.fail("ambiguous target must not reach target validation"),
    )

    resolved, _, _, recovered_count = _resolve_temporal_frame_calibrations(
        frames,
        {index: (960, 540) for index in range(3)},
        {
            0: _calibration(left, frame_index=100, side="left"),
            2: _calibration(right, frame_index=102, side="right"),
        },
        {1: _motion(), 2: _motion()},
        evidence,
        _person_frames(3),
        PITCH,
    )

    assert set(resolved) == {0, 2}
    assert recovered_count == 0
    ambiguous = evidence[1]
    assert ambiguous["observationStatus"] == "missing"
    assert ambiguous["solutionStatus"] == "ambiguous"
    assert ambiguous["projectionSource"] == "none"
    assert ambiguous["selectedHypothesisId"] is None
    assert ambiguous["temporal"] is None
    assert ambiguous["uncertainty"] is None
    assert "conflicting-temporal-hypotheses" in ambiguous["rejectionReasons"]
    assert len(ambiguous["hypotheses"]) == 2
    assert all(item["selected"] is False for item in ambiguous["hypotheses"])
    assert {item["origin"] for item in ambiguous["hypotheses"]} == {
        "temporal-forward",
        "temporal-backward",
    }


def test_target_line_validation_can_veto_temporal_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _frames(2)
    evidence = _evidence(2, {1})
    calibration = _base_homography()
    evidence[1]["imageToPitch"] = calibration.tolist()
    bad_alignment = SimpleNamespace(
        residual_p95=23.0,
        f1=0.03,
        as_dict=lambda: {
            "precision": 0.05,
            "recall": 0.02,
            "f1": 0.03,
            "residualP50": 12.0,
            "residualP95": 23.0,
        },
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.cv2.imread",
        lambda _path: np.zeros((540, 960, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.calibration_alignment_metrics",
        lambda _image, _calibration: bad_alignment,
    )

    resolved, _, _, recovered_count = _resolve_temporal_frame_calibrations(
        frames,
        {index: (960, 540) for index in range(2)},
        {1: _calibration(calibration, frame_index=101)},
        {1: _motion()},
        evidence,
        _person_frames(2),
        PITCH,
    )

    assert set(resolved) == {1}
    assert recovered_count == 0
    rejected = evidence[0]
    assert rejected["observationStatus"] == "missing"
    assert rejected["solutionStatus"] == "temporal-rejected"
    assert rejected["projectionSource"] == "none"
    assert rejected["selectedHypothesisId"] is None
    assert "temporal-semantic-line-alignment-poor" in rejected["rejectionReasons"]
    hypothesis = rejected["hypotheses"][0]
    assert hypothesis["selected"] is False
    assert (
        "temporal-semantic-line-alignment-poor"
        in hypothesis["targetValidation"]["rejectionReasons"]
    )
    assert hypothesis["targetValidation"]["alignmentMetrics"]["residualP95"] == 23.0
