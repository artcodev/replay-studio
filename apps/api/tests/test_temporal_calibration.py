import numpy as np
import pytest

from app.pitch_calibration import PitchCalibration, pitch_side
from app.temporal_calibration import (
    CameraMotionEstimate,
    TemporalCalibrationFrame,
    normalize_homography,
    propagate_homography,
    solve_calibration_sequence,
)


def _frame(sample_index: int, scene_time: float | None = None) -> TemporalCalibrationFrame:
    return TemporalCalibrationFrame(
        sample_index=sample_index,
        source_frame_index=101 + sample_index,
        scene_time=sample_index * 0.2 if scene_time is None else scene_time,
        width=960,
        height=540,
    )


def _calibration(
    matrix: np.ndarray,
    *,
    confidence: float = 0.92,
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
        detected_keypoint_count=12,
        inlier_ratio=11 / 12,
        reprojection_error=1.2,
        reprojection_p95=2.0,
    )


def _motion(
    matrix: np.ndarray | None = None,
    *,
    status: str = "estimated",
    confidence: float = 0.96,
    residual_p95: float = 0.8,
) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        matrix=np.eye(3, dtype=np.float64) if matrix is None else matrix,
        status=status,
        confidence=confidence,
        tracked_count=80,
        inlier_count=72,
        inlier_ratio=0.9,
        residual_p50=residual_p95 * 0.5,
        residual_p95=residual_p95,
        forward_backward_p95=residual_p95 * 0.75,
        coverage_ratio=0.45,
        reason=None if status == "estimated" else f"motion-{status}",
    )


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = homogeneous @ normalize_homography(matrix).T
    return projected[:, :2] / projected[:, 2:3]


def _assert_same_mapping(left: np.ndarray, right: np.ndarray) -> None:
    points = np.asarray(
        [
            [120.0, 400.0],
            [360.0, 250.0],
            [520.0, 340.0],
            [840.0, 470.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(_project(left, points), _project(right, points), atol=1e-8)


def _motion_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_image_to_pitch = np.asarray(
        [
            [0.082, 0.009, -45.0],
            [-0.004, 0.091, -24.0],
            [0.00002, -0.00003, 1.0],
        ],
        dtype=np.float64,
    )
    # A_i maps image coordinates in frame i into frame i - 1.  The two
    # projective transforms deliberately do not commute, so a reversed product
    # cannot accidentally satisfy the tests.
    frame_1_to_0 = np.asarray(
        [
            [1.02, 0.012, -11.0],
            [0.004, 0.99, 2.0],
            [0.00001, -0.00002, 1.0],
        ],
        dtype=np.float64,
    )
    frame_2_to_1 = np.asarray(
        [
            [0.98, -0.015, -8.0],
            [0.002, 1.01, 1.0],
            [-0.00002, 0.00001, 1.0],
        ],
        dtype=np.float64,
    )
    return reference_image_to_pitch, frame_1_to_0, frame_2_to_1


def test_future_direct_anchor_recovers_earlier_frames_backward() -> None:
    h0, a1, a2 = _motion_fixture()
    h2 = normalize_homography(h0 @ a1 @ a2)
    frames = [_frame(index) for index in range(3)]
    result = solve_calibration_sequence(
        frames,
        {2: _calibration(h2)},
        {1: _motion(a1), 2: _motion(a2)},
    )

    assert result[2].projection_source == "direct"
    assert result[2].selected is not None
    assert result[0].projection_source == "temporal-backward"
    assert result[1].projection_source == "temporal-backward"
    assert result[0].selected is not None
    assert result[1].selected is not None
    _assert_same_mapping(result[0].selected.calibration.image_to_pitch, h0)
    _assert_same_mapping(result[1].selected.calibration.image_to_pitch, h0 @ a1)
    assert result[0].selected.anchor_source_frame_index == frames[2].source_frame_index
    assert result[0].selected.motion_edge_indices == (1, 2)

    # Temporal recovery makes all frames usable without pretending that the
    # inferred frames were direct calibration observations.
    sources = [result[index].projection_source for index in range(3)]
    assert sum(source == "direct" for source in sources) == 1
    assert sum(source != "none" for source in sources) == 3


def test_motion_matrix_order_and_bidirectional_roundtrip() -> None:
    h0, a1, a2 = _motion_fixture()
    edges = {1: _motion(a1), 2: _motion(a2)}
    expected_h2 = normalize_homography(h0 @ a1 @ a2)
    wrong_order = normalize_homography(h0 @ a2 @ a1)

    propagated_h2 = propagate_homography(h0, edges, 2, 0)

    assert propagated_h2 is not None
    _assert_same_mapping(propagated_h2, expected_h2)
    with pytest.raises(AssertionError):
        _assert_same_mapping(propagated_h2, wrong_order)

    recovered_h0 = propagate_homography(propagated_h2, edges, 0, 2)

    assert recovered_h0 is not None
    _assert_same_mapping(recovered_h0, h0)


@pytest.mark.parametrize("barrier_status", ["cut", "unestimated", "unreliable"])
def test_cut_or_invalid_motion_edge_is_a_hard_propagation_barrier(
    barrier_status: str,
) -> None:
    h0, a1, a2 = _motion_fixture()
    h2 = normalize_homography(h0 @ a1 @ a2)
    edges = {
        1: _motion(a1),
        2: _motion(a2, status=barrier_status, confidence=0.0),
    }

    assert propagate_homography(h2, edges, 0, 2) is None
    result = solve_calibration_sequence(
        [_frame(index) for index in range(3)],
        {2: _calibration(h2)},
        edges,
    )

    assert result[0].selected is None
    assert result[1].selected is None
    assert result[0].projection_source == "none"
    assert "no-reliable-temporal-path" in result[0].rejection_reasons
    assert result[2].projection_source == "direct"


def test_temporal_max_gap_is_inclusive_and_blocks_more_distant_frames() -> None:
    h0, _, _ = _motion_fixture()
    frames = [_frame(0, 0.0), _frame(1, 0.6), _frame(2, 0.601)]
    result = solve_calibration_sequence(
        frames,
        {0: _calibration(h0)},
        {1: _motion(), 2: _motion()},
        max_gap_seconds=0.6,
        minimum_score=0.0,
        maximum_uncertainty_metres=20.0,
    )

    assert result[1].selected is not None
    assert result[1].selected.temporal_distance_seconds == pytest.approx(0.6)
    assert result[2].selected is None
    assert result[2].hypotheses == ()


def test_uncertainty_grows_with_temporal_distance_and_motion_path() -> None:
    h0, _, _ = _motion_fixture()
    frames = [_frame(index) for index in range(4)]
    result = solve_calibration_sequence(
        frames,
        {0: _calibration(h0)},
        {
            1: _motion(confidence=0.94, residual_p95=0.8),
            2: _motion(confidence=0.94, residual_p95=0.8),
            3: _motion(confidence=0.94, residual_p95=0.8),
        },
        minimum_score=0.0,
        maximum_uncertainty_metres=20.0,
    )

    selected = [result[index].selected for index in range(4)]
    assert all(item is not None for item in selected)
    uncertainties = [item.uncertainty_metres for item in selected if item is not None]
    scores = [item.score for item in selected if item is not None]
    assert uncertainties == sorted(uncertainties)
    assert len(set(uncertainties)) == len(uncertainties)
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_similarly_scored_conflicting_hypotheses_remain_ambiguous() -> None:
    base = np.asarray(
        [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    shifted = base.copy()
    shifted[0, 2] += 20.0
    frames = [_frame(index) for index in range(3)]
    result = solve_calibration_sequence(
        frames,
        {
            0: _calibration(base, side="left"),
            2: _calibration(shifted, side="right"),
        },
        {1: _motion(), 2: _motion()},
    )

    middle = result[1]
    assert middle.selected is None
    assert middle.projection_source == "none"
    assert middle.ambiguity_margin == pytest.approx(0.0)
    assert "conflicting-temporal-hypotheses" in middle.rejection_reasons
    assert len(middle.hypotheses) == 2
    assert {pitch_side(item.calibration.rectangle) for item in middle.hypotheses} == {
        "left",
        "right",
    }
    assert all(
        item.disagreement_metres is not None and item.disagreement_metres > 2.5
        for item in middle.hypotheses
    )


def test_third_candidate_can_expose_conflict_hidden_by_two_compatible_leaders() -> None:
    base = np.asarray(
        [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    conflicting = base.copy()
    conflicting[0, 2] += 22.0
    frames = [_frame(index) for index in range(4)]

    result = solve_calibration_sequence(
        frames,
        {
            0: _calibration(base, confidence=0.92, side="left"),
            1: _calibration(base, confidence=0.92, side="left"),
            3: _calibration(conflicting, confidence=0.88, side="right"),
        },
        {1: _motion(), 2: _motion(), 3: _motion()},
    )

    middle = result[2]
    assert len(middle.hypotheses) == 3
    assert middle.selected is None
    assert "conflicting-temporal-hypotheses" in middle.rejection_reasons


def test_consistent_bracketing_anchors_form_bidirectional_consensus() -> None:
    calibration_matrix = np.asarray(
        [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frames = [_frame(index) for index in range(3)]
    result = solve_calibration_sequence(
        frames,
        {
            0: _calibration(calibration_matrix),
            2: _calibration(calibration_matrix),
        },
        {1: _motion(), 2: _motion()},
    )

    middle = result[1]
    assert middle.selected is not None
    assert middle.projection_source == "temporal-bidirectional"
    assert middle.selected.calibration.method == "temporal-bidirectional"
    assert middle.rejection_reasons == ()
    assert middle.ambiguity_margin == pytest.approx(0.0)
    assert len(middle.hypotheses) == 2
    assert {item.direction for item in middle.hypotheses} == {"forward", "backward"}
    assert middle.selected.disagreement_metres == pytest.approx(0.0)
    assert middle.selected.uncertainty_metres < max(
        item.uncertainty_metres for item in middle.hypotheses[1:]
    )
    _assert_same_mapping(middle.selected.calibration.image_to_pitch, calibration_matrix)


def test_degenerate_compatible_consensus_fails_closed() -> None:
    singular = np.asarray(
        [[0.1, 0.0, -48.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frames = [_frame(index) for index in range(3)]

    result = solve_calibration_sequence(
        frames,
        {
            0: _calibration(singular),
            2: _calibration(singular),
        },
        {1: _motion(), 2: _motion()},
    )

    middle = result[1]
    assert middle.selected is None
    assert middle.projection_source == "none"
    assert "temporal-bidirectional-consensus-failed" in middle.rejection_reasons
    assert len(middle.hypotheses) == 2
