import numpy as np

from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_calibration_resolution import (
    merge_direct_calibration_anchors as _merge_direct_calibration_anchors,
)
from app.camera_motion_contract import CameraMotionEstimate
from app.temporal_calibration_contract import TemporalCalibrationFrame
from app.temporal_calibration_solver import solve_calibration_sequence
from app.temporal_homography import normalize_homography


def _frame(sample_index: int) -> TemporalCalibrationFrame:
    return TemporalCalibrationFrame(
        sample_index=sample_index,
        source_frame_index=501 + sample_index,
        scene_time=sample_index * 0.2,
        width=960,
        height=540,
    )


def _calibration(
    matrix: np.ndarray,
    *,
    method: str,
    confidence: float = 0.94,
    side: str = "right",
) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=normalize_homography(matrix),
        confidence=confidence,
        supported_lines=10,
        mean_line_score=0.88,
        rectangle=f"field-keypoints-{side}",
        method=method,
        keypoint_count=12,
        detected_keypoint_count=12,
        inlier_count=11,
        inlier_ratio=11 / 12,
        reprojection_error=1.0,
        reprojection_p95=1.8,
    )


def _motion(
    matrix: np.ndarray,
    *,
    status: str = "estimated",
    confidence: float = 0.96,
) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        matrix=normalize_homography(matrix),
        status=status,
        confidence=confidence if status == "estimated" else 0.0,
        tracked_count=100,
        inlier_count=92,
        inlier_ratio=0.92,
        residual_p50=0.35,
        residual_p95=0.8,
        forward_backward_p95=0.55,
        coverage_ratio=0.48,
        reason=None if status == "estimated" else f"motion-{status}",
    )


def _fixture(
    frame_count: int = 7,
) -> tuple[list[TemporalCalibrationFrame], dict[int, CameraMotionEstimate], list[np.ndarray]]:
    frames = [_frame(index) for index in range(frame_count)]
    image_to_pitch_0 = np.asarray(
        [
            [0.084, 0.008, -44.0],
            [-0.003, 0.092, -25.0],
            [0.00002, -0.00003, 1.0],
        ],
        dtype=np.float64,
    )
    edge_matrices = {
        index: np.asarray(
            [
                [1.0 + index * 0.001, 0.002 * index, -3.0 - index],
                [0.001, 1.0 - index * 0.001, 0.7 * index],
                [index * 0.000002, -index * 0.000001, 1.0],
            ],
            dtype=np.float64,
        )
        for index in range(1, frame_count)
    }
    motion_edges = {
        index: _motion(matrix) for index, matrix in edge_matrices.items()
    }
    expected = [normalize_homography(image_to_pitch_0)]
    for index in range(1, frame_count):
        expected.append(
            normalize_homography(expected[-1] @ edge_matrices[index])
        )
    return frames, motion_edges, expected


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = homogeneous @ normalize_homography(matrix).T
    return projected[:, :2] / projected[:, 2:3]


def _assert_same_mapping(left: np.ndarray, right: np.ndarray) -> None:
    image_points = np.asarray(
        [
            [100.0, 180.0],
            [260.0, 430.0],
            [570.0, 310.0],
            [870.0, 490.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        _project(left, image_points),
        _project(right, image_points),
        atol=1e-7,
    )


def test_manual_and_auto_anchors_interpolate_a_partial_view_bidirectionally() -> None:
    frames, motion_edges, expected = _fixture(frame_count=5)
    direct = {
        1: _calibration(
            expected[1],
            method="manual-pitch-anchors",
            confidence=0.95,
        ),
        3: _calibration(
            expected[3],
            method="pnlcalib-points-lines",
            confidence=0.94,
        ),
    }

    result = solve_calibration_sequence(frames, direct, motion_edges)

    assert result[1].projection_source == "direct"
    assert result[1].selected is not None
    assert result[1].selected.calibration.method == "manual-pitch-anchors"
    assert result[3].projection_source == "direct"
    assert result[3].selected is not None
    assert result[3].selected.calibration.method == "pnlcalib-points-lines"

    middle = result[2]
    assert middle.selected is not None
    assert middle.projection_source == "temporal-bidirectional"
    assert middle.rejection_reasons == ()
    assert {item.direction for item in middle.hypotheses} == {
        "forward",
        "backward",
    }
    assert {item.anchor_sample_index for item in middle.hypotheses} == {1, 3}
    assert all(
        item.disagreement_metres is not None
        and item.disagreement_metres < 1e-6
        for item in middle.hypotheses
    )
    _assert_same_mapping(middle.selected.calibration.image_to_pitch, expected[2])

    # The frames outside the two anchors remain one-sided, with the nearest
    # anchor providing the exact world mapping in the appropriate direction.
    assert result[0].projection_source == "temporal-backward"
    assert result[0].selected is not None
    assert result[0].selected.anchor_sample_index == 1
    _assert_same_mapping(result[0].selected.calibration.image_to_pitch, expected[0])
    assert result[4].projection_source == "temporal-forward"
    assert result[4].selected is not None
    assert result[4].selected.anchor_sample_index == 3
    _assert_same_mapping(result[4].selected.calibration.image_to_pitch, expected[4])


def test_multiple_anchors_on_both_sides_keep_alternatives_and_choose_nearest_consensus() -> None:
    frames, motion_edges, expected = _fixture(frame_count=7)
    direct = {
        0: _calibration(
            expected[0],
            method="manual-pitch-anchors",
            confidence=0.82,
        ),
        2: _calibration(
            expected[2],
            method="pnlcalib-points-lines",
            confidence=0.95,
        ),
        4: _calibration(
            expected[4],
            method="manual-pitch-anchors",
            confidence=0.95,
        ),
        6: _calibration(
            expected[6],
            method="roboflow-field-keypoints",
            confidence=0.82,
        ),
    }

    result = solve_calibration_sequence(
        frames,
        direct,
        motion_edges,
        max_anchors_per_direction=2,
    )

    middle = result[3]
    assert middle.selected is not None
    assert middle.projection_source == "temporal-bidirectional"
    assert middle.rejection_reasons == ()
    assert len(middle.hypotheses) == 4
    assert {item.anchor_sample_index for item in middle.hypotheses} == {0, 2, 4, 6}
    assert middle.selected.anchor_sample_index in {2, 4}
    assert {item.direction for item in middle.hypotheses} == {
        "forward",
        "backward",
    }
    assert all(
        item.disagreement_metres is not None
        and item.disagreement_metres < 1e-6
        for item in middle.hypotheses
    )
    _assert_same_mapping(middle.selected.calibration.image_to_pitch, expected[3])


def test_bidirectional_consensus_stays_continuous_when_local_scores_cross() -> None:
    frames = [_frame(index) for index in range(6)]
    earlier_matrix = np.asarray(
        [
            [0.1, 0.0, -48.0],
            [0.0, 0.1, -27.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    later_matrix = earlier_matrix.copy()
    later_matrix[0, 2] += 2.0
    direct = {
        0: _calibration(
            earlier_matrix,
            method="manual-pitch-anchors",
            confidence=0.94,
        ),
        5: _calibration(
            later_matrix,
            method="pnlcalib-points-lines",
            confidence=0.94,
        ),
    }
    motion_edges = {index: _motion(np.eye(3)) for index in range(1, 6)}

    result = solve_calibration_sequence(frames, direct, motion_edges)

    # The local winner changes between adjacent frames, which previously made
    # the selected matrix jump from one anchor propagation to the other.
    assert result[2].selected is not None
    assert result[2].selected.direction == "forward"
    assert result[3].selected is not None
    assert result[3].selected.direction == "backward"
    assert all(
        result[index].projection_source == "temporal-bidirectional"
        for index in range(1, 5)
    )

    probe = np.asarray([[480.0, 360.0]], dtype=np.float64)
    earlier_x = float(_project(earlier_matrix, probe)[0, 0])
    projected_x = [
        float(_project(result[index].selected.calibration.image_to_pitch, probe)[0, 0])
        for index in range(6)
        if result[index].selected is not None
    ]
    expected_x = [
        earlier_x + 2.0 * (progress * progress * (3.0 - 2.0 * progress))
        for progress in np.linspace(0.0, 1.0, 6)
    ]
    np.testing.assert_allclose(projected_x, expected_x, atol=1e-7)
    assert max(np.diff(projected_x)) < 0.7

    # Direct observations are immutable endpoints; fusion is used only on the
    # missing frames between them.
    assert result[0].selected is not None
    assert result[0].selected.calibration is direct[0]
    assert result[5].selected is not None
    assert result[5].selected.calibration is direct[5]
    _assert_same_mapping(result[0].selected.calibration.image_to_pitch, earlier_matrix)
    _assert_same_mapping(result[5].selected.calibration.image_to_pitch, later_matrix)


def test_manual_auto_conflict_stays_ambiguous_instead_of_interpolating() -> None:
    frames, motion_edges, expected = _fixture(frame_count=3)
    conflicting_auto = expected[2].copy()
    conflicting_auto[0, 2] += 18.0
    direct = {
        0: _calibration(
            expected[0],
            method="manual-pitch-anchors",
            confidence=0.94,
            side="left",
        ),
        2: _calibration(
            conflicting_auto,
            method="pnlcalib-points-lines",
            confidence=0.94,
            side="right",
        ),
    }

    result = solve_calibration_sequence(frames, direct, motion_edges)

    middle = result[1]
    assert middle.selected is None
    assert middle.projection_source == "none"
    assert "conflicting-temporal-hypotheses" in middle.rejection_reasons
    assert {item.anchor_sample_index for item in middle.hypotheses} == {0, 2}
    assert all(
        item.disagreement_metres is not None
        and item.disagreement_metres > 2.5
        for item in middle.hypotheses
    )


def test_cut_blocks_manual_anchor_but_auto_anchor_on_target_side_can_recover() -> None:
    frames, motion_edges, expected = _fixture(frame_count=5)
    motion_edges[2] = _motion(
        motion_edges[2].matrix,
        status="cut",
    )
    direct = {
        0: _calibration(
            expected[0],
            method="manual-pitch-anchors",
            confidence=0.95,
        ),
        4: _calibration(
            expected[4],
            method="pnlcalib-points-lines",
            confidence=0.94,
        ),
    }

    result = solve_calibration_sequence(frames, direct, motion_edges)

    # Frame 1 is on the manual side of the barrier. Frame 2 is on the auto side;
    # neither solution is allowed to combine evidence across the cut.
    assert result[1].selected is not None
    assert result[1].projection_source == "temporal-forward"
    assert result[1].selected.anchor_sample_index == 0
    assert {item.anchor_sample_index for item in result[1].hypotheses} == {0}
    assert result[2].selected is not None
    assert result[2].projection_source == "temporal-backward"
    assert result[2].selected.anchor_sample_index == 4
    assert {item.anchor_sample_index for item in result[2].hypotheses} == {4}


def test_rejected_manual_anchor_is_not_passed_to_solver_and_auto_still_recovers() -> None:
    frames, motion_edges, expected = _fixture(frame_count=3)
    # The caller's acceptance gate deliberately omits a rejected manual fit.
    # The temporal solver must therefore see only the accepted automatic anchor,
    # rather than turning a rejected manual candidate into metric evidence.
    accepted_direct = {
        2: _calibration(
            expected[2],
            method="pnlcalib-points-lines",
            confidence=0.94,
        )
    }

    result = solve_calibration_sequence(frames, accepted_direct, motion_edges)

    assert result[0].selected is not None
    assert result[0].projection_source == "temporal-backward"
    assert result[0].selected.anchor_sample_index == 2
    assert result[0].rejection_reasons == ()
    _assert_same_mapping(result[0].selected.calibration.image_to_pitch, expected[0])


def test_same_sample_manual_anchor_wins_without_mutating_automatic_candidates() -> None:
    _, _, expected = _fixture(frame_count=4)
    automatic_at_one = _calibration(
        expected[1],
        method="pnlcalib-points-lines",
        confidence=0.93,
    )
    automatic_at_three = _calibration(
        expected[3],
        method="roboflow-field-keypoints",
        confidence=0.90,
    )
    manual_at_one = _calibration(
        expected[1],
        method="manual-pitch-anchors",
        confidence=0.96,
    )
    automatic = {1: automatic_at_one, 3: automatic_at_three}
    manual = {1: manual_at_one}

    merged = _merge_direct_calibration_anchors(automatic, manual)

    assert merged is not automatic
    assert merged is not manual
    assert merged == {1: manual_at_one, 3: automatic_at_three}
    assert merged[1] is manual_at_one
    assert automatic[1] is automatic_at_one
    assert manual == {1: manual_at_one}


def test_same_sample_manual_precedence_still_interpolates_surrounding_frames() -> None:
    frames, motion_edges, expected = _fixture(frame_count=3)
    automatic = {
        1: _calibration(
            expected[1],
            method="pnlcalib-points-lines",
            confidence=0.93,
        )
    }
    manual = {
        1: _calibration(
            expected[1],
            method="manual-pitch-anchors",
            confidence=0.96,
        )
    }
    direct = _merge_direct_calibration_anchors(automatic, manual)

    result = solve_calibration_sequence(frames, direct, motion_edges)

    assert result[1].projection_source == "direct"
    assert result[1].selected is not None
    assert result[1].selected.calibration.method == "manual-pitch-anchors"
    assert result[0].projection_source == "temporal-backward"
    assert result[0].selected is not None
    assert result[0].selected.anchor_sample_index == 1
    _assert_same_mapping(result[0].selected.calibration.image_to_pitch, expected[0])
    assert result[2].projection_source == "temporal-forward"
    assert result[2].selected is not None
    assert result[2].selected.anchor_sample_index == 1
    _assert_same_mapping(result[2].selected.calibration.image_to_pitch, expected[2])
