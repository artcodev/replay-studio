from dataclasses import fields

import pytest

import app.ball_tracking_solver as ball_tracking_solver
from app.ball_tracking import resolve_ball_trajectory
from app.ball_tracking_contract import BallTrackingConfig


FRAME_SIZE = (960, 540)
PITCH = {"length": 105, "width": 68}


def test_global_path_prefers_physically_consistent_metric_candidates():
    frames = []
    for index in range(6):
        frames.append(
            (
                [
                    {
                        "id": f"true-{index}",
                        "x": 300 + index * 4,
                        "y": 280,
                        "confidence": 0.55,
                        "pitchX": index * 0.5,
                        "pitchZ": 1.0,
                    },
                    {
                        "id": f"false-{index}",
                        "x": 700 if index % 2 == 0 else 100,
                        "y": 100,
                        "confidence": 0.90,
                        "pitchX": 50.0 if index % 2 == 0 else -50.0,
                        "pitchZ": 30.0,
                    },
                ],
                index * 0.1,
            )
        )

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH)

    assert result.diagnostics["selectedCandidateIds"] == [
        f"true-{index}" for index in range(6)
    ]
    assert result.diagnostics["motion"]["metricTransitionCount"] == 5
    assert result.diagnostics["motion"]["speedViolationCount"] == 0
    assert [keyframe["x"] for keyframe in result.keyframes] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )


def test_short_bounded_gap_is_inferred_and_preserves_detector_provenance():
    frames = [
        (
            [
                {
                    "candidateId": "roboflow-a",
                    "x": 100,
                    "y": 200,
                    "confidence": 0.8,
                    "pitchX": 0.0,
                    "pitchZ": 0.0,
                    "provenance": {"model": "football-ball-v4", "tile": [0, 0]},
                }
            ],
            0.0,
        ),
        ([], 0.1),
        (
            [
                {
                    "candidateId": "roboflow-b",
                    "x": 110,
                    "y": 200,
                    "confidence": 0.75,
                    "pitchX": 1.0,
                    "pitchZ": 0.0,
                    "provenance": {"model": "football-ball-v4", "tile": [0, 1]},
                }
            ],
            0.2,
        ),
    ]

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH)

    assert [item["state"] for item in result.keyframes] == [
        "observed",
        "inferred",
        "observed",
    ]
    assert result.keyframes[0]["confidence"] == 0.8
    assert result.keyframes[0]["detectionConfidence"] == 0.8
    assert result.keyframes[0]["candidateProvenance"] == {
        "model": "football-ball-v4",
        "tile": [0, 0],
        "candidateId": "roboflow-a",
    }
    assert result.keyframes[1]["x"] == pytest.approx(0.5)
    assert result.keyframes[1]["sourceCandidateIds"] == [
        "roboflow-a",
        "roboflow-b",
    ]
    assert result.diagnostics["inferredFrameCount"] == 1
    assert result.diagnostics["publishedCoverage"] == 1.0


def test_long_gap_remains_occluded_instead_of_fabricating_keyframes():
    config = BallTrackingConfig(max_interpolation_gap_seconds=0.5)
    frames = [
        ([{"id": "left", "x": 100, "y": 200, "confidence": 0.8}], 0.0),
        ([], 0.6),
        ([{"id": "right", "x": 110, "y": 200, "confidence": 0.8}], 1.2),
    ]

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH, config=config)

    assert [item["state"] for item in result.keyframes] == ["observed", "observed"]
    assert [item["state"] for item in result.diagnostics["path"]] == [
        "observed",
        "occluded",
        "observed",
    ]
    assert result.diagnostics["path"][1]["reason"] == (
        "gap-exceeds-interpolation-limit"
    )
    assert result.diagnostics["occludedFrameCount"] == 1


def test_camera_stabilized_coordinates_are_used_for_motion_not_projection():
    frames = []
    for index in range(4):
        frames.append(
            (
                [
                    {
                        "id": f"ball-{index}",
                        "x": 100.0 + index * 600.0,
                        "y": 220.0,
                        "stabilizedX": 100.0 + index * 3.0,
                        "stabilizedY": 220.0,
                        "confidence": 0.8,
                    }
                ],
                index * 0.1,
            )
        )

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH)

    motion = result.diagnostics["motion"]
    assert motion["stabilizedTransitionCount"] == 3
    assert motion["speedViolationCount"] == 0
    assert result.keyframes[1]["imagePosition"]["x"] == 700.0
    assert result.keyframes[1]["stabilizedImagePosition"] == {
        "x": 103.0,
        "y": 220.0,
        "source": "camera-stabilized",
    }


def test_custom_camera_motion_and_projection_hooks_are_supported():
    frames = [
        (
            [
                {
                    "id": f"custom-{index}",
                    "x": 100.0 + index * 500.0,
                    "y": 200.0,
                    "confidence": 0.85,
                    "flow": {"x": 50.0 + index * 2.0, "y": 70.0},
                }
            ],
            index * 0.1,
        )
        for index in range(3)
    ]

    result = resolve_ball_trajectory(
        frames,
        FRAME_SIZE,
        PITCH,
        coordinate_selector=lambda candidate, _: (
            candidate["flow"]["x"],
            candidate["flow"]["y"],
            "optical-flow",
        ),
        projector=lambda candidate: {
            "x": candidate["x"] / 100.0,
            "z": 3.0,
            "projectionSource": "test-calibration",
            "positionUncertaintyMetres": 0.4,
        },
    )

    assert result.diagnostics["motion"]["stabilizedTransitionCount"] == 2
    assert [item["x"] for item in result.keyframes] == [1.0, 6.0, 11.0]
    assert all(item["projectionSource"] == "test-calibration" for item in result.keyframes)


def test_top_k_and_invalid_candidates_are_reported():
    config = BallTrackingConfig(top_k_per_frame=2)
    frames = []
    for index in range(2):
        frames.append(
            {
                "t": index * 0.1,
                "candidates": [
                    {"id": "invalid", "x": None, "y": 1, "confidence": 0.99},
                    {"id": f"a-{index}", "x": 100 + index, "y": 200, "score": 0.8},
                    {"id": f"b-{index}", "x": 300, "y": 200, "score": 0.7},
                    {"id": f"c-{index}", "x": 500, "y": 200, "score": 0.6},
                ],
            }
        )

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH, config=config)

    assert result.diagnostics["invalidCandidateCount"] == 2
    assert result.diagnostics["droppedByTopKCount"] == 2
    assert result.diagnostics["candidateCount"] == 4


def test_insufficient_evidence_returns_empty_track_with_diagnostics():
    result = resolve_ball_trajectory(
        [([{"id": "single", "x": 100, "y": 200, "confidence": 0.9}], 0.0)],
        FRAME_SIZE,
        PITCH,
    )

    assert result.keyframes == []
    assert result.diagnostics["status"] == "no-stable-trajectory"
    assert result.diagnostics["path"][0]["state"] == "occluded"


def test_non_monotonic_timestamps_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_ball_trajectory(
            [([], 0.2), ([], 0.2)],
            FRAME_SIZE,
            PITCH,
        )


def test_long_sequence_uses_shared_backpointers_and_keeps_beam_bounded(monkeypatch):
    frame_count = 2_000
    config = BallTrackingConfig(top_k_per_frame=1, beam_width=8)
    materialised_depths = []
    original_materialise = ball_tracking_solver._materialise_steps

    def record_materialisation(tail):
        materialised_depths.append(tail.depth)
        return original_materialise(tail)

    monkeypatch.setattr(
        ball_tracking_solver, "_materialise_steps", record_materialisation
    )
    frames = [
        (
            [
                {
                    "id": f"ball-{index}",
                    "x": 100.0 + index * 0.1,
                    "y": 220.0,
                    "confidence": 0.95,
                }
            ],
            index * 0.04,
        )
        for index in range(frame_count)
    ]

    result = resolve_ball_trajectory(frames, FRAME_SIZE, PITCH, config=config)

    hypothesis_fields = {
        item.name for item in fields(ball_tracking_solver._Hypothesis)
    }
    assert "tail" in hypothesis_fields
    assert "steps" not in hypothesis_fields
    assert materialised_depths == [frame_count]
    assert len(result.diagnostics["path"]) == frame_count
    assert len(result.keyframes) == frame_count
    assert result.diagnostics["selectedCandidateIds"][0] == "ball-0"
    assert result.diagnostics["selectedCandidateIds"][-1] == "ball-1999"
    assert result.diagnostics["peakHypothesisCount"] <= (
        config.beam_width * (config.top_k_per_frame + 1)
    )
