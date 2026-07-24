import numpy as np

from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_calibration_evidence import keypoint_evidence as _keypoint_evidence
from app.reconstruction_shot_calibration_quality import (
    evaluate_calibration_quality as _evaluate_calibration_quality,
)
from app.reconstruction_track_state import TrackState
from app.reconstruction_latent_presence import (
    materialize_continuous_presence as _materialize_continuous_presence,
)
from app.reconstruction_scene_track_publisher import (
    publish_scene_tracks as _scene_tracks,
)


def _frame(index: int, status: str = "accepted") -> dict:
    return {
        "sourceFrameIndex": index + 1,
        "sampleIndex": index,
        "sceneTime": index * 0.2,
        "status": status,
        "projectionSource": "direct" if status == "accepted" else "none",
        "reprojectionError": 2.0 if status == "accepted" else None,
        "reprojectionP95": 4.0 if status == "accepted" else None,
        "visiblePitchSide": "right" if status == "accepted" else None,
        "alignmentMetrics": {"f1": 0.5} if status == "accepted" else None,
    }


def test_calibration_quality_requires_shot_wide_coverage():
    passing = [_frame(index) for index in range(6)]
    sparse = [_frame(index, "accepted" if index < 2 else "missing") for index in range(6)]

    assert _evaluate_calibration_quality(passing)["verdict"] == "pass"
    rejected = _evaluate_calibration_quality(sparse)
    assert rejected["verdict"] == "reject"
    assert rejected["summary"]["usableCoverage"] == 0.333
    assert "calibration-coverage" in rejected["failedGateIds"]


def test_recovered_frames_do_not_multiply_their_anchor_orientation_vote():
    evidence = [_frame(index) for index in range(10)]
    evidence[0]["visiblePitchSide"] = "left"
    evidence[9]["visiblePitchSide"] = "right"
    for index in range(1, 9):
        evidence[index].update(
            {
                "projectionSource": "temporal-forward",
                "visiblePitchSide": "left",
                "solutionStatus": "temporal-accepted",
                "temporal": {"anchorFrameIndices": [1]},
                "uncertainty": {"p95Metres": 1.5},
            }
        )

    report = _evaluate_calibration_quality(evidence)

    assert report["summary"]["sideVotes"] == {"left": 1, "right": 1}
    assert report["summary"]["sideAgreement"] == 0.5
    assert report["verdict"] == "reject"


def test_metric_tracks_skip_observations_without_calibration(monkeypatch):
    points = []
    for index in range(5):
        point = {
            "t": index * 0.2,
            "px": 100.0 + index,
            "py": 200.0,
            "confidence": 0.8,
        }
        if index != 2:
            point.update(
                {
                    "pitchX": float(index),
                    "pitchZ": 1.0,
                    "projectionSource": "direct",
                    "calibrationFrameIndex": index + 1,
                    "positionUncertaintyMetres": 0.5,
                }
            )
        points.append(point)
    track = TrackState(
        id=1,
        points=points,
        feature_sum=np.zeros(12, dtype=np.float32),
        feature_count=1,
        last_frame=4,
        last_height=40.0,
        annotation_ids={"manual-1"},
        manual_kind="home-player",
    )
    scene = {
        "duration": 0.8,
        "payload": {"pitch": {"length": 105, "width": 68}},
    }
    monkeypatch.setattr(
        "app.reconstruction_scene_track_publisher.frame_paths",
        lambda _: [(f"frame-{index}", index * 0.2) for index in range(5)],
    )

    result = _scene_tracks(
        [track],
        {1: "home"},
        {"home": "#fff"},
        (960, 540),
        scene,
        coordinate_mode="metric",
    )

    assert len(result) == 1
    assert {frame["t"] for frame in result[0]["keyframes"]} == {0.0, 0.2, 0.6, 0.8}
    assert all(
        frame["projection"]["source"] == "direct"
        for frame in result[0]["keyframes"]
    )


def test_presence_is_bounded_by_observed_lifetime():
    keyframes, presence = _materialize_continuous_presence(
        [
            {"t": 1.0, "x": 8.0, "z": 4.0, "confidence": 0.9},
            {"t": 1.2, "x": 8.5, "z": 4.2, "confidence": 0.85},
        ],
        3.0,
        {"length": 105, "width": 68},
        7,
    )

    assert keyframes[0]["t"] == 1.0
    assert keyframes[-1]["t"] == 1.2
    assert keyframes[0]["presenceState"] == "observed"
    assert keyframes[-1]["presenceState"] == "observed"
    assert sum(frame["observed"] is True for frame in keyframes) == 2
    assert presence["policy"] == "observed-window-with-latent-gaps"
    assert presence["coverage"] == 0.067
    assert presence["observationCount"] == 2
    assert presence["inferredKeyframeCount"] == 0


def test_continuous_presence_marks_long_internal_gap_as_inferred():
    keyframes, _ = _materialize_continuous_presence(
        [
            {"t": 0.0, "x": -10.0, "z": 0.0, "confidence": 0.9},
            {"t": 2.0, "x": 0.0, "z": 4.0, "confidence": 0.9},
        ],
        2.0,
        {"length": 105, "width": 68},
        3,
    )

    inferred = [frame for frame in keyframes if frame["presenceState"] == "inferred-gap"]
    assert inferred
    assert all(-10.0 < frame["x"] < 0.0 for frame in inferred)
    assert all(frame["observed"] is False for frame in inferred)


def test_manual_confirmation_is_not_dropped_by_team_capacity(monkeypatch):
    def automatic(track_id: int) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {"t": index * 0.2, "px": 100.0 + track_id, "py": 200.0, "confidence": 0.8}
                for index in range(5)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=4,
            last_height=40.0,
        )

    tracks = [automatic(track_id) for track_id in range(1, 12)]
    tracks.append(
        TrackState(
            id=99,
            points=[{"t": 0.4, "px": 600.0, "py": 220.0, "confidence": 1.0}],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=2,
            last_height=45.0,
            annotation_ids={"confirmed-person"},
            manual_kind="home-player",
            manual_label="Confirmed person",
        )
    )
    scene = {
        "duration": 0.8,
        "payload": {"pitch": {"length": 105, "width": 68}},
    }
    monkeypatch.setattr(
        "app.reconstruction_scene_track_publisher.frame_paths",
        lambda _: [(f"frame-{index}", index * 0.2) for index in range(5)],
    )

    result = _scene_tracks(
        tracks,
        {track.id: "home" for track in tracks},
        {"home": "#fff"},
        (960, 540),
        scene,
        coordinate_mode="approximate",
    )

    assert len(result) == 11
    assert any(track.get("annotationIds") == ["confirmed-person"] for track in result)


def test_keypoint_evidence_contains_reprojection_vector():
    calibration = PitchCalibration(
        image_to_pitch=np.eye(3),
        confidence=0.9,
        supported_lines=6,
        mean_line_score=0.9,
        rectangle="field-keypoints-right",
        raw_keypoints=(
            {
                "id": 1,
                "image": {"x": 12.0, "y": 8.0},
                "pitch": {"x": 12.0, "z": 8.0},
                "inlier": True,
            },
        ),
    )

    evidence = _keypoint_evidence(calibration)

    assert evidence[0]["projectedImage"] == {"x": 12.0, "y": 8.0}
    assert evidence[0]["residualVector"]["magnitude"] == 0.0
