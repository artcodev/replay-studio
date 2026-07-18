from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pitch_anchor_calibration import calibration_from_anchors
from app.pitch_calibration_contract import PitchCalibration
from app.pitch_calibration_orientation import canonicalize_penalty_side
from app.pitch_geometry import projected_pitch_markings
from app.reconstruction_calibration_apply import apply_scene_pitch_calibration
from app.reconstruction_pitch_side_command import set_scene_pitch_side
from app.reconstruction_errors import IdentityCorrectionError, ReconstructionError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete as delete_frame_person_annotation,
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_progress import ReconstructionProgress
from app.reconstruction_calibration_detection import (
    best_pitch_calibration as _best_pitch_calibration,
)
from app.reconstruction_metric_projection import (
    calibration_person_support as _calibration_person_support,
)
from app.reconstruction_identity_correction_service import (
    apply_track_identity_corrections as _apply_track_identity_corrections,
)
from app.reconstruction_identity_scene_corrections import (
    apply_scene_track_identity_corrections as _apply_scene_track_identity_corrections,
)
from app.reconstruction_identity_splitting import (
    apply_canonical_split_corrections as _apply_canonical_split_corrections,
)
from app.reconstruction_identity_read_model import (
    interpolate_scene_keyframes as _interpolate_scene_keyframes,
    saved_pitch_calibration as _saved_pitch_calibration,
)
from app.reconstruction_identity_validation import (
    validate_identity_corrections as _validate_identity_corrections,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)
from app.reconstruction_team_classification import (
    cluster_color as _cluster_color,
    include_goalkeeper_candidates as _include_goalkeeper_candidates,
)
from app.person_appearance import is_pitch_person as _is_pitch_person
from app.reconstruction_person_annotations import (
    apply_person_annotations as _apply_person_annotations,
    frame_annotations as _frame_annotations,
)
from app.reconstruction_pitch_projection import (
    project_pitch_point as _project,
    project_pitch_point_unclamped as _project_unclamped,
)
from app.scene_document import reconstruction_input_fingerprint


def test_project_maps_screen_center_to_pitch_center_and_clamps_edges():
    pitch = {"length": 105, "width": 68}

    assert _project(480, 270, 960, 540, pitch) == (0.0, 0.0)
    assert _project(-500, 900, 960, 540, pitch) == (-52.5, 34.0)


def test_unclamped_projection_retains_outside_pitch_evidence():
    pitch = {"length": 105, "width": 68}

    x, z = _project_unclamped(-500, 900, 960, 540, pitch)

    assert x < -52.5
    assert z > 34.0


def test_project_uses_metric_pitch_calibration_when_available():
    pitch = {"length": 105, "width": 68}
    calibration = PitchCalibration(
        image_to_pitch=np.array(
            [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.8,
        supported_lines=6,
        mean_line_score=0.7,
        rectangle="penalty-area-right",
    )

    assert _project(480, 270, 960, 540, pitch, calibration) == (0.0, 0.0)
    assert _project(600, 350, 960, 540, pitch, calibration) == (12.0, 8.0)


def test_frame_analysis_accepts_review_calibration_as_metric_evidence():
    scene = {
        "payload": {
            "videoAsset": {
                "reconstruction": {
                    "pitchCalibration": {
                        "status": "review",
                        "imageToPitch": [
                            [0.1, 0.0, -48.0],
                            [0.0, 0.1, -27.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "confidence": 0.8,
                        "supportedLines": 6,
                        "meanLineScore": 0.7,
                        "rectangle": "penalty-area-right",
                    }
                }
            }
        }
    }

    calibration = _saved_pitch_calibration(scene)

    assert calibration is not None
    assert _project(480, 270, 960, 540, {"length": 105, "width": 68}, calibration) == (0.0, 0.0)


def test_metric_projection_outlier_falls_back_instead_of_piling_on_pitch_corner():
    pitch = {"length": 105, "width": 68}
    calibration = PitchCalibration(
        image_to_pitch=np.array(
            [[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.9,
        supported_lines=8,
        mean_line_score=0.9,
        rectangle="field-keypoints-left",
    )

    assert _project(480, 270, 960, 540, pitch, calibration) == (0.0, 0.0)


def test_pnlcalib_is_representative_even_when_local_fallback_reports_higher_confidence():
    pnl = PitchCalibration(
        np.eye(3),
        0.76,
        8,
        0.8,
        "field-keypoints-right",
        method="pnlcalib-points-lines",
    )
    local = PitchCalibration(
        np.eye(3),
        0.92,
        9,
        0.9,
        "field-keypoints-left",
        method="roboflow-field-keypoints",
    )

    assert _best_pitch_calibration({1: local, 2: pnl}) is pnl


def test_calibration_support_rejects_people_projected_outside_pitch():
    calibration = PitchCalibration(
        np.array([[1.0, 0.0, -1000.0], [0.0, 1.0, -1000.0], [0.0, 0.0, 1.0]]),
        0.9,
        8,
        0.9,
        "field-keypoints-left",
    )
    people = [
        Detection(100.0 + index * 20, 250.0, 18.0, 44.0, 0.8, np.zeros(12))
        for index in range(6)
    ]

    assert _calibration_person_support(people, calibration, {"length": 105, "width": 68}) == (0, 6)


def test_project_uses_visible_half_when_metric_fit_is_weak():
    pitch = {"length": 105, "width": 68}
    calibration = PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.69,
        supported_lines=6,
        mean_line_score=0.6,
        rectangle="penalty-area-right",
    )

    assert _project(0, 270, 960, 540, pitch, calibration) == (0.0, 0.0)
    assert _project(960, 270, 960, 540, pitch, calibration) == (52.5, 0.0)


def test_manual_pitch_anchors_define_metric_homography_and_overlay():
    anchors = [
        {"image": {"x": 100, "y": 100}, "pitch": {"x": -52.5, "z": -34}},
        {"image": {"x": 500, "y": 100}, "pitch": {"x": 52.5, "z": -34}},
        {"image": {"x": 100, "y": 400}, "pitch": {"x": -52.5, "z": 34}},
        {"image": {"x": 500, "y": 400}, "pitch": {"x": 52.5, "z": 34}},
    ]

    calibration = calibration_from_anchors(anchors, "center-circle")
    projected = calibration.image_to_pitch @ np.array([300.0, 250.0, 1.0])
    markings = projected_pitch_markings(calibration, 600, 500)

    assert abs(float(projected[0] / projected[2])) < 0.01
    assert abs(float(projected[1] / projected[2])) < 0.01
    assert any(marking["id"] == "center-circle" for marking in markings)


def test_cluster_color_keeps_red_jersey_red():
    center = np.zeros(12, dtype=np.float32)
    center[0] = 1.0
    center[10] = 0.8
    center[11] = 0.8

    color = _cluster_color(center)

    assert color.startswith("#e1")


def test_far_pitch_player_is_not_rejected_by_old_horizon_cutoff():
    hsv = np.zeros((540, 960, 3), dtype=np.uint8)
    hsv[:, :, 0] = 60
    hsv[:, :, 1] = 190
    hsv[:, :, 2] = 140
    far_player = (145.5, 139.0, 161.0, 171.5)

    assert _is_pitch_person(hsv, far_player, 0.636)
    assert not _is_pitch_person(hsv, far_player, 0.06)


def test_penalty_side_uses_fitted_rectangles_screen_position():
    calibration = PitchCalibration(
        image_to_pitch=np.array(
            [[1.0, 0.0, -844.25], [0.0, 1.0, -270.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.8,
        supported_lines=6,
        mean_line_score=0.7,
        rectangle="penalty-area-left",
    )

    corrected = canonicalize_penalty_side(calibration, 960)

    assert corrected.rectangle == "penalty-area-right"
    projected = corrected.image_to_pitch @ np.array([800.0, 270.0, 1.0])
    assert projected[0] == 44.25


def test_goal_area_side_is_canonicalized_from_screen_position():
    calibration = PitchCalibration(
        image_to_pitch=np.array(
            [[1.0, 0.0, -749.75], [0.0, 1.0, -270.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.9,
        supported_lines=4,
        mean_line_score=0.8,
        rectangle="goal-area-left",
    )

    corrected = canonicalize_penalty_side(calibration, 960)

    assert corrected.rectangle == "goal-area-right"
    pitch_to_image = np.linalg.inv(corrected.image_to_pitch)
    projected = pitch_to_image @ np.array([49.75, 0.0, 1.0])
    assert round(float(projected[0] / projected[2]), 2) == 700.0


def test_setting_attacking_goal_does_not_flip_geometric_pitch_side(monkeypatch):
    scene = {
        "id": "shot-side",
        "payload": {
            "videoAsset": {
                "selectedSegmentId": "segment-1",
                "reconstruction": {
                    "status": "ready",
                    "pitchCalibration": {
                        "rectangle": "goal-area-right",
                        "pitchSide": "right",
                        "imageToPitch": [[1.0, 0.0, -20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    "pitchCalibrationOverrides": [
                        {
                            "preset": "goal-area-right",
                            "imageToPitch": [[1.0, 0.0, -20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "anchors": [{"pitch": {"x": 47.0, "z": -9.16}}],
                        }
                    ],
                },
            },
            "tracks": [{"keyframes": [{"x": 12.5, "z": 3.0}]}],
            "ball": {"keyframes": [{"x": -4.5, "z": 2.0}]},
        },
    }
    monkeypatch.setattr(
        "app.reconstruction_pitch_side_command.scenes.put",
        lambda value: value,
    )

    set_scene_pitch_side(scene, "left")
    set_scene_pitch_side(scene, "left")

    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    assert scene["payload"]["tracks"][0]["keyframes"][0]["x"] == 12.5
    assert scene["payload"]["ball"]["keyframes"][0]["x"] == -4.5
    assert reconstruction["pitchCalibration"]["rectangle"] == "goal-area-right"
    assert reconstruction["pitchCalibration"]["imageToPitch"][0] == [1.0, 0.0, -20.0]
    assert reconstruction["pitchCalibrationOverrides"][0]["preset"] == "goal-area-right"
    assert reconstruction["pitchCalibrationOverrides"][0]["anchors"][0]["pitch"]["x"] == 47.0
    assert reconstruction["pitchOrientation"]["attackingGoal"] == "left"
    assert reconstruction["pitchOrientation"]["attackingGoalSource"] == "manual"


def test_apply_pitch_calibration_saves_stabilized_override(monkeypatch):
    scene = {
        "id": "shot-1",
        "title": "Shot 1",
        "duration": 4.0,
        "payload": {
            "videoAsset": {"id": "asset-1", "reconstruction": {"status": "ready"}},
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    draft = {
        "sceneTime": 0.6,
        "frameIndex": 4,
        "confidence": 0.91,
        "alignmentError": 3.2,
        "alignmentMetrics": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.747,
            "residualP50": 3.2,
            "residualP95": 6.4,
        },
        "quality": "good",
        "imageToPitch": [[0.1, 0, -48], [0, 0.1, -27], [0, 0, 1]],
    }
    anchors = [
        {"id": str(index), "label": str(index), "image": {"x": index, "y": index}, "pitch": {"x": index, "z": index}}
        for index in range(4)
    ]
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.preview_scene_pitch_calibration",
        lambda *args: draft,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.calibration_frame_context",
        lambda *args: (3, 0.6, np.zeros((540, 960, 3), dtype=np.uint8), np.eye(3)),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.frame_paths", lambda *args: []
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.scenes.put",
        lambda value: value,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.queue_reconstruction",
        lambda value, **_kwargs: value,
    )

    applied = apply_scene_pitch_calibration(scene, 0.6, "center-circle", anchors)

    override = applied["payload"]["videoAsset"]["reconstruction"]["pitchCalibrationOverrides"][0]
    assert override["method"] == "manual-pitch-anchors"
    assert override["alignmentError"] == 3.2
    assert override["imageToPitch"] == draft["imageToPitch"]
