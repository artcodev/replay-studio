from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pitch_calibration import (
    PitchCalibration,
    calibration_from_anchors,
    canonicalize_penalty_side,
    projected_pitch_markings,
)
from app.reconstruction import (
    Detection,
    IdentityCorrectionError,
    ReconstructionError,
    ReconstructionProgress,
    TrackState,
    _apply_canonical_split_corrections,
    _apply_scene_track_identity_corrections,
    _apply_track_identity_corrections,
    _apply_person_annotations,
    _ball_keyframes,
    _best_pitch_calibration,
    _calibration_person_support,
    _cluster_color,
    _frame_annotations,
    _include_goalkeeper_candidates,
    _interpolate_scene_keyframes,
    _is_pitch_person,
    _project,
    _project_unclamped,
    _resolve_canonical_track_states,
    _saved_pitch_calibration,
    _validate_identity_corrections,
    apply_scene_pitch_calibration,
    delete_frame_person_annotation,
    queue_reconstruction,
    set_scene_pitch_side,
    upsert_frame_person_annotation,
)
from app.store import reconstruction_input_fingerprint


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


def test_frame_annotations_can_add_and_ignore_detections():
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    feature = np.zeros(12, dtype=np.float32)
    automatic = Detection(300, 250, 20, 50, 0.7, feature)
    ignored = {
        "id": "ignored",
        "kind": "ignore",
        "bbox": {"x": 290, "y": 200, "width": 20, "height": 50},
    }
    manual = {
        "id": "manual",
        "kind": "home-player",
        "label": "Player A",
        "externalPlayerId": None,
        "bbox": {"x": 140, "y": 135, "width": 18, "height": 38},
    }

    result = _apply_person_annotations(image, [automatic], [ignored, manual])

    assert len(result) == 1
    assert result[0].annotation_id == "manual"
    assert result[0].annotation_kind == "home-player"
    assert result[0].annotation_label == "Player A"


def test_two_explicit_identity_owners_cannot_claim_the_same_observation():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    shared = {
        "sceneTime": 0.0,
        "frameIndex": 0,
        "bbox": {"x": 100, "y": 80, "width": 24, "height": 60},
        "kind": "home-player",
        "action": "confirm",
        "scope": "identity",
    }

    with pytest.raises(
        ReconstructionError,
        match="Conflicting explicit canonical identities target one observation",
    ):
        _apply_person_annotations(
            image,
            [],
            [
                {**shared, "id": "confirm-a", "canonicalPersonId": "canonical-a"},
                {**shared, "id": "confirm-b", "canonicalPersonId": "canonical-b"},
            ],
        )

    detection = Detection(
        x=112,
        y=140,
        width=24,
        height=60,
        confidence=1.0,
        feature=np.ones(8, dtype=np.float32),
        annotation_id="stale-import",
        manual_identity_owner_ids={"canonical-a", "canonical-b"},
    )
    track = TrackState(id=1)
    with pytest.raises(
        ReconstructionError,
        match="Conflicting explicit canonical identities reached one raw track",
    ):
        track.append(detection, 0, 0.0)
    assert track.points == []


def test_identity_exclude_keeps_exact_negative_anchor_until_tracking():
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    detection = Detection(
        x=300,
        y=250,
        width=20,
        height=50,
        confidence=0.8,
        feature=np.zeros(12, dtype=np.float32),
    )
    annotation = {
        "id": "phantom",
        "kind": "ignore",
        "action": "exclude",
        "scope": "identity",
        "bbox": {"x": 290, "y": 200, "width": 20, "height": 50},
    }

    result = _apply_person_annotations(image, [detection], [annotation])

    assert result == [detection]
    assert detection.annotation_id == "phantom"
    assert detection.annotation_kind == "ignore"


def test_frame_zero_annotation_is_addressable():
    annotation = {"id": "frame-zero", "frameIndex": 0}
    scene = {
        "payload": {
            "videoAsset": {"reconstruction": {"frameAnnotations": [annotation]}}
        }
    }

    assert _frame_annotations(scene, 0) == [annotation]


def test_frame_annotation_is_snapped_and_persisted(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "shot-label",
        "duration": 4.0,
        "payload": {
            "videoAsset": {"sourceStart": 10.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": None,
            "scene_time": 0.03,
            "bbox": {"x": 140, "y": 135, "width": 18, "height": 38},
            "kind": "home-player",
            "label": "Player A",
            "external_player_id": None,
        },
    )

    assert annotation["sceneTime"] == 0.0
    assert annotation["sourceTime"] == 10.0
    assert annotation["frameIndex"] == 1
    assert annotation["scope"] == "observation"
    assert scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == [annotation]

    with pytest.raises(ReconstructionError, match="canonical Bind / Unbind endpoint"):
        upsert_frame_person_annotation(
            scene,
            {
                "annotation_id": None,
                "scene_time": 0.03,
                "bbox": {"x": 140, "y": 135, "width": 18, "height": 38},
                "kind": "home-player",
                "label": "Player A",
                "external_player_id": "roster-home-8",
            },
        )


def test_frame_identity_merge_persists_target_semantics_without_roster_snapshot(
    monkeypatch, tmp_path
):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "shot-identity-merge",
        "duration": 4.0,
        "payload": {
            "tracks": [
                {
                    "id": "auto-away-03",
                    "label": "Away player 3",
                    "teamId": "away",
                    "role": "player",
                    "externalPlayerId": "roster-3",
                    "keyframes": [],
                }
            ],
            "videoAsset": {"sourceStart": 10.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-a",
            "scene_time": 0.0,
            "bbox": {"x": 140, "y": 135, "width": 18, "height": 38},
            "kind": "home-player",
            "label": "Wrong provisional label",
            "external_player_id": None,
            "action": "merge",
            "merge_target_id": "auto-away-03",
            "source_track_id": "auto-home-02",
        },
    )

    assert annotation["action"] == "merge"
    assert annotation["scope"] == "identity"
    assert annotation["mergeTargetId"] == "auto-away-03"
    assert annotation["sourceTrackId"] == "auto-home-02"
    assert annotation["previewState"] == "merged"
    assert annotation["kind"] == "away-player"
    assert annotation["label"] == "Away player 3"
    # Roster state is live canonical state owned by the dedicated Bind / Unbind
    # correction.  A merge may inherit role/label for its preview, but must not
    # retain a stale roster snapshot of its target.
    assert annotation["externalPlayerId"] is None


def test_frame_identity_merge_rejects_invalid_self_and_cyclic_targets(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "shot-identity-validation",
        "duration": 4.0,
        "payload": {
            "tracks": [{"id": "track-a", "teamId": "home", "keyframes": []}],
            "videoAsset": {"sourceStart": 0.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    base = {
        "scene_time": 0.0,
        "bbox": {"x": 100, "y": 100, "width": 20, "height": 40},
        "kind": "home-player",
        "label": None,
        "external_player_id": None,
    }
    with np.testing.assert_raises_regex(ReconstructionError, "already belongs"):
        upsert_frame_person_annotation(
            scene,
            {
                **base,
                "annotation_id": "self-track",
                "action": "merge",
                "merge_target_id": "track-a",
                "source_track_id": "track-a",
            },
        )
    with np.testing.assert_raises_regex(ReconstructionError, "no longer exists"):
        upsert_frame_person_annotation(
            scene,
            {
                **base,
                "annotation_id": "missing-target",
                "action": "merge",
                "merge_target_id": "does-not-exist",
                "source_track_id": None,
            },
        )

    upsert_frame_person_annotation(
        scene,
        {**base, "annotation_id": "person-b", "action": "confirm"},
    )
    upsert_frame_person_annotation(
        scene,
        {
            **base,
            "annotation_id": "person-a",
            "action": "merge",
            "merge_target_id": "person-b",
        },
    )
    with np.testing.assert_raises_regex(ReconstructionError, "cycle"):
        upsert_frame_person_annotation(
            scene,
            {
                **base,
                "annotation_id": "person-b",
                "action": "merge",
                "merge_target_id": "person-a",
            },
        )


def test_identity_correction_rejects_inconsistent_kind_and_missing_identity_source(
    monkeypatch, tmp_path
):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "shot-identity-contract",
        "duration": 4.0,
        "payload": {
            "tracks": [],
            "videoAsset": {"sourceStart": 0.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)
    base = {
        "scene_time": 0.0,
        "bbox": {"x": 100, "y": 100, "width": 20, "height": 40},
        "label": None,
        "external_player_id": None,
    }

    with np.testing.assert_raises_regex(ReconstructionError, "Choose a person role"):
        upsert_frame_person_annotation(
            scene,
            {**base, "kind": "ignore", "action": "confirm"},
        )
    with np.testing.assert_raises_regex(ReconstructionError, "tracked identity"):
        upsert_frame_person_annotation(
            scene,
            {
                **base,
                "kind": "ignore",
                "action": "exclude",
                "scope": "identity",
                "source_track_id": None,
            },
        )


def test_legacy_ignore_without_action_remains_observation_scoped(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((120, 200, 3), dtype=np.uint8))
    scene = {
        "id": "shot-legacy-ignore",
        "duration": 4.0,
        "payload": {
            "tracks": [],
            "videoAsset": {"sourceStart": 0.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "scene_time": 0.0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "ignore",
            # Pydantic supplies the scope default even when a legacy client did
            # not send action/scope. Missing action remains authoritative here.
            "action": None,
            "scope": "identity",
            "source_track_id": None,
        },
    )

    assert annotation["action"] == "exclude"
    assert annotation["scope"] == "observation"
    assert annotation["sourceTrackId"] is None


def test_observation_exclude_does_not_block_merge_into_its_source_track(
    monkeypatch, tmp_path
):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    scene = {
        "id": "shot-observation-merge-target",
        "duration": 4.0,
        "payload": {
            "tracks": [
                {"id": "track-a", "label": "A", "teamId": "home", "keyframes": []},
                {"id": "track-b", "label": "B", "teamId": "home", "keyframes": []},
            ],
            "videoAsset": {"sourceStart": 0.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)
    base = {
        "scene_time": 0.0,
        "bbox": {"x": 100, "y": 100, "width": 20, "height": 40},
        "label": None,
        "external_player_id": None,
    }
    upsert_frame_person_annotation(
        scene,
        {
            **base,
            "annotation_id": "one-frame",
            "kind": "ignore",
            "action": "exclude",
            "scope": "observation",
            "source_track_id": "track-a",
        },
    )

    merged = upsert_frame_person_annotation(
        scene,
        {
            **base,
            "annotation_id": "merge-b",
            "kind": "home-player",
            "action": "merge",
            "merge_target_id": "track-a",
            "source_track_id": "track-b",
        },
    )

    assert merged["mergeTargetId"] == "track-a"


def test_identity_merge_unifies_raw_association_tracks_and_scene_output():
    feature_a = np.zeros(12, dtype=np.float32)
    feature_a[0] = 1.0
    feature_b = np.zeros(12, dtype=np.float32)
    feature_b[1] = 1.0
    target_detection = Detection(100, 250, 20, 50, 0.8, feature_a)
    target_detection.annotation_id = "person-b"
    target_detection.annotation_kind = "home-player"
    source_detection = Detection(300, 250, 20, 50, 0.98, feature_b)
    source_detection.annotation_id = "person-a"
    source_detection.annotation_kind = "home-player"
    target = TrackState(id=1)
    target.append(target_detection, 0, 0.0)
    source = TrackState(id=2)
    source.append(source_detection, 1, 0.1)
    scene = {
        "payload": {
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {"id": "person-b", "kind": "home-player", "action": "confirm"},
                        {
                            "id": "person-a",
                            "kind": "home-player",
                            "action": "merge",
                            "mergeTargetId": "person-b",
                        },
                    ]
                }
            },
        }
    }

    corrected = _apply_track_identity_corrections([target, source], scene)

    assert len(corrected) == 1
    assert corrected[0].annotation_ids == {"person-a", "person-b"}
    assert [point["t"] for point in corrected[0].points] == [0.0, 0.1]

    rendered = _apply_scene_track_identity_corrections(
        [
            {
                "id": "auto-home-01",
                "label": "Player B",
                "teamId": "home",
                "annotationIds": ["person-b"],
                "keyframes": [{"t": 0.0, "x": 0.0, "z": 0.0, "confidence": 0.8}],
            },
            {
                "id": "auto-home-02",
                "label": "Player A",
                "teamId": "home",
                "annotationIds": ["person-a"],
                "keyframes": [{"t": 0.1, "x": 0.5, "z": 0.0, "confidence": 0.98}],
            },
        ],
        scene,
    )

    assert len(rendered) == 1
    assert rendered[0]["id"] == "auto-home-01"
    assert rendered[0]["annotationIds"] == ["person-a", "person-b"]
    assert rendered[0]["identityCorrection"]["status"] == "merged"
    assert rendered[0]["identityCorrection"]["mergedTrackIds"] == ["auto-home-02"]
    assert [item["t"] for item in rendered[0]["keyframes"]] == [0.0, 0.1]
    assert rendered[0]["presence"]["coverage"] == 1.0
    assert rendered[0]["presence"]["observationCount"] == 2


def test_explicit_merge_assigns_the_selected_canonical_target_owner_before_resolver():
    def track(track_id: int, start: float, x: float) -> TrackState:
        result = TrackState(id=track_id)
        for offset in range(3):
            time = start + offset * 0.25
            result.points.append(
                {
                    "t": time,
                    "px": x,
                    "py": 200.0,
                    "frameIndex": int(time * 100),
                    "observationId": f"obs-{track_id}-{offset}",
                    "sourceTrackletId": result.local_tracklet_id,
                    "bbox": {"x": x - 10, "y": 150, "width": 20, "height": 50},
                    "confidence": 0.9,
                    "annotationId": None,
                }
            )
        result.feature_sum = np.ones(12, dtype=np.float32) * 3
        result.feature_count = 3
        result.last_frame = int((start + 0.5) * 100)
        result.last_height = 50.0
        result.source_tracklet_ids = {result.local_tracklet_id}
        return result

    source = track(1, 0.0, 100.0)
    source.annotation_ids = {"merge-a-into-b"}
    source.points[0]["annotationId"] = "merge-a-into-b"
    source.manual_identity_owner_ids = {"canonical-a"}
    target = track(2, 1.0, 200.0)
    later_target_fragment = track(3, 2.0, 202.0)
    later_target_fragment.annotation_ids = {"confirm-b"}
    later_target_fragment.points[0]["annotationId"] = "confirm-b"
    later_target_fragment.manual_identity_owner_ids = {"canonical-b"}
    target_observations = [
        {
            "frameIndex": point["frameIndex"],
            "sceneTime": point["t"],
            "bbox": deepcopy(point["bbox"]),
            "observationId": point["observationId"],
        }
        for point in target.points
    ]
    scene = {
        "duration": 3.0,
        "payload": {
            "canonicalPeople": [
                {
                    "id": "canonical-a",
                    "canonicalPersonId": "canonical-a",
                    "teamId": "home",
                    "role": "player",
                    "observations": [],
                },
                {
                    "id": "canonical-b",
                    "canonicalPersonId": "canonical-b",
                    "teamId": "home",
                    "role": "player",
                    "observations": target_observations,
                },
            ],
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "merge-a-into-b",
                            "kind": "home-player",
                            "action": "merge",
                            "scope": "identity",
                            "canonicalPersonId": "canonical-a",
                            "mergeTargetId": "canonical-b",
                        }
                    ]
                }
            },
        },
    }

    corrected = _apply_track_identity_corrections(
        [source, target, later_target_fragment],
        scene,
    )
    merged = next(item for item in corrected if item.id == 2)
    assert merged.manual_identity_owner_ids == {"canonical-b"}

    resolved, _ = _resolve_canonical_track_states(
        corrected,
        {item.id: "home" for item in corrected},
    )
    assert len(resolved) == 1
    assert resolved[0].manual_identity_owner_ids == {"canonical-b"}
    assert resolved[0].source_tracklet_ids == {
        "tracklet-0001",
        "tracklet-0002",
        "tracklet-0003",
    }


def _split_test_track(track_id: int, *, target_observation_id: str = "detector-order-new") -> TrackState:
    points = []
    for index, time in enumerate((0.0, 1.0, 2.0, 3.0)):
        points.append(
            {
                "t": time,
                "px": 100.0 + index,
                "py": 200.0,
                "frameIndex": index * 10,
                "observationId": target_observation_id if time == 2.0 else f"obs-{index}",
                "sourceTrackletId": f"tracklet-{track_id:04d}",
                "bbox": {
                    "x": 90.0 + index,
                    "y": 150.0,
                    "width": 20.0,
                    "height": 50.0,
                },
                "confidence": 0.9,
                "annotationId": None,
            }
        )
    return TrackState(
        id=track_id,
        points=points,
        feature_sum=np.ones(12, dtype=np.float32),
        feature_count=4,
        last_frame=30,
        last_height=50.0,
        source_tracklet_ids={f"tracklet-{track_id:04d}"},
    )


def _split_test_annotation() -> dict:
    return {
        "id": "split-crossing",
        "kind": "home-player",
        "action": "split",
        "scope": "range",
        "canonicalPersonId": "canonical-original",
        "targetObservationId": "frame-000020:old-order-002",
        "targetObservation": {
            "observationId": "frame-000020:old-order-002",
            "frameIndex": 20,
            "sceneTime": 2.0,
            "bbox": {"x": 92.0, "y": 150.0, "width": 20.0, "height": 50.0},
            "canonicalPersonId": "canonical-original",
        },
        "rangeStart": 1.0,
        "rangeEnd": 3.0,
        "splitCanonicalPersonId": "canonical-split-manual",
    }


def test_frame_identity_split_snapshots_range_and_preview(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00020.jpg"
    cv2.imwrite(str(frame), np.zeros((540, 960, 3), dtype=np.uint8))
    observations = [
        {
            "id": f"stable-{index}",
            "observationId": f"stable-{index}",
            "frameIndex": index * 10,
            "sceneTime": float(index),
            "bbox": {"x": 90 + index, "y": 150, "width": 20, "height": 50},
            "confidence": 0.9,
        }
        for index in range(4)
    ]
    scene = {
        "id": "shot-split",
        "duration": 4.0,
        "payload": {
            "canonicalPeople": [
                {
                    "id": "canonical-original",
                    "canonicalPersonId": "canonical-original",
                    "displayName": "Home person",
                    "teamId": "home",
                    "role": "player",
                    "observations": observations,
                }
            ],
            "tracks": [],
            "videoAsset": {"sourceStart": 10.0, "reconstruction": {"status": "ready"}},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 2.0)])

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "split-a",
            "scene_time": 2.0,
            "bbox": observations[2]["bbox"],
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-original",
            "target_observation_id": "stable-2",
            "range_start": 1.0,
            "range_end": 3.0,
        },
        persist=False,
    )

    assert annotation["scope"] == "range"
    assert annotation["targetObservation"] == {
        "observationId": "stable-2",
        "frameIndex": 20,
        "sceneTime": 2.0,
        "bbox": {"x": 92.0, "y": 150.0, "width": 20.0, "height": 50.0},
        "canonicalPersonId": "canonical-original",
    }
    assert annotation["rangeStart"] == 1.0
    assert annotation["rangeEnd"] == 3.0
    assert annotation["affectedPreview"]["affectedObservationCount"] == 2
    assert annotation["affectedPreview"]["remainingObservationCount"] == 2
    assert annotation["splitCanonicalPersonId"].startswith("canonical-split-")
    assert annotation["externalPlayerId"] is None


def test_split_survives_detector_reorder_by_geometry_and_never_uses_recycled_id():
    source = _split_test_track(1)
    neighbour = TrackState(
        id=2,
        points=[
            {
                "t": 2.0,
                "px": 300.0,
                "py": 200.0,
                "frameIndex": 20,
                # A detector-index ID was recycled for another person.
                "observationId": "frame-000020:old-order-002",
                "sourceTrackletId": "tracklet-0002",
                "bbox": {"x": 290.0, "y": 150.0, "width": 20.0, "height": 50.0},
                "confidence": 0.95,
                "annotationId": None,
            }
        ],
        feature_sum=np.ones(12, dtype=np.float32),
        feature_count=1,
        last_frame=20,
        last_height=50.0,
    )
    scene = {
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "reconstruction": {"frameAnnotations": [_split_test_annotation()]}
            }
        },
    }

    corrected, diagnostics = _apply_canonical_split_corrections([source, neighbour], scene)

    original = next(track for track in corrected if track.canonical_person_id == "canonical-original")
    split = next(track for track in corrected if track.canonical_person_id == "canonical-split-manual")
    untouched = next(track for track in corrected if track.id == 2)
    assert [point["t"] for point in original.points] == [0.0, 3.0]
    assert [point["t"] for point in split.points] == [1.0, 2.0]
    assert untouched.points[0]["bbox"]["x"] == 290.0
    assert split.manual_external_player_id is None
    assert split.identity_split_partitions == {"split-crossing": "range"}
    assert original.identity_split_partitions == {"split-crossing": "remaining"}
    assert diagnostics["applied"][0]["affectedObservationCount"] == 2


@pytest.mark.parametrize(
    ("binding_time", "source_external_id", "split_external_id"),
    [
        (0.0, "roster-home-8", None),
        (2.0, None, "roster-home-8"),
    ],
)
def test_split_keeps_roster_binding_only_on_its_anchored_partition(
    binding_time,
    source_external_id,
    split_external_id,
):
    source = _split_test_track(1)
    source.canonical_person_id = "canonical-original"
    source.manual_external_player_id = "roster-home-8"
    source.manual_kind = "home-player"
    source.manual_label = "Home Eight"
    source.annotation_ids = {"roster-binding-home-8"}
    binding_point = next(point for point in source.points if point["t"] == binding_time)
    binding_point["annotationId"] = "roster-binding-home-8"
    binding_observation_id = str(binding_point["observationId"])
    roster_binding = {
        "id": "roster-binding-home-8",
        "kind": "home-player",
        "label": "Home Eight",
        "externalPlayerId": "roster-home-8",
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-original",
        "targetObservationId": binding_observation_id,
        "targetObservation": {
            "observationId": binding_observation_id,
            "frameIndex": binding_point["frameIndex"],
            "sceneTime": binding_time,
            "bbox": deepcopy(binding_point["bbox"]),
            "canonicalPersonId": "canonical-original",
        },
        "correctionKind": "canonical-roster-binding-v1",
        "rosterBindingState": "bound",
    }
    scene = {
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [roster_binding, _split_test_annotation()]
                }
            }
        },
    }

    corrected, _ = _apply_canonical_split_corrections([source], scene)

    remaining = next(
        track for track in corrected if track.canonical_person_id == "canonical-original"
    )
    split = next(
        track
        for track in corrected
        if track.canonical_person_id == "canonical-split-manual"
    )
    assert remaining.manual_external_player_id == source_external_id
    assert split.manual_external_player_id == split_external_id
    assert ("roster-binding-home-8" in remaining.annotation_ids) == (
        binding_time == 0.0
    )
    assert ("roster-binding-home-8" in split.annotation_ids) == (
        binding_time == 2.0
    )
    if binding_time == 2.0:
        assert remaining.manual_kind is None
        assert remaining.manual_label is None
        assert split.manual_kind == "home-player"
        assert split.manual_label == "Home Eight"
    else:
        assert remaining.manual_kind == "home-player"


@pytest.mark.parametrize(
    ("binding_time", "binding_moves_to_split"),
    [(0.0, False), (2.0, True)],
)
def test_split_only_allows_new_team_semantics_when_roster_anchor_stays_outside_range(
    binding_time,
    binding_moves_to_split,
):
    source = _split_test_track(1)
    source.canonical_person_id = "canonical-original"
    source.manual_external_player_id = "roster-home-8"
    source.manual_kind = "home-player"
    source.annotation_ids = {"roster-binding-home-8"}
    binding_point = next(point for point in source.points if point["t"] == binding_time)
    binding_point["annotationId"] = "roster-binding-home-8"
    binding_observation_id = str(binding_point["observationId"])
    roster_binding = {
        "id": "roster-binding-home-8",
        "kind": "home-player",
        "externalPlayerId": "roster-home-8",
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-original",
        "targetObservationId": binding_observation_id,
        "targetObservation": {
            "observationId": binding_observation_id,
            "frameIndex": binding_point["frameIndex"],
            "sceneTime": binding_time,
            "bbox": deepcopy(binding_point["bbox"]),
            "canonicalPersonId": "canonical-original",
        },
        "correctionKind": "canonical-roster-binding-v1",
        "rosterBindingState": "bound",
    }
    split = {**_split_test_annotation(), "kind": "away-player"}
    scene = {
        "duration": 4.0,
        "payload": {
            "canonicalPeople": [
                {
                    "id": "canonical-original",
                    "canonicalPersonId": "canonical-original",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": "roster-home-8",
                    "annotationIds": ["roster-binding-home-8"],
                    "observations": deepcopy(source.points),
                }
            ],
            "tracks": [],
            "videoAsset": {
                "reconstruction": {"frameAnnotations": [roster_binding, split]}
            },
        },
    }

    if binding_moves_to_split:
        with pytest.raises(
            ReconstructionError,
            match="Unbind the roster player before splitting",
        ):
            _validate_identity_corrections(scene, [roster_binding, split])
        with pytest.raises(
            IdentityCorrectionError,
            match="incompatible team or role semantics",
        ):
            _apply_canonical_split_corrections([source], scene)
        return

    _validate_identity_corrections(scene, [roster_binding, split])
    corrected, _ = _apply_canonical_split_corrections([source], scene)
    remaining = next(
        track for track in corrected if track.canonical_person_id == "canonical-original"
    )
    split_track = next(
        track
        for track in corrected
        if track.canonical_person_id == "canonical-split-manual"
    )
    assert remaining.manual_external_player_id == "roster-home-8"
    assert remaining.manual_kind == "home-player"
    assert split_track.manual_external_player_id is None
    assert split_track.manual_kind == "away-player"


def test_nested_splits_follow_canonical_lineage_before_range_ordering():
    source = TrackState(id=1, canonical_person_id="canonical-a")
    for index in range(10):
        source.points.append(
            {
                "t": float(index),
                "px": 100.0 + index,
                "py": 200.0,
                "frameIndex": index,
                "observationId": f"obs-{index}",
                "sourceTrackletId": "tracklet-0001",
                "bbox": {
                    "x": 90.0 + index,
                    "y": 150.0,
                    "width": 20.0,
                    "height": 50.0,
                },
                "confidence": 0.9,
                "annotationId": None,
            }
        )
    source.feature_sum = np.ones(12, dtype=np.float32) * 10
    source.feature_count = 10
    source.last_frame = 9
    source.last_height = 50.0
    source.source_tracklet_ids = {"tracklet-0001"}

    def split_row(
        correction_id: str,
        owner: str,
        produced: str,
        anchor: int,
        start: float,
        end: float,
    ) -> dict:
        point = source.points[anchor]
        return {
            "id": correction_id,
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonicalPersonId": owner,
            "targetObservationId": point["observationId"],
            "targetObservation": {
                "observationId": point["observationId"],
                "frameIndex": point["frameIndex"],
                "sceneTime": point["t"],
                "bbox": deepcopy(point["bbox"]),
                "canonicalPersonId": owner,
            },
            "rangeStart": start,
            "rangeEnd": end,
            "splitCanonicalPersonId": produced,
        }

    parent = split_row("split-parent", "canonical-a", "canonical-split-s1", 4, 4.0, 8.0)
    # This child sorts before the parent by its shorter rangeEnd under the old
    # implementation, even though S1 does not exist until parent is applied.
    child = split_row(
        "split-child",
        "canonical-split-s1",
        "canonical-split-s2",
        5,
        4.0,
        6.0,
    )
    scene = {
        "duration": 10.0,
        "payload": {
            "canonicalPeople": [
                {
                    "id": "canonical-a",
                    "canonicalPersonId": "canonical-a",
                    "teamId": "home",
                    "role": "player",
                    "observations": deepcopy(source.points),
                }
            ],
            "tracks": [],
            "videoAsset": {
                "reconstruction": {"frameAnnotations": [child, parent]}
            },
        },
    }

    _validate_identity_corrections(scene, [child, parent])
    corrected, diagnostics = _apply_canonical_split_corrections([source], scene)
    by_owner = {track.canonical_person_id: track for track in corrected}
    assert [point["t"] for point in by_owner["canonical-a"].points] == [
        0.0,
        1.0,
        2.0,
        3.0,
        8.0,
        9.0,
    ]
    assert [point["t"] for point in by_owner["canonical-split-s1"].points] == [6.0, 7.0]
    assert [point["t"] for point in by_owner["canonical-split-s2"].points] == [4.0, 5.0]
    assert [item["correctionId"] for item in diagnostics["applied"]] == [
        "split-parent",
        "split-child",
    ]

    with pytest.raises(ReconstructionError, match="dependent identity corrections"):
        delete_frame_person_annotation(scene, "split-parent", persist=False)

    with pytest.raises(ReconstructionError, match="parent correction is missing"):
        _validate_identity_corrections(scene, [child])


def test_split_recomputes_partition_appearance_roles_and_drops_inherited_range_evidence():
    source = _split_test_track(1)
    outside_feature = np.zeros(12, dtype=np.float32)
    outside_feature[0] = 1.0
    inside_feature = np.zeros(12, dtype=np.float32)
    inside_feature[1] = 1.0
    for point in source.points:
        point["_appearanceFeature"] = (
            inside_feature.copy()
            if 1.0 <= float(point["t"]) < 3.0
            else outside_feature.copy()
        )
        if 1.0 <= float(point["t"]) < 3.0:
            point["_reidRole"] = "goalkeeper"
            point["_reidRoleConfidence"] = 0.9
    source.feature_sum = np.ones(12, dtype=np.float32) * 100.0
    source.feature_count = 100
    source.reid_role_votes = {"player": 99.0}
    source.role = "player"
    source.identity_evidence = [{"id": "resolver-edge", "kind": "reid"}]
    source.identity_conflicts = [{"id": "resolver-review", "code": "ambiguous"}]
    scene = {
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "reconstruction": {"frameAnnotations": [_split_test_annotation()]}
            }
        },
    }

    corrected, _ = _apply_canonical_split_corrections([source], scene)

    remaining = next(
        track for track in corrected if track.canonical_person_id == "canonical-original"
    )
    split = next(
        track
        for track in corrected
        if track.canonical_person_id == "canonical-split-manual"
    )
    np.testing.assert_allclose(remaining.feature, outside_feature)
    np.testing.assert_allclose(split.feature, inside_feature)
    assert remaining.reid_role_votes == {}
    assert remaining.role is None
    assert split.reid_role_votes == {"goalkeeper": 1.8}
    assert [item["id"] for item in split.identity_evidence] == [
        "split-crossing:manual-split"
    ]
    assert split.identity_conflicts == []
    assert [item["id"] for item in remaining.identity_evidence] == [
        "resolver-edge",
        "split-crossing:manual-split",
    ]


def test_split_fails_closed_when_two_observations_match_target_bbox():
    first = _split_test_track(1)
    second = _split_test_track(2, target_observation_id="other-id")
    before = [[deepcopy(point) for point in track.points] for track in (first, second)]
    scene = {
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "reconstruction": {"frameAnnotations": [_split_test_annotation()]}
            }
        },
    }

    with pytest.raises(IdentityCorrectionError, match="ambiguous") as caught:
        _apply_canonical_split_corrections([first, second], scene)

    assert caught.value.diagnostic["reason"] == "multiple-target-observation-matches"
    assert len(caught.value.diagnostic["candidates"]) == 2
    assert first.points == before[0]
    assert second.points == before[1]


def test_deleting_split_is_deterministic_undo():
    annotation = _split_test_annotation()
    scene = {
        "duration": 4.0,
        "payload": {
            "videoAsset": {"reconstruction": {"frameAnnotations": [annotation]}}
        },
    }
    original = _split_test_track(1)
    corrected, _ = _apply_canonical_split_corrections([deepcopy(original)], scene)
    assert len(corrected) == 2

    deleted = delete_frame_person_annotation(scene, "split-crossing", persist=False)
    restored, diagnostics = _apply_canonical_split_corrections([deepcopy(original)], scene)

    assert deleted == annotation
    assert len(restored) == 1
    assert [point["t"] for point in restored[0].points] == [0.0, 1.0, 2.0, 3.0]
    assert diagnostics == {"appliedCount": 0, "applied": []}


def test_scene_merge_cannot_cross_manual_split_barrier():
    scene = {
        "duration": 4.0,
        "payload": {
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {"id": "merge-source", "action": "merge", "mergeTargetId": "merge-target"},
                        {"id": "merge-target", "action": "confirm"},
                    ]
                }
            },
        },
    }
    tracks = [
        {
            "id": "source",
            "annotationIds": ["merge-source"],
            "identitySplitPartitions": {"split-crossing": "range"},
            "keyframes": [{"t": 2.0, "x": 1.0, "z": 1.0, "observed": True}],
        },
        {
            "id": "target",
            "annotationIds": ["merge-target"],
            "identitySplitPartitions": {"split-crossing": "remaining"},
            "keyframes": [{"t": 0.0, "x": 0.0, "z": 0.0, "observed": True}],
        },
    ]

    corrected = _apply_scene_track_identity_corrections(tracks, scene)

    assert [track["id"] for track in corrected] == ["source", "target"]


def test_raw_identity_merge_fails_closed_for_distinct_confirmed_roster_players():
    target_detection = Detection(
        100, 250, 20, 50, 0.9, np.ones(12, dtype=np.float32)
    )
    target_detection.annotation_id = "target-confirm"
    target_detection.annotation_kind = "home-player"
    target_detection.external_player_id = "roster-home-8"
    source_detection = Detection(
        300, 250, 20, 50, 0.9, np.ones(12, dtype=np.float32)
    )
    source_detection.annotation_id = "source-merge"
    source_detection.annotation_kind = "home-player"
    source_detection.external_player_id = "roster-home-10"
    target = TrackState(id=1)
    target.append(target_detection, 0, 0.0)
    source = TrackState(id=2)
    source.append(source_detection, 1, 0.1)
    scene = {
        "payload": {
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "target-confirm",
                            "action": "confirm",
                            "externalPlayerId": "roster-home-8",
                        },
                        {
                            "id": "source-merge",
                            "action": "merge",
                            "mergeTargetId": "target-confirm",
                            "externalPlayerId": "roster-home-10",
                        },
                    ]
                }
            },
        }
    }

    with pytest.raises(IdentityCorrectionError, match="cannot merge confirmed roster") as caught:
        _apply_track_identity_corrections([target, source], scene)

    assert caught.value.diagnostic["status"] == "conflict"
    assert (
        caught.value.diagnostic["reason"]
        == "conflicting-confirmed-external-player-ids"
    )


def test_merge_validation_uses_source_annotation_roster_id_without_saved_subject():
    scene = {
        "duration": 1.0,
        "payload": {
            "tracks": [
                {
                    "id": "target-track",
                    "canonicalPersonId": "canonical-target",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": "roster-home-8",
                }
            ],
            "videoAsset": {"reconstruction": {"frameAnnotations": []}},
        },
    }
    merge = {
        "id": "source-merge",
        "action": "merge",
        "scope": "identity",
        "mergeTargetId": "canonical-target",
        # No canonical/source subject is persisted for this new observation;
        # its own confirmed roster id must still make save-time QA fail.
        "externalPlayerId": "roster-home-10",
    }

    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _validate_identity_corrections(scene, [merge])


def test_merge_validation_uses_annotation_roster_id_when_saved_subject_has_none():
    scene = {
        "duration": 1.0,
        "payload": {
            "tracks": [
                {
                    "id": "source-track",
                    "canonicalPersonId": "canonical-source",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": None,
                },
                {
                    "id": "target-track",
                    "canonicalPersonId": "canonical-target",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": "roster-home-8",
                },
            ],
            "videoAsset": {"reconstruction": {"frameAnnotations": []}},
        },
    }
    merge = {
        "id": "source-merge",
        "action": "merge",
        "scope": "identity",
        "sourceTrackId": "source-track",
        "canonicalPersonId": "canonical-source",
        "mergeTargetId": "canonical-target",
        "externalPlayerId": "roster-home-10",
    }

    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _validate_identity_corrections(scene, [merge])


def test_merge_validation_uses_target_subject_id_when_target_annotation_has_none():
    scene = {
        "duration": 1.0,
        "payload": {
            "tracks": [
                {
                    "id": "source-track",
                    "canonicalPersonId": "canonical-source",
                    "externalPlayerId": "roster-home-10",
                },
                {
                    "id": "target-track",
                    "canonicalPersonId": "canonical-target",
                    "externalPlayerId": "roster-home-8",
                },
            ],
            "videoAsset": {"reconstruction": {"frameAnnotations": []}},
        },
    }
    target = {
        "id": "target-confirm",
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-target",
        "externalPlayerId": None,
    }
    merge = {
        "id": "source-merge",
        "action": "merge",
        "scope": "identity",
        "canonicalPersonId": "canonical-source",
        "mergeTargetId": "target-confirm",
    }

    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _validate_identity_corrections(scene, [target, merge])


def test_merge_validation_rejects_stale_annotation_subject_roster_disagreement():
    scene = {
        "duration": 1.0,
        "payload": {
            "tracks": [
                {
                    "id": "source-track",
                    "canonicalPersonId": "canonical-source",
                    "externalPlayerId": "roster-home-8",
                },
                {
                    "id": "target-track",
                    "canonicalPersonId": "canonical-target",
                    "externalPlayerId": "roster-home-8",
                },
            ],
            "videoAsset": {"reconstruction": {"frameAnnotations": []}},
        },
    }
    merge = {
        "id": "source-merge",
        "action": "merge",
        "scope": "identity",
        "canonicalPersonId": "canonical-source",
        "mergeTargetId": "canonical-target",
        "externalPlayerId": "roster-home-10",
    }

    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _validate_identity_corrections(scene, [merge])


def test_published_identity_merge_fails_closed_for_distinct_roster_players():
    scene = {
        "duration": 1.0,
        "payload": {
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "source-merge",
                            "action": "merge",
                            "mergeTargetId": "target-confirm",
                        },
                        {"id": "target-confirm", "action": "confirm"},
                    ]
                }
            },
        },
    }
    tracks = [
        {
            "id": "source",
            "annotationIds": ["source-merge"],
            "externalPlayerId": "roster-home-10",
            "keyframes": [{"t": 0.1, "x": 1.0, "z": 0.0, "observed": True}],
        },
        {
            "id": "target",
            "annotationIds": ["target-confirm"],
            "externalPlayerId": "roster-home-8",
            "keyframes": [{"t": 0.0, "x": 0.0, "z": 0.0, "observed": True}],
        },
    ]

    with pytest.raises(IdentityCorrectionError, match="cannot merge confirmed roster") as caught:
        _apply_scene_track_identity_corrections(tracks, scene)

    assert (
        caught.value.diagnostic["reason"]
        == "conflicting-confirmed-external-player-ids"
    )


def test_identity_scoped_exclusion_removes_only_matching_raw_track():
    def raw_track(track_id: int, x: float) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
        )

    previous_keyframes = [
        {"t": index * 0.2, "x": index * 0.1, "z": 2.0, "confidence": 0.9, "observed": True}
        for index in range(4)
    ]
    scene = {
        "payload": {
            "tracks": [{"id": "auto-home-01", "keyframes": previous_keyframes}],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    corrected = _apply_track_identity_corrections(
        [raw_track(1, 0.0), raw_track(2, 20.0)],
        scene,
    )

    assert [track.id for track in corrected] == [2]


def test_identity_exclusion_prefers_exact_annotation_anchor_after_calibration_shift():
    def raw_track(track_id: int, x: float, annotation_id: str | None = None) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
            annotation_ids={annotation_id} if annotation_id else set(),
        )

    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {"t": index * 0.2, "x": index * 0.1, "z": 2.0, "observed": True}
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    corrected = _apply_track_identity_corrections(
        [raw_track(1, 20.0, "phantom"), raw_track(2, 0.0)],
        scene,
    )

    assert [track.id for track in corrected] == [2]


def test_identity_exclusion_rejects_ambiguous_geometry_without_exact_anchor():
    def raw_track(track_id: int, x: float) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                    "positionUncertaintyMetres": 0.5,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
        )

    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {
                            "t": index * 0.2,
                            "x": index * 0.1,
                            "z": 2.0,
                            "observed": True,
                            "positionUncertaintyMetres": 0.5,
                        }
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    with pytest.raises(IdentityCorrectionError, match="ambiguous") as caught:
        _apply_track_identity_corrections(
            [raw_track(1, 0.2), raw_track(2, 0.3)],
            scene,
        )

    diagnostic = caught.value.diagnostic
    assert diagnostic["correctionId"] == "phantom"
    assert diagnostic["action"] == "exclude"
    assert diagnostic["status"] == "ambiguous"
    assert diagnostic["reason"] == "nearby-trajectories"
    assert [item["rawTrackId"] for item in diagnostic["candidates"]] == [1, 2]
    assert all("medianDistanceMetres" in item for item in diagnostic["candidates"])


def test_identity_exclusion_reports_unresolved_when_remap_evidence_is_missing():
    raw = TrackState(
        id=7,
        points=[
            {
                "t": index * 0.2,
                "px": 100.0 + index,
                "py": 200.0,
                "confidence": 0.9,
            }
            for index in range(4)
        ],
        feature_sum=np.zeros(12, dtype=np.float32),
        feature_count=1,
        last_frame=3,
        last_height=40.0,
    )
    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {
                            "t": index * 0.2,
                            "x": index * 0.1,
                            "z": 2.0,
                            "observed": True,
                        }
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    with pytest.raises(IdentityCorrectionError, match="could not resolve") as caught:
        _apply_track_identity_corrections([raw], scene)

    diagnostic = caught.value.diagnostic
    assert diagnostic["correctionId"] == "phantom"
    assert diagnostic["status"] == "unresolved"
    assert diagnostic["reason"] == "insufficient-observation-overlap"
    assert diagnostic["candidates"] == [{"rawTrackId": 7}]


def test_observation_scoped_exclusion_does_not_remove_complete_raw_track():
    track = TrackState(
        id=1,
        points=[
            {"t": 0.0, "px": 100.0, "py": 200.0, "pitchX": 0.0, "pitchZ": 2.0, "confidence": 0.9}
        ],
        feature_sum=np.zeros(12, dtype=np.float32),
        feature_count=1,
        last_frame=0,
        last_height=40.0,
    )
    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {"t": 0.0, "x": 0.0, "z": 2.0, "confidence": 0.9, "observed": True}
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "one-frame",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "observation",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    assert _apply_track_identity_corrections([track], scene) == [track]


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
                    "pitchCalibrationOverride": {
                        "preset": "goal-area-right",
                        "imageToPitch": [[1.0, 0.0, -20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "anchors": [{"pitch": {"x": 47.0, "z": -9.16}}],
                    },
                },
            },
            "tracks": [{"keyframes": [{"x": 12.5, "z": 3.0}]}],
            "ball": {"keyframes": [{"x": -4.5, "z": 2.0}]},
        },
    }
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    set_scene_pitch_side(scene, "left")
    set_scene_pitch_side(scene, "left")

    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    assert scene["payload"]["tracks"][0]["keyframes"][0]["x"] == 12.5
    assert scene["payload"]["ball"]["keyframes"][0]["x"] == -4.5
    assert reconstruction["pitchCalibration"]["rectangle"] == "goal-area-right"
    assert reconstruction["pitchCalibration"]["imageToPitch"][0] == [1.0, 0.0, -20.0]
    assert reconstruction["pitchCalibrationOverride"]["preset"] == "goal-area-right"
    assert reconstruction["pitchCalibrationOverride"]["anchors"][0]["pitch"]["x"] == 47.0
    assert reconstruction["pitchOrientation"]["attackingGoal"] == "left"
    assert reconstruction["pitchOrientation"]["attackingGoalSource"] == "manual"


def _track(track_id: int, x: float, samples: int = 10) -> TrackState:
    feature = np.zeros(12, dtype=np.float32)
    return TrackState(
        id=track_id,
        points=[{"t": index * 0.2, "px": x, "py": 220.0, "confidence": 0.8} for index in range(samples)],
        feature_sum=feature,
        feature_count=1,
        last_frame=samples - 1,
        last_height=40.0,
    )


def test_goalkeeper_candidate_survives_third_kit_cluster():
    tracks = [
        _track(1, 650),
        _track(2, 700),
        _track(3, 760),
        _track(4, 620),
        _track(5, 900),
        _track(6, 480),
    ]
    mapping = {1: "home", 2: "away", 3: "away", 4: "home"}

    result = _include_goalkeeper_candidates(tracks, mapping, 960)

    assert result[5] == "away"
    assert tracks[4].role == "goalkeeper"
    assert 6 not in result


def test_ball_track_prefers_strong_evidence_over_long_false_positive():
    frames = []
    for index in range(6):
        detections = [{"x": 100.0, "y": 400.0, "confidence": 0.06}]
        if index < 4:
            detections.append({"x": 500.0 + index * 2, "y": 280.0, "confidence": 0.4})
        frames.append((detections, index * 0.2))
    scene = {"payload": {"pitch": {"length": 105, "width": 68}}}

    keyframes = _ball_keyframes(frames, (960, 540), scene)

    assert len(keyframes) == 4
    assert keyframes[0]["x"] > 0


def test_frame_analysis_interpolates_track_at_requested_time():
    keyframes = [
        {"t": 0.0, "x": 10.0, "z": -4.0, "confidence": 0.8},
        {"t": 1.0, "x": 20.0, "z": 6.0, "confidence": 0.6},
    ]

    position = _interpolate_scene_keyframes(keyframes, 0.25)

    assert position is not None
    assert position["x"] == 12.5
    assert position["z"] == -1.5
    assert position["confidence"] == 0.75


def test_queue_reconstruction_preserves_last_good_result_and_records_previous(monkeypatch):
    scene = {
        "id": "shot-1",
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "status": "ready",
                    "completedAt": "yesterday",
                    "pitchCalibration": {"status": "ready"},
                    "pitchCalibrationOverride": {
                        "method": "manual-pitch-anchors",
                        "imageToPitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    },
                },
            },
            "tracks": [{"id": "old-track"}],
            "ball": {"keyframes": [{"t": 1.0}]},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("one", 0.0), ("two", 0.2)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    queued = queue_reconstruction(scene)

    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    assert queued["payload"]["tracks"] == [{"id": "old-track"}]
    assert queued["payload"]["ball"] == {"keyframes": [{"t": 1.0}]}
    assert reconstruction["status"] == "queued"
    assert reconstruction["frameCount"] == 2
    assert reconstruction["progress"]["phase"] == "preparing"
    assert reconstruction["progress"]["overallPercent"] == 0
    assert [item["status"] for item in reconstruction["progress"]["phases"]] == [
        "current",
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert reconstruction["previousResult"] == {
        "completedAt": "yesterday",
        "trackCount": 1,
        "ballSamples": 1,
        "calibrationStatus": "ready",
    }
    assert reconstruction["pitchCalibrationOverride"]["method"] == "manual-pitch-anchors"


def test_reconstruction_progress_exposes_completed_current_and_pending_phases(monkeypatch):
    scene = {
        "id": "progress-scene",
        "payload": {"videoAsset": {"reconstruction": {"status": "processing"}}},
    }
    persisted = []
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: persisted.append(value))

    payload = ReconstructionProgress(scene).update(
        "calibration",
        2,
        "Calibrating the pitch",
        "PnLCalib · 4/8 frames.",
        4,
        62,
        completed=4,
        total=8,
    )

    assert payload["phasePercent"] == 50
    assert payload["overallPercent"] == 33
    assert payload["completed"] == 4
    assert payload["total"] == 8
    assert [item["status"] for item in payload["phases"]] == [
        "completed",
        "current",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert persisted


def test_queue_reconstruction_uses_requested_model(monkeypatch):
    scene = {
        "id": "shot-model",
        "duration": 2.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {"status": "ready", "model": "yolo26n.pt"},
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("one", 0.0)])
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    queued = queue_reconstruction(scene, "yolo26m.pt")

    assert queued["payload"]["videoAsset"]["reconstruction"]["model"] == "yolo26m.pt"


def test_queue_reconstruction_persists_requested_ball_backend_and_input(monkeypatch):
    scene = {
        "id": "shot-ball-backend",
        "duration": 2.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "status": "ready",
                    "ballBackend": "generic-ultralytics",
                    "ballDetectionInput": {
                        "backend": "generic-ultralytics",
                        "analysisFrameRate": 10.0,
                    },
                },
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    requested_input = {
        "schemaVersion": 1,
        "backend": "wasb-service",
        "checkpoint": {"name": "wasb-soccer-best.pth.tar", "size": 1234},
        "analysisFrameRate": 25.0,
        "failurePolicy": "fallback",
    }
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("one", 0.0)])
    monkeypatch.setattr(
        "app.reconstruction._ball_detection_input",
        lambda backend: {**requested_input, "backend": backend},
    )
    monkeypatch.setattr("app.reconstruction.scene_store.get", lambda _: None)
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)

    queued = queue_reconstruction(scene, ball_backend="wasb-service")
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]

    assert reconstruction["ballBackend"] == "wasb-service"
    assert reconstruction["ballDetectionInput"] == requested_input
    assert reconstruction["inputFingerprint"] == reconstruction_input_fingerprint(queued)


def test_ball_backend_and_detector_config_are_reconstruction_fingerprint_inputs():
    scene = {
        "id": "shot-ball-fingerprint",
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "sourceStart": 1.0,
                "sourceEnd": 3.0,
                "reconstruction": {
                    "model": "yolo26m.pt",
                    "ballBackend": "dedicated-ultralytics",
                    "ballDetectionInput": {
                        "schemaVersion": 1,
                        "backend": "dedicated-ultralytics",
                        "checkpoint": {"name": "football-ball-detection.pt", "size": 1234},
                        "analysisFrameRate": 25.0,
                        "inferenceBatchSize": 8,
                    },
                    # Worker output is deliberately not an immutable input.
                    "ballDetection": {"candidateCount": 10},
                },
            }
        },
    }
    baseline = reconstruction_input_fingerprint(scene)

    changed_backend = deepcopy(scene)
    changed_backend["payload"]["videoAsset"]["reconstruction"][
        "ballBackend"
    ] = "generic-ultralytics"
    assert reconstruction_input_fingerprint(changed_backend) != baseline

    changed_config = deepcopy(scene)
    changed_config["payload"]["videoAsset"]["reconstruction"][
        "ballDetectionInput"
    ]["inferenceBatchSize"] = 4
    assert reconstruction_input_fingerprint(changed_config) != baseline

    changed_runtime_output = deepcopy(scene)
    changed_runtime_output["payload"]["videoAsset"]["reconstruction"][
        "ballDetection"
    ]["candidateCount"] = 999
    assert reconstruction_input_fingerprint(changed_runtime_output) == baseline


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
    monkeypatch.setattr("app.reconstruction.preview_scene_pitch_calibration", lambda *args: draft)
    monkeypatch.setattr(
        "app.reconstruction._frame_context",
        lambda *args: (3, 0.6, np.zeros((540, 960, 3), dtype=np.uint8), np.eye(3)),
    )
    monkeypatch.setattr("app.reconstruction.scene_store.put", lambda value: value)
    monkeypatch.setattr("app.reconstruction.queue_reconstruction", lambda value: value)

    applied = apply_scene_pitch_calibration(scene, 0.6, "center-circle", anchors)

    override = applied["payload"]["videoAsset"]["reconstruction"]["pitchCalibrationOverride"]
    assert override["method"] == "manual-pitch-anchors"
    assert override["alignmentError"] == 3.2
    assert override["imageToPitch"] == draft["imageToPitch"]
