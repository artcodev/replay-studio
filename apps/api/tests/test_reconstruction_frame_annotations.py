from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pitch_anchor_calibration import calibration_from_anchors
from app.pitch_calibration_contract import PitchCalibration
from app.pitch_calibration_orientation import canonicalize_penalty_side
from app.pitch_geometry import projected_pitch_markings
from app.reconstruction_pitch_side_command import set_scene_pitch_side
from app.reconstruction_errors import IdentityCorrectionError, ReconstructionError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation
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
from app.scene_document import reconstruction_input_fingerprint


def test_frame_annotations_can_add_and_ignore_detections():
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    feature = np.zeros(12, dtype=np.float32)
    automatic = Detection(300, 250, 20, 50, 0.7, feature)
    ignored = {
        "id": "ignored",
        "kind": "ignore",
        "action": "exclude",
        "scope": "observation",
        "bbox": {"x": 290, "y": 200, "width": 20, "height": 50},
    }
    manual = {
        "id": "manual",
        "kind": "home-player",
        "action": "confirm",
        "scope": "observation",
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
        append_track_observation(track, detection, 0, 0.0)
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
    annotation = {
        "id": "frame-zero",
        "frameIndex": 0,
        "action": "confirm",
        "scope": "observation",
    }
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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)

    annotation = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": None,
            "scene_time": 0.03,
            "bbox": {"x": 140, "y": 135, "width": 18, "height": 38},
            "kind": "home-player",
            "label": "Player A",
            "external_player_id": None,
            "action": "confirm",
            "scope": "identity",
        },
    )

    assert annotation["sceneTime"] == 0.0
    assert annotation["sourceTime"] == 10.0
    assert annotation["frameIndex"] == 1
    assert annotation["scope"] == "identity"
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
                "action": "confirm",
                "scope": "identity",
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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)

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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)

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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)
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


def test_identity_annotation_requires_explicit_action(monkeypatch, tmp_path):
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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)

    with pytest.raises(ReconstructionError, match="action is required"):
        upsert_frame_person_annotation(
            scene,
            {
                "scene_time": 0.0,
                "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
                "kind": "ignore",
                "scope": "observation",
                "source_track_id": None,
            },
        )


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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)])
    monkeypatch.setattr("app.reconstruction_identity_annotation_commit.scenes.put", lambda value: value)
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
