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
    append_track_observation(target, target_detection, 0, 0.0)
    source = TrackState(id=2)
    append_track_observation(source, source_detection, 1, 0.1)
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


def test_raw_identity_merge_fails_closed_for_distinct_confirmed_roster_players():
    target_detection = Detection(
        100, 250, 20, 50, 0.9, np.ones(12, dtype=np.float32)
    )
    target_detection.annotation_id = "target-confirm"
    target_detection.annotation_kind = "home-player"
    target_detection.external_player_id = "roster-home-8"
    target_detection.roster_binding_state = "bound"
    target_detection.roster_binding_annotation_ids = {"target-confirm"}
    source_detection = Detection(
        300, 250, 20, 50, 0.9, np.ones(12, dtype=np.float32)
    )
    source_detection.annotation_id = "source-merge"
    source_detection.annotation_kind = "home-player"
    source_detection.external_player_id = "roster-home-10"
    source_detection.roster_binding_state = "bound"
    source_detection.roster_binding_annotation_ids = {"source-binding"}
    target = TrackState(id=1)
    append_track_observation(target, target_detection, 0, 0.0)
    source = TrackState(id=2)
    append_track_observation(source, source_detection, 1, 0.1)
    scene = {
        "payload": {
            "tracks": [],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "target-confirm",
                            "action": "confirm",
                            "scope": "identity",
                            "kind": "home-player",
                            "externalPlayerId": "roster-home-8",
                            "correctionKind": "canonical-roster-binding-v1",
                            "rosterBindingState": "bound",
                        },
                        {
                            "id": "source-binding",
                            "action": "confirm",
                            "scope": "identity",
                            "kind": "home-player",
                            "externalPlayerId": "roster-home-10",
                            "correctionKind": "canonical-roster-binding-v1",
                            "rosterBindingState": "bound",
                        },
                        {
                            "id": "source-merge",
                            "action": "merge",
                            "scope": "identity",
                            "kind": "home-player",
                            "mergeTargetId": "target-confirm",
                            "externalPlayerId": None,
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


def test_merge_validation_allows_unbound_new_observation_into_bound_identity():
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
    }

    _validate_identity_corrections(scene, [merge])


def test_merge_validation_allows_unbound_saved_subject_into_bound_identity():
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
    }

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


def test_merge_validation_allows_identities_with_same_dedicated_roster_binding():
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
    }

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
