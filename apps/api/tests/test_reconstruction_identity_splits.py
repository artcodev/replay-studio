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
from app.scene_document import reconstruction_input_fingerprint


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
    monkeypatch.setattr("app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 2.0)])

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
        }
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
        delete_frame_person_annotation(scene, "split-parent")

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

    deleted = delete_frame_person_annotation(scene, "split-crossing")
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
