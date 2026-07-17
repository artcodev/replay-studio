from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.store as store_module
from app.database import Base
from app.main import app
from app.reconstruction import (
    ReconstructionError,
    StaleReconstructionRun,
    TrackState,
    _apply_canonical_split_corrections,
    _apply_person_annotations,
    _assign_persistent_canonical_person_ids,
    _canonical_people_documents,
    _frame_annotations,
    _merge_raw_track_states,
    _resolve_canonical_track_states,
    clear_canonical_roster_binding,
    delete_frame_person_annotation,
    queue_reconstruction,
    set_canonical_roster_binding,
    upsert_frame_person_annotation,
)
from app.store import SceneStore, reconstruction_input_fingerprint


def _scene(*, status: str = "ready") -> dict:
    return {
        "id": "roster-binding-scene",
        "title": "Roster binding",
        "version": 1,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "selectedSegmentId": "segment-1",
                "sourceStart": 20.0,
                "sourceEnd": 24.0,
                "analysisFps": 10.0,
                "reconstruction": {
                    "status": status,
                    "runId": "run-ready",
                    "runRevision": 3,
                    "model": "yolo26m.pt",
                    "frameAnnotations": [],
                },
            },
            "matchBinding": {
                "source": "thesportsdb",
                "eventId": "event-1",
                "teams": {
                    "home": {"id": "team-home", "name": "Home"},
                    "away": {"id": "team-away", "name": "Away"},
                },
                "players": [
                    {
                        "id": "player-home-8",
                        "name": "Home Eight",
                        "team_id": "team-home",
                        "number": "8",
                    },
                    {
                        "id": "player-away-10",
                        "name": "Away Ten",
                        "team_id": "team-away",
                        "number": "10",
                    },
                ],
            },
            "canonicalPeople": [
                {
                    "id": "canonical-offscreen",
                    "canonicalPersonId": "canonical-offscreen",
                    "displayName": "Home person 02",
                    "identityStatus": "provisional",
                    "identityConfidence": 0.72,
                    "identitySource": "tracker+trajectory",
                    "teamId": "home",
                    "role": "player",
                    "externalPlayerId": None,
                    "annotationIds": [],
                    # Intentionally no renderTrackId: this identity is currently
                    # off screen / rejected by metric projection.
                    "renderTrackId": None,
                    "observations": [
                        {
                            "id": "obs-low",
                            "observationId": "obs-low",
                            "frameIndex": 201,
                            "sceneTime": 0.0,
                            "sourceTime": 20.0,
                            "bbox": {"x": 20, "y": 30, "width": 18, "height": 42},
                            "confidence": 0.45,
                            "metricStatus": "unprojected",
                            "sourceTrackletId": "tracklet-1",
                        },
                        {
                            "id": "obs-best",
                            "observationId": "obs-best",
                            "frameIndex": 207,
                            "sceneTime": 0.6,
                            "sourceTime": 20.6,
                            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
                            "confidence": 0.94,
                            "metricStatus": "accepted",
                            "sourceTrackletId": "tracklet-1",
                        },
                    ],
                    "evidence": [],
                    "rosterCandidates": [],
                    "conflicts": [],
                }
            ],
            "tracks": [],
        },
    }


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


@pytest.fixture
def isolated_store(monkeypatch) -> SceneStore:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(store_module, "SessionLocal", session_local)
    return SceneStore()


def test_offscreen_roster_binding_uses_saved_detector_observation_and_survives_worker_outage(
    monkeypatch,
):
    scene = _scene()
    before = reconstruction_input_fingerprint(scene)
    monkeypatch.setattr(
        "app.reconstruction._frame_paths",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not read the current frame")),
    )

    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )

    assert annotation["action"] == "confirm"
    assert annotation["scope"] == "identity"
    assert annotation["correctionKind"] == "canonical-roster-binding-v1"
    assert annotation["targetObservationId"] == "obs-best"
    assert annotation["frameIndex"] == 207
    assert annotation["externalPlayerId"] == "player-home-8"
    assert scene["payload"]["canonicalPeople"][0]["externalPlayerId"] == "player-home-8"
    assert annotation["id"] in scene["payload"]["canonicalPeople"][0]["annotationIds"]
    assert reconstruction_input_fingerprint(scene) != before

    # If the detector/ReID/OCR services are unavailable on the rebuild, the
    # correction still reconstructs one exact image-space anchor.
    detections = _apply_person_annotations(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [],
        [annotation],
    )
    assert len(detections) == 1
    assert detections[0].external_player_id == "player-home-8"
    assert detections[0].annotation_id == annotation["id"]

    rebuilt_track = TrackState(id=77)
    rebuilt_track.append(detections[0], annotation["frameIndex"], annotation["sceneTime"])
    _assign_persistent_canonical_person_ids([rebuilt_track], scene, {77: "home"})
    assert rebuilt_track.canonical_person_id == "canonical-offscreen"


def test_ambiguous_roster_anchor_ownership_fails_closed() -> None:
    scene = _scene()
    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    duplicate = deepcopy(scene["payload"]["canonicalPeople"][0])
    duplicate["id"] = "canonical-duplicate"
    duplicate["canonicalPersonId"] = "canonical-duplicate"
    duplicate["externalPlayerId"] = None
    duplicate["annotationIds"] = [annotation["id"]]
    scene["payload"]["canonicalPeople"].append(duplicate)

    with pytest.raises(ReconstructionError, match="owned by multiple canonical people"):
        set_canonical_roster_binding(
            scene,
            "canonical-duplicate",
            None,
            persist=False,
        )


def test_single_explicit_owner_preserves_canonical_id_without_long_overlap() -> None:
    scene = _scene()
    track = TrackState(
        id=7,
        points=[
            {
                "t": 0.6,
                "frameIndex": 207,
                "observationId": "new-detector-row",
                "sourceTrackletId": "tracklet-0007",
                "bbox": {"x": 121, "y": 80, "width": 24, "height": 58},
                "confidence": 0.9,
                "annotationId": "confirm-owner",
            }
        ],
        feature_sum=np.ones(8, dtype=np.float32),
        feature_count=1,
        last_frame=207,
        last_height=58,
        annotation_ids={"confirm-owner"},
        manual_identity_owner_ids={"canonical-offscreen"},
    )

    _assign_persistent_canonical_person_ids([track], scene, {7: "home"})

    assert track.canonical_person_id == "canonical-offscreen"

    duplicate = deepcopy(track)
    duplicate.id = 8
    duplicate.canonical_person_id = None
    with pytest.raises(ReconstructionError, match="multiple unresolved tracks"):
        _assign_persistent_canonical_person_ids(
            [track, duplicate],
            scene,
            {7: "home", 8: "home"},
        )


def test_roster_binding_rejects_unknown_player_and_wrong_team():
    scene = _scene()
    with pytest.raises(ReconstructionError, match="not present"):
        set_canonical_roster_binding(
            scene,
            "canonical-offscreen",
            "player-missing",
            persist=False,
        )

    scene = _scene()
    with pytest.raises(ReconstructionError, match="other team"):
        set_canonical_roster_binding(
            scene,
            "canonical-offscreen",
            "player-away-10",
            persist=False,
        )


def test_roster_unbind_replaces_the_same_durable_correction_deterministically():
    scene = _scene()
    bound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    unbound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None,
        persist=False,
    )
    repeated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None,
        persist=False,
    )

    annotations = scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    person = scene["payload"]["canonicalPeople"][0]
    assert len(annotations) == 1
    assert bound["id"] == unbound["id"] == repeated["id"]
    assert unbound["externalPlayerId"] is None
    assert unbound["rosterBindingState"] == "unbound"
    assert repeated == unbound
    assert person["externalPlayerId"] is None
    assert person["displayName"] == "Home person 02"
    assert person["identityStatus"] == "provisional"
    assert person["identityConfidence"] == 0.72
    assert person["identitySource"] == "tracker+trajectory"


def test_roster_unbind_tombstone_preserves_id_without_positive_manual_evidence():
    scene = _scene()
    set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None,
        persist=False,
    )

    detections = _apply_person_annotations(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [],
        [annotation],
    )
    assert len(detections) == 1
    assert detections[0].external_player_id is None
    assert detections[0].annotation_is_identity_evidence is False

    rebuilt = TrackState(id=91)
    rebuilt.append(
        detections[0], annotation["frameIndex"], annotation["sceneTime"]
    )
    assert rebuilt.annotation_ids == {annotation["id"]}
    assert rebuilt.identity_tombstone_ids == {annotation["id"]}
    assert rebuilt.positive_annotation_ids == set()
    _assign_persistent_canonical_person_ids([rebuilt], scene, {91: "home"})
    assert rebuilt.canonical_person_id == "canonical-offscreen"

    resolved, resolver_diagnostics = _resolve_canonical_track_states(
        [rebuilt],
        {91: "home"},
    )
    assert resolved == [rebuilt]
    assert rebuilt.identity_status == "provisional"
    assert rebuilt.identity_confidence == 0.0
    assert resolver_diagnostics["resolvedIdentityCount"] == 0
    assert resolver_diagnostics["provisionalIdentityCount"] == 1

    # The canonical document may retain the pre-binding confidence baseline,
    # but the resolver itself must never fabricate manual 1.0 from a tombstone.
    rebuilt.identity_confidence = 0.72
    people, diagnostics = _canonical_people_documents(
        [rebuilt],
        {91: "home"},
        [],
        scene,
    )
    assert len(people) == 1
    person = people[0]
    assert person["externalPlayerId"] is None
    assert person["identityStatus"] == "provisional"
    assert person["identityConfidence"] == 0.72
    assert person["identitySource"] == "tracker+trajectory"
    assert person["annotationIds"] == [annotation["id"]]
    assert all(item.get("kind") != "manual" for item in person["evidence"])
    assert diagnostics["manualDecisionCount"] == 0


@pytest.mark.parametrize("dedicated_state", ["bound", "unbound"])
@pytest.mark.parametrize("dedicated_first", [False, True])
def test_dedicated_roster_decision_supersedes_legacy_confirm_independent_of_time(
    dedicated_state: str,
    dedicated_first: bool,
) -> None:
    scene = _scene()
    dedicated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    if dedicated_state == "unbound":
        dedicated = set_canonical_roster_binding(
            scene,
            "canonical-offscreen",
            None,
            persist=False,
        )
    legacy = {
        "id": "legacy-generic-roster-confirm",
        "sceneTime": 1.2,
        "frameIndex": 212,
        "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-player",
        "label": "Legacy Eight",
        "externalPlayerId": "legacy-player-8",
        "action": "confirm",
        "scope": "identity",
    }
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    dedicated_detection = _apply_person_annotations(image, [], [dedicated])[0]
    legacy_detection = _apply_person_annotations(image, [], [legacy])[0]
    ordered = (
        [(dedicated_detection, 0.0), (legacy_detection, 1.2)]
        if dedicated_first
        else [(legacy_detection, 0.0), (dedicated_detection, 1.2)]
    )
    track = TrackState(id=92)
    for frame_index, (detection, time) in enumerate(ordered):
        track.append(detection, frame_index, time)

    assert track.roster_binding_state == dedicated_state
    assert track.manual_external_player_id == (
        "player-home-8" if dedicated_state == "bound" else None
    )
    assert track.manual_label == (
        "Home Eight" if dedicated_state == "bound" else "Home person 02"
    )
    people, _ = _canonical_people_documents(
        [track],
        {track.id: "home"},
        [],
        scene,
    )
    assert people[0]["displayName"] == track.manual_label

    same_frame = _apply_person_annotations(
        image,
        [],
        [dedicated, legacy] if dedicated_first else [legacy, dedicated],
    )[0]
    assert same_frame.annotation_ids == {dedicated["id"], legacy["id"]}
    assert same_frame.roster_binding_state == dedicated_state
    assert same_frame.external_player_id == (
        "player-home-8" if dedicated_state == "bound" else None
    )


def test_later_authored_role_label_edit_does_not_change_dedicated_roster_id() -> None:
    scene = _scene()
    dedicated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    generic = {
        "id": "newer-role-edit",
        "frameIndex": 212,
        "sceneTime": 1.2,
        "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-goalkeeper",
        "label": "Reviewed goalkeeper",
        "externalPlayerId": None,
        "action": "confirm",
        "scope": "identity",
        "updatedAt": "9999-01-01T00:00:00+00:00",
    }
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    track = TrackState(id=94)
    track.append(_apply_person_annotations(image, [], [dedicated])[0], 0, 0.0)
    track.append(_apply_person_annotations(image, [], [generic])[0], 1, 1.2)

    assert track.manual_external_player_id == "player-home-8"
    assert track.manual_kind == "home-goalkeeper"
    assert track.manual_label == "Reviewed goalkeeper"


def test_merge_preserves_dedicated_roster_precedence_and_rejects_two_decisions() -> None:
    dedicated = TrackState(
        id=1,
        roster_binding_state="bound",
        roster_binding_annotation_ids={"binding-a"},
        manual_external_player_id="player-home-8",
    )
    legacy = TrackState(id=2, manual_external_player_id="legacy-player-10")

    _merge_raw_track_states(dedicated, legacy)

    assert dedicated.roster_binding_state == "bound"
    assert dedicated.manual_external_player_id == "player-home-8"

    unbound = TrackState(
        id=3,
        roster_binding_state="unbound",
        roster_binding_annotation_ids={"binding-b"},
        manual_external_player_id=None,
    )
    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _merge_raw_track_states(dedicated, unbound)


def _append_second_canonical_person(scene: dict) -> None:
    second = deepcopy(scene["payload"]["canonicalPeople"][0])
    second.update(
        {
            "id": "canonical-second",
            "canonicalPersonId": "canonical-second",
            "displayName": "Home person 03",
            "annotationIds": [],
        }
    )
    for index, observation in enumerate(second["observations"]):
        observation["id"] = observation["observationId"] = f"second-{index}"
        observation["frameIndex"] += 1
        observation["sceneTime"] += 0.1
        observation["bbox"]["x"] += 100
    scene["payload"]["canonicalPeople"].append(second)


def test_split_undo_restores_and_rekeys_preexisting_unbind(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])

    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "split-with-preexisting-unbind",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        },
        persist=False,
    )

    stored = split["preSplitRosterCorrections"]
    assert len(stored) == 1
    assert stored[0]["id"] == original_unbind["id"]
    assert stored[0]["rosterDecisionOriginId"] == original_unbind["id"]

    # Model the completed split: the original Unbind anchor belongs to the new
    # range child, so the next explicit edit transactionally rekeys it there.
    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(original_person)
    original_person["observations"] = [
        item
        for item in original_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    original_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [original_person, child_person]

    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None,
        persist=False,
    )
    assert child_unbind["id"] != original_unbind["id"]
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    delete_frame_person_annotation(scene, split["id"], persist=False)

    corrections = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert len(corrections) == 1
    restored = corrections[0]
    assert restored["id"] == original_unbind["id"]
    assert restored["canonicalPersonId"] == "canonical-offscreen"
    assert restored["rosterBindingState"] == "unbound"
    assert restored["targetObservation"]["canonicalPersonId"] == "canonical-offscreen"
    assert restored["targetObservation"]["annotationId"] == original_unbind["id"]


def test_clear_split_unbind_removes_pre_split_resurrection_path(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "split-before-clear",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        },
        persist=False,
    )

    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(original_person)
    original_person["observations"] = [
        item
        for item in original_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    original_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [original_person, child_person]
    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None,
        persist=False,
    )
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    clear_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        persist=False,
    )
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [split["id"]]
    assert remaining[0]["preSplitRosterCorrections"] == []

    delete_frame_person_annotation(scene, split["id"], persist=False)
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_post_merge_unbind_blocks_undo_until_explicit_clear(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-before-new-unbind",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        },
        persist=False,
    )

    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(source["observations"] + target["observations"])
    merged["annotationIds"] = [merge["id"]]
    scene["payload"]["canonicalPeople"] = [merged]

    post_merge_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None, persist=False
    )
    assert post_merge_unbind["identityCorrectionDependencies"] == [merge["id"]]

    with pytest.raises(
        ReconstructionError,
        match="created or changed after this merge",
    ):
        delete_frame_person_annotation(scene, merge["id"], persist=False)

    with pytest.raises(ReconstructionError, match="Bind / Unbind / Clear"):
        delete_frame_person_annotation(
            scene, post_merge_unbind["id"], persist=False
        )

    cleared = clear_canonical_roster_binding(
        scene, "canonical-second", persist=False
    )
    assert cleared["id"] == post_merge_unbind["id"]
    delete_frame_person_annotation(scene, merge["id"], persist=False)
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_merge_consolidates_two_unbind_tombstones_and_delete_restores_them(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-first-into-second",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        },
        persist=False,
    )

    annotations = scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert merge["consolidatedRosterCorrectionIds"] == [first_unbind["id"]]
    assert merge["consolidatedRosterCorrections"][0]["id"] == first_unbind["id"]
    assert {item["id"] for item in annotations} == {
        "merge-first-into-second",
        second_unbind["id"],
    }

    delete_frame_person_annotation(
        scene, "merge-first-into-second", persist=False
    )
    restored = scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert {item["id"] for item in restored} == {
        first_unbind["id"],
        second_unbind["id"],
    }
    assert all(item["rosterBindingState"] == "unbound" for item in restored)


def test_clear_merged_unbind_removes_consolidated_lineage_before_undo(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None, persist=False
    )
    assert first_unbind["rosterDecisionOriginId"] != second_unbind[
        "rosterDecisionOriginId"
    ]
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])

    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-before-clear",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        },
        persist=False,
    )
    assert merge["consolidatedRosterCorrectionIds"] == [first_unbind["id"]]

    # Model the completed rebuild: the published target represents both source
    # identities and owns the one visible compatible Unbind correction.
    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(
        source["observations"] + target["observations"]
    )
    merged["annotationIds"] = [merge["id"], second_unbind["id"]]
    scene["payload"]["canonicalPeople"] = [merged]

    cleared = clear_canonical_roster_binding(
        scene, "canonical-second", persist=False
    )
    assert cleared["id"] == second_unbind["id"]
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [merge["id"]]
    assert "consolidatedRosterCorrectionIds" not in remaining[0]
    assert "consolidatedRosterCorrections" not in remaining[0]

    delete_frame_person_annotation(scene, merge["id"], persist=False)
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_merged_unbind_rejects_unrelated_valid_snapshot_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    first_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    second_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])
    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "merge-with-corrupt-lineage",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": "canonical-offscreen",
            "merge_target_id": "canonical-second",
        },
        persist=False,
    )
    unrelated = deepcopy(first_unbind)
    unrelated.update(
        {
            "id": "unrelated-hidden-unbind",
            "canonicalPersonId": "canonical-unrelated",
            "rosterDecisionOriginId": "unrelated-hidden-unbind",
        }
    )
    unrelated["targetObservation"][
        "canonicalPersonId"
    ] = "canonical-unrelated"
    merge["consolidatedRosterCorrectionIds"] = [unrelated["id"]]
    merge["consolidatedRosterCorrections"] = [unrelated]

    source, target = scene["payload"]["canonicalPeople"]
    merged = deepcopy(target)
    merged["observations"] = deepcopy(
        source["observations"] + target["observations"]
    )
    merged["annotationIds"] = [merge["id"], second_unbind["id"]]
    scene["payload"]["canonicalPeople"] = [merged]
    before = deepcopy(scene)

    with pytest.raises(ReconstructionError, match="belongs to another identity"):
        clear_canonical_roster_binding(
            scene, "canonical-second", persist=False
        )
    assert scene == before


def test_clear_unbind_preserves_unrelated_malformed_split_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    unbound = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    frame = tmp_path / "frame_00208.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.7)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "unrelated-split",
            "scene_time": 0.7,
            "bbox": {"x": 220, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-second",
            "target_observation_id": "second-1",
            "range_start": 0.6,
            "range_end": 1.1,
        },
        persist=False,
    )
    split["preSplitRosterCorrections"] = "legacy-corrupt-but-unrelated"

    cleared = clear_canonical_roster_binding(
        scene, "canonical-offscreen", persist=False
    )
    assert cleared["id"] == unbound["id"]
    remaining = scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]
    assert [item["id"] for item in remaining] == [split["id"]]
    assert (
        remaining[0]["preSplitRosterCorrections"]
        == "legacy-corrupt-but-unrelated"
    )


def test_clear_unbind_rejects_related_malformed_split_metadata_atomically(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "related-corrupt-split",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        },
        persist=False,
    )
    split["preSplitRosterCorrections"][0]["correctionKind"] = "legacy-corrupt"
    before = deepcopy(scene)

    with pytest.raises(ReconstructionError, match="unsafe roster undo metadata"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen", persist=False
        )
    assert scene == before


def test_clear_nested_split_merge_lineage_prevents_ordered_undo_resurrection(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])
    split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "nested-split",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        },
        persist=False,
    )

    source_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    child_person = deepcopy(source_person)
    source_person["observations"] = [
        item
        for item in source_person["observations"]
        if item["observationId"] == "obs-low"
    ]
    source_person["annotationIds"] = []
    child_person.update(
        {
            "id": split["splitCanonicalPersonId"],
            "canonicalPersonId": split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_person["observations"]
                if item["observationId"] == "obs-best"
            ],
            "annotationIds": [original_unbind["id"], split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_person, child_person]
    child_unbind = set_canonical_roster_binding(
        scene,
        split["splitCanonicalPersonId"],
        None,
        persist=False,
    )
    assert child_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    _append_second_canonical_person(scene)
    target_person = scene["payload"]["canonicalPeople"][-1]
    target_unbind = set_canonical_roster_binding(
        scene, "canonical-second", None, persist=False
    )
    merge = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "nested-merge",
            "scene_time": 0.6,
            "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
            "kind": "home-player",
            "action": "merge",
            "scope": "identity",
            "canonical_person_id": split["splitCanonicalPersonId"],
            "merge_target_id": "canonical-second",
        },
        persist=False,
    )
    assert merge["consolidatedRosterCorrectionIds"] == [child_unbind["id"]]

    merged_target = deepcopy(target_person)
    merged_target["observations"] = deepcopy(
        child_person["observations"] + target_person["observations"]
    )
    merged_target["annotationIds"] = [
        split["id"],
        merge["id"],
        target_unbind["id"],
    ]
    scene["payload"]["canonicalPeople"] = [source_person, merged_target]

    clear_canonical_roster_binding(
        scene, "canonical-second", persist=False
    )
    remaining = {
        item["id"]: item
        for item in scene["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ]
    }
    assert set(remaining) == {split["id"], merge["id"]}
    assert remaining[split["id"]]["preSplitRosterCorrections"] == []
    assert "consolidatedRosterCorrections" not in remaining[merge["id"]]

    delete_frame_person_annotation(scene, merge["id"], persist=False)
    child_after_undo = deepcopy(child_person)
    child_after_undo["annotationIds"] = [split["id"]]
    target_after_undo = deepcopy(target_person)
    target_after_undo["annotationIds"] = []
    scene["payload"]["canonicalPeople"] = [
        source_person,
        child_after_undo,
        target_after_undo,
    ]
    delete_frame_person_annotation(scene, split["id"], persist=False)
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_nested_split_ancestry_prevents_parent_undo_resurrection(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    best = scene["payload"]["canonicalPeople"][0]["observations"][1]
    middle = deepcopy(best)
    middle.update(
        {
            "id": "obs-middle",
            "observationId": "obs-middle",
            "frameIndex": 209,
            "sceneTime": 0.8,
            "sourceTime": 20.8,
            "confidence": 0.99,
        }
    )
    middle["bbox"] = {
        "x": 150,
        "y": 80,
        "width": 24,
        "height": 58,
    }
    scene["payload"]["canonicalPeople"][0]["observations"].append(middle)
    original_unbind = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    assert original_unbind["targetObservationId"] == "obs-middle"
    parent_frame = tmp_path / "frame_00207.jpg"
    child_frame = tmp_path / "frame_00209.jpg"
    assert cv2.imwrite(
        str(parent_frame), np.zeros((240, 360, 3), dtype=np.uint8)
    )
    assert cv2.imwrite(
        str(child_frame), np.zeros((240, 360, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        "app.reconstruction._frame_paths",
        lambda _: [(parent_frame, 0.6), (child_frame, 0.8)],
    )
    parent_split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "parent-split",
            "scene_time": 0.6,
            "bbox": best["bbox"],
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-offscreen",
            "target_observation_id": "obs-best",
            "range_start": 0.5,
            "range_end": 1.0,
        },
        persist=False,
    )

    original_person = deepcopy(scene["payload"]["canonicalPeople"][0])
    source_a = deepcopy(original_person)
    source_a["observations"] = [
        item
        for item in source_a["observations"]
        if item["observationId"] == "obs-low"
    ]
    source_a["annotationIds"] = []
    child_b = deepcopy(original_person)
    child_b.update(
        {
            "id": parent_split["splitCanonicalPersonId"],
            "canonicalPersonId": parent_split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_b["observations"]
                if item["observationId"] in {"obs-best", "obs-middle"}
            ],
            "annotationIds": [original_unbind["id"], parent_split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_a, child_b]
    child_b_unbind = set_canonical_roster_binding(
        scene,
        parent_split["splitCanonicalPersonId"],
        None,
        persist=False,
    )
    child_split = upsert_frame_person_annotation(
        scene,
        {
            "annotation_id": "child-split",
            "scene_time": 0.8,
            "bbox": middle["bbox"],
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": parent_split["splitCanonicalPersonId"],
            "target_observation_id": "obs-middle",
            "range_start": 0.75,
            "range_end": 0.9,
        },
        persist=False,
    )
    assert child_split["preSplitRosterCorrections"][0]["id"] == child_b_unbind["id"]

    remaining_b = deepcopy(child_b)
    remaining_b["observations"] = [
        item
        for item in remaining_b["observations"]
        if item["observationId"] == "obs-best"
    ]
    remaining_b["annotationIds"] = [parent_split["id"]]
    child_c = deepcopy(child_b)
    child_c.update(
        {
            "id": child_split["splitCanonicalPersonId"],
            "canonicalPersonId": child_split["splitCanonicalPersonId"],
            "observations": [
                item
                for item in child_c["observations"]
                if item["observationId"] == "obs-middle"
            ],
            # Deliberately omit the parent id: Clear must discover it through
            # the B -> C split's transitive source ancestry.
            "annotationIds": [child_b_unbind["id"], child_split["id"]],
        }
    )
    scene["payload"]["canonicalPeople"] = [source_a, remaining_b, child_c]
    child_c_unbind = set_canonical_roster_binding(
        scene,
        child_split["splitCanonicalPersonId"],
        None,
        persist=False,
    )
    assert child_c_unbind["rosterDecisionOriginId"] == original_unbind["id"]

    clear_canonical_roster_binding(
        scene,
        child_split["splitCanonicalPersonId"],
        persist=False,
    )
    remaining = {
        item["id"]: item
        for item in scene["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ]
    }
    assert remaining[parent_split["id"]]["preSplitRosterCorrections"] == []
    assert remaining[child_split["id"]]["preSplitRosterCorrections"] == []

    delete_frame_person_annotation(scene, child_split["id"], persist=False)
    recombined_b = deepcopy(child_b)
    recombined_b["annotationIds"] = [parent_split["id"]]
    scene["payload"]["canonicalPeople"] = [source_a, recombined_b]
    delete_frame_person_annotation(scene, parent_split["id"], persist=False)
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == []


def test_clear_unbind_is_explicit_idempotent_and_preserves_generic_correction() -> None:
    scene = _scene()
    generic = {
        "id": "reviewed-role-only",
        "frameIndex": 207,
        "sceneTime": 0.6,
        "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-goalkeeper",
        "label": "Reviewed goalkeeper",
        "externalPlayerId": None,
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-offscreen",
        "updatedAt": "2026-07-17T00:00:00+00:00",
    }
    scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ].append(generic)
    bound = set_canonical_roster_binding(
        scene, "canonical-offscreen", "player-home-8", persist=False
    )

    with pytest.raises(ReconstructionError, match="Unbind the roster player"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen", persist=False
        )
    assert bound in scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ]

    unbound = set_canonical_roster_binding(
        scene, "canonical-offscreen", None, persist=False
    )
    cleared = clear_canonical_roster_binding(
        scene, "canonical-offscreen", persist=False
    )
    assert cleared["id"] == unbound["id"]
    assert scene["payload"]["videoAsset"]["reconstruction"][
        "frameAnnotations"
    ] == [generic]

    with pytest.raises(ReconstructionError, match="no roster decision to clear"):
        clear_canonical_roster_binding(
            scene, "canonical-offscreen", persist=False
        )


def test_merge_rejects_bound_and_unbound_dedicated_decisions(
    monkeypatch,
    tmp_path,
) -> None:
    scene = _scene()
    _append_second_canonical_person(scene)
    set_canonical_roster_binding(
        scene, "canonical-offscreen", "player-home-8", persist=False
    )
    set_canonical_roster_binding(scene, "canonical-second", None, persist=False)
    frame = tmp_path / "frame_00207.jpg"
    assert cv2.imwrite(str(frame), np.zeros((240, 360, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.6)])

    with pytest.raises(
        ReconstructionError,
        match="different dedicated Bind / Unbind decisions",
    ):
        upsert_frame_person_annotation(
            scene,
            {
                "annotation_id": "merge-first-into-second",
                "scene_time": 0.6,
                "bbox": {"x": 120, "y": 80, "width": 24, "height": 58},
                "kind": "home-player",
                "action": "merge",
                "scope": "identity",
                "canonical_person_id": "canonical-offscreen",
                "merge_target_id": "canonical-second",
            },
            persist=False,
        )


def test_conflicting_legacy_roster_confirms_on_one_track_fail_closed() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    annotations = [
        {
            "id": f"legacy-{external_id}",
            "frameIndex": index,
            "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
            "kind": "home-player",
            "label": external_id,
            "externalPlayerId": external_id,
            "action": "confirm",
        }
        for index, external_id in enumerate(("legacy-a", "legacy-b"))
    ]
    detections = [
        _apply_person_annotations(image, [], [annotation])[0]
        for annotation in annotations
    ]
    track = TrackState(id=93)
    track.append(detections[0], 0, 0.0)

    with pytest.raises(ReconstructionError, match="Conflicting legacy roster confirmations"):
        track.append(detections[1], 1, 0.1)


def test_dedicated_binding_suppresses_same_identity_legacy_value_before_tracking() -> None:
    scene = _scene()
    dedicated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    legacy = {
        "id": "legacy-other-frame",
        "frameIndex": 212,
        "sceneTime": 1.2,
        "bbox": {"x": 140.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-player",
        "label": "Legacy",
        "externalPlayerId": "legacy-player-10",
        "action": "confirm",
        "scope": "identity",
        "canonicalPersonId": "canonical-offscreen",
    }
    scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"].append(
        legacy
    )

    filtered = _frame_annotations(scene, 212)

    assert len(filtered) == 1
    assert filtered[0]["id"] == legacy["id"]
    assert filtered[0]["externalPlayerId"] is None
    assert filtered[0]["rosterValueSupersededByDedicatedCorrection"] is True
    assert _frame_annotations(scene, dedicated["frameIndex"])[0]["externalPlayerId"] == "player-home-8"


def test_split_binding_is_rekeyed_before_unbind_without_leaving_old_positive_correction():
    scene = _scene()
    bound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    old_binding_id = bound["id"]
    feature = np.ones(12, dtype=np.float32)
    source = TrackState(
        id=1,
        points=[
            {
                "t": 0.0,
                "px": 29.0,
                "py": 72.0,
                "frameIndex": 201,
                "observationId": "obs-low",
                "sourceTrackletId": "tracklet-0001",
                "bbox": {"x": 20.0, "y": 30.0, "width": 18.0, "height": 42.0},
                "confidence": 0.45,
                "annotationId": None,
                "_appearanceFeature": feature.copy(),
            },
            {
                "t": 0.6,
                "px": 132.0,
                "py": 138.0,
                "frameIndex": 207,
                "observationId": "obs-best",
                "sourceTrackletId": "tracklet-0001",
                "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
                "confidence": 0.94,
                "annotationId": old_binding_id,
                "_appearanceFeature": feature.copy(),
            },
            {
                "t": 1.5,
                "px": 150.0,
                "py": 140.0,
                "frameIndex": 215,
                "observationId": "obs-late",
                "sourceTrackletId": "tracklet-0001",
                "bbox": {"x": 138.0, "y": 82.0, "width": 24.0, "height": 58.0},
                "confidence": 0.88,
                "annotationId": None,
                "_appearanceFeature": feature.copy(),
            },
        ],
        feature_sum=feature * 3,
        feature_count=3,
        last_frame=215,
        last_height=58.0,
        annotation_ids={old_binding_id},
        manual_kind="home-player",
        manual_label="Home Eight",
        manual_external_player_id="player-home-8",
        source_tracklet_ids={"tracklet-0001"},
        canonical_person_id="canonical-offscreen",
    )
    split_annotation = {
        "id": "split-bound-range",
        "kind": "home-player",
        "action": "split",
        "scope": "range",
        "canonicalPersonId": "canonical-offscreen",
        "targetObservationId": "obs-best",
        "targetObservation": {
            "observationId": "obs-best",
            "frameIndex": 207,
            "sceneTime": 0.6,
            "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
            "canonicalPersonId": "canonical-offscreen",
        },
        "rangeStart": 0.5,
        "rangeEnd": 1.0,
        "splitCanonicalPersonId": "canonical-split-bound",
    }
    scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"].append(
        split_annotation
    )

    split_tracks, _ = _apply_canonical_split_corrections([source], scene)
    people, _ = _canonical_people_documents(
        split_tracks,
        {track.id: "home" for track in split_tracks},
        [],
        scene,
    )
    scene["payload"]["canonicalPeople"] = people
    split_person = next(
        item
        for item in people
        if item["canonicalPersonId"] == "canonical-split-bound"
    )
    assert split_person["externalPlayerId"] == "player-home-8"
    assert old_binding_id in split_person["annotationIds"]

    with pytest.raises(ReconstructionError, match="owned by another canonical person"):
        set_canonical_roster_binding(
            scene,
            "canonical-offscreen",
            None,
            persist=False,
        )

    unbound = set_canonical_roster_binding(
        scene,
        "canonical-split-bound",
        None,
        persist=False,
    )

    roster_corrections = [
        item
        for item in scene["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ]
        if item.get("correctionKind") == "canonical-roster-binding-v1"
    ]
    assert len(roster_corrections) == 1
    assert roster_corrections[0] == unbound
    assert unbound["id"] != old_binding_id
    assert unbound["canonicalPersonId"] == "canonical-split-bound"
    assert unbound["externalPlayerId"] is None
    assert unbound["rosterBindingState"] == "unbound"
    assert all(
        old_binding_id not in item.get("annotationIds", [])
        for item in scene["payload"]["canonicalPeople"]
    )
    updated_split = next(
        item
        for item in scene["payload"]["canonicalPeople"]
        if item["canonicalPersonId"] == "canonical-split-bound"
    )
    assert unbound["id"] in updated_split["annotationIds"]
    assert updated_split["displayName"] == "Home person"
    assert updated_split["identityStatus"] == "resolved"
    assert updated_split["identityConfidence"] == 1.0
    assert updated_split["identitySource"] == "manual"
    assert unbound["baseDisplayName"] == "Home person"
    assert unbound["baseIdentityStatus"] == "resolved"
    assert unbound["baseIdentityConfidence"] == 1.0
    assert unbound["baseIdentitySource"] == "manual"

    rebuilt_detections = _apply_person_annotations(
        np.zeros((240, 320, 3), dtype=np.uint8),
        [],
        [unbound],
    )
    assert len(rebuilt_detections) == 1
    assert rebuilt_detections[0].external_player_id is None
    assert rebuilt_detections[0].annotation_is_identity_evidence is False


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_roster_binding_api_rejects_a_running_reconstruction(monkeypatch, status):
    scene = _scene(status=status)
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(scene))

    response = _request(
        "PUT",
        "/api/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
        json={"external_player_id": "player-home-8"},
    )

    assert response.status_code == 409
    assert "Wait for reconstruction" in response.json()["detail"]


def test_roster_binding_api_queues_the_correction_without_a_current_frame(monkeypatch):
    scene = _scene()
    captured: dict = {}
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(scene))

    def save(value, canonical_person_id, external_player_id, *, persist=True):
        captured.update(
            canonical_person_id=canonical_person_id,
            external_player_id=external_player_id,
            persist=persist,
        )
        value["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
            {"id": "roster-correction", "action": "confirm", "scope": "identity"}
        ]

    def queue(value, **kwargs):
        captured["expected_scene_fingerprint"] = kwargs["expected_scene_fingerprint"]
        value["payload"]["videoAsset"]["reconstruction"].update(
            {
                "status": "queued",
                "runId": "run-roster",
                "runRevision": 4,
                "inputFingerprint": "sha256:roster",
            }
        )
        return value

    monkeypatch.setattr("app.main.set_canonical_roster_binding", save)
    monkeypatch.setattr("app.main.queue_reconstruction", queue)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)

    response = _request(
        "PUT",
        "/api/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
        json={"external_player_id": "player-home-8"},
    )

    assert response.status_code == 202
    assert response.json()["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-roster"
    assert captured["canonical_person_id"] == "canonical-offscreen"
    assert captured["external_player_id"] == "player-home-8"
    assert captured["persist"] is False
    assert captured["expected_scene_fingerprint"].startswith("sha256:")


def test_roster_clear_api_queues_without_running_frame_analysis(monkeypatch):
    scene = _scene()
    captured: dict = {}
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(scene))

    def clear(value, canonical_person_id, *, persist=True):
        captured.update(
            canonical_person_id=canonical_person_id,
            persist=persist,
        )
        value["payload"]["videoAsset"]["reconstruction"][
            "frameAnnotations"
        ] = []

    def queue(value, **kwargs):
        captured["expected_scene_fingerprint"] = kwargs[
            "expected_scene_fingerprint"
        ]
        value["payload"]["videoAsset"]["reconstruction"].update(
            {
                "status": "queued",
                "runId": "run-clear",
                "runRevision": 4,
                "inputFingerprint": "sha256:clear",
            }
        )
        return value

    monkeypatch.setattr("app.main.clear_canonical_roster_binding", clear)
    monkeypatch.setattr("app.main.queue_reconstruction", queue)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr(
        "app.main.analyze_scene_frame",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not analyze a frame")),
    )

    response = _request(
        "DELETE",
        "/api/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
    )

    assert response.status_code == 202
    assert response.json()["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-clear"
    assert captured["canonical_person_id"] == "canonical-offscreen"
    assert captured["persist"] is False
    assert captured["expected_scene_fingerprint"].startswith("sha256:")


def test_stale_roster_binding_api_cannot_partially_persist_the_correction(monkeypatch):
    persisted = _scene()
    original = deepcopy(persisted)
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(persisted))

    def stale(*_args, **_kwargs):
        raise StaleReconstructionRun("superseded")

    monkeypatch.setattr("app.main.queue_reconstruction", stale)

    response = _request(
        "PUT",
        "/api/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
        json={"external_player_id": "player-home-8"},
    )

    assert response.status_code == 409
    assert persisted == original
    assert persisted["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == []


def test_roster_binding_queue_supersedes_an_old_worker_atomically(
    isolated_store,
    monkeypatch,
):
    ready = _scene()
    old_fingerprint = reconstruction_input_fingerprint(ready)
    ready["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = old_fingerprint
    isolated_store.put(ready)
    stale_worker_result = deepcopy(ready)
    stale_worker_result["payload"]["tracks"] = [{"id": "stale-track"}]

    edited = isolated_store.get(ready["id"])
    assert edited is not None
    set_canonical_roster_binding(
        edited,
        "canonical-offscreen",
        "player-home-8",
        persist=False,
    )
    monkeypatch.setattr("app.reconstruction.scene_store", isolated_store)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda *_: [])
    queued = queue_reconstruction(
        edited,
        expected_scene_fingerprint=old_fingerprint,
    )

    assert queued["payload"]["videoAsset"]["reconstruction"]["runId"] != "run-ready"
    assert isolated_store.put_if_reconstruction_run(
        stale_worker_result,
        "run-ready",
        old_fingerprint,
    ) is False
    saved = isolated_store.get(ready["id"])
    assert saved is not None
    annotations = saved["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert len(annotations) == 1
    assert annotations[0]["externalPlayerId"] == "player-home-8"
    assert saved["payload"]["tracks"] == []
