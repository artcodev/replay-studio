from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.project_resource_access as resource_access
import app.scene_identity_routes as scene_identity_routes
from app.project_lifecycle_contract import ProjectCreate
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.database import Base
from app.main import app
from app.reconstruction_errors import ReconstructionError, StaleReconstructionRun
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete as delete_frame_person_annotation,
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_identity_roster_binding_planning import (
    plan_canonical_roster_binding as _plan_canonical_roster_binding,
)
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_run_repository import ReconstructionRunRepository
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation
from app.reconstruction_identity_merging import (
    merge_raw_track_states as _merge_raw_track_states,
)
from app.reconstruction_identity_splitting import (
    apply_canonical_split_corrections as _apply_canonical_split_corrections,
)
from app.reconstruction_canonical_people_projection import (
    canonical_people_documents as _canonical_people_documents,
)
from app.reconstruction_identity_persistence import (
    assign_persistent_canonical_person_ids as _assign_persistent_canonical_person_ids,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)
from app.reconstruction_person_annotations import (
    apply_person_annotations as _apply_person_annotations,
    frame_annotations as _frame_annotations,
)
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


_MATCH_SNAPSHOT = {
    "homeTeam": {"id": "team-home", "name": "Home"},
    "awayTeam": {"id": "team-away", "name": "Away"},
    "roster": [
        {
            "id": "player-home-8",
            "name": "Home Eight",
            "teamId": "team-home",
            "number": "8",
        },
        {
            "id": "player-away-10",
            "name": "Away Ten",
            "teamId": "team-away",
            "number": "10",
        },
    ],
}


def set_canonical_roster_binding(*args, **kwargs):
    kwargs.setdefault("match_snapshot", _MATCH_SNAPSHOT)
    return _plan_canonical_roster_binding(*args, **kwargs)


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
    def owned_scene(_project_id: str, scene_id: str):
        scene = resource_access.scenes.get(scene_id)
        if scene is None:
            raise HTTPException(status_code=404, detail="Scene not found in project")
        return scene

    with patch.object(
        resource_access,
        "project_scene_or_404",
        side_effect=owned_scene,
    ), patch.object(
        scene_identity_routes.project_matches,
        "current_snapshot",
        return_value=SimpleNamespace(payload=_MATCH_SNAPSHOT),
    ):
        return asyncio.run(_async_request(method, path, **kwargs))


@dataclass(frozen=True)
class Persistence:
    documents: SceneRepository
    runs: ReconstructionRunRepository
    sessions: sessionmaker


@pytest.fixture
def isolated_store(monkeypatch) -> Persistence:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    return Persistence(
        documents=SceneRepository(session_local),
        runs=ReconstructionRunRepository(session_local),
        sessions=session_local,
    )


def test_offscreen_roster_binding_uses_saved_detector_observation_and_survives_worker_outage(
    monkeypatch,
):
    scene = _scene()
    before = reconstruction_input_fingerprint(scene)
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.frame_paths",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not read the current frame")),
    )

    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8"
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
    append_track_observation(rebuilt_track, detections[0], annotation["frameIndex"], annotation["sceneTime"])
    _assign_persistent_canonical_person_ids([rebuilt_track], scene, {77: "home"})
    assert rebuilt_track.canonical_person_id == "canonical-offscreen"


def test_ambiguous_roster_anchor_ownership_fails_closed() -> None:
    scene = _scene()
    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8"
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
            None
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
            "player-missing"
        )

    scene = _scene()
    with pytest.raises(ReconstructionError, match="other team"):
        set_canonical_roster_binding(
            scene,
            "canonical-offscreen",
            "player-away-10"
        )


def test_roster_unbind_replaces_the_same_durable_correction_deterministically():
    scene = _scene()
    bound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8"
    )
    unbound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None
    )
    repeated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None
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
        "player-home-8"
    )
    annotation = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        None
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
    append_track_observation(rebuilt,
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


def test_generic_frame_annotation_cannot_carry_roster_identity() -> None:
    annotation = {
        "id": "invalid-generic-roster-confirm",
        "sceneTime": 1.2,
        "frameIndex": 212,
        "bbox": {"x": 120.0, "y": 80.0, "width": 24.0, "height": 58.0},
        "kind": "home-player",
        "label": "Invalid binding",
        "externalPlayerId": "player-home-8",
        "action": "confirm",
        "scope": "identity",
    }
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(
        ReconstructionError,
        match="requires a dedicated Bind / Unbind correction",
    ):
        _apply_person_annotations(image, [], [annotation])


def test_later_authored_role_label_edit_does_not_change_dedicated_roster_id() -> None:
    scene = _scene()
    dedicated = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8"
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
    append_track_observation(track, _apply_person_annotations(image, [], [dedicated])[0], 0, 0.0)
    append_track_observation(track, _apply_person_annotations(image, [], [generic])[0], 1, 1.2)

    assert track.manual_external_player_id == "player-home-8"
    assert track.manual_kind == "home-goalkeeper"
    assert track.manual_label == "Reviewed goalkeeper"


def test_merge_rejects_invalid_or_conflicting_roster_decisions() -> None:
    bound = TrackState(
        id=1,
        roster_binding_state="bound",
        roster_binding_annotation_ids={"binding-a"},
        manual_external_player_id="player-home-8",
    )
    invalid = TrackState(id=2, manual_external_player_id="player-home-10")
    with pytest.raises(ReconstructionError, match="missing its dedicated binding state"):
        _merge_raw_track_states(bound, invalid)

    unbound = TrackState(
        id=3,
        roster_binding_state="unbound",
        roster_binding_annotation_ids={"binding-b"},
        manual_external_player_id=None,
    )
    with pytest.raises(ReconstructionError, match="different confirmed roster players"):
        _merge_raw_track_states(bound, unbound)
