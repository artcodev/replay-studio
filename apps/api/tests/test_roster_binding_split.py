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


def test_split_binding_is_rekeyed_before_unbind_without_leaving_old_positive_correction():
    scene = _scene()
    bound = set_canonical_roster_binding(
        scene,
        "canonical-offscreen",
        "player-home-8"
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
            None
        )

    unbound = set_canonical_roster_binding(
        scene,
        "canonical-split-bound",
        None
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
