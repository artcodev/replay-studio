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
from app.reconstruction_calibration_fingerprint import calibration_input_fingerprint
from app.reconstruction_calibration_snapshot import calibration_data_fingerprint
from app.artifact_store import reconstruction_artifact_store
from app.reconstruction_artifact_codec import materialized_artifacts
from app.reconstruction_artifact_manifest import (
    artifact_references,
    merge_artifact_manifest,
)
from app.reconstruction_calibration_artifacts import publish_calibration_frames_artifact
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
    scene = {
        "id": "roster-binding-scene",
        "title": "Roster binding",
        "version": 1,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "fps": 10.0,
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
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["stage"] = "reconstruction"
    reconstruction["trackingCoordinatePolicy"] = "metric-required"
    reconstruction["calibrationInputFingerprint"] = calibration_input_fingerprint(
        scene
    )
    reconstruction["calibration"] = {
        "schemaVersion": 2,
        "summary": {},
        "manualFrameAnchors": [],
        "frameEvidence": [],
    }
    reconstruction["pitchCalibration"] = {"status": "ready"}
    reconstruction["pitchOrientation"] = {}
    data_fingerprint = calibration_data_fingerprint(reconstruction)
    published_calibration = publish_calibration_frames_artifact(
        scene,
        reconstruction,
        artifact_references(reconstruction),
        materialized_artifacts(reconstruction),
        store=reconstruction_artifact_store(),
    )
    calibration_artifact = published_calibration.reference
    reconstruction["artifactManifest"] = merge_artifact_manifest(
        reconstruction.get("artifactManifest"),
        calibrationFrames=calibration_artifact,
    )
    assert published_calibration.encoding is not None
    reconstruction["calibration"] = (
        published_calibration.encoding.compact_calibration
    )
    reconstruction["ballDetection"] = (
        published_calibration.encoding.compact_ball_detection
    )
    reconstruction["calibrationProvenance"] = {
        "schemaVersion": 1,
        "runId": "calibration-run",
        "producedAt": "2026-07-21T12:00:00+00:00",
        "calibrationInputFingerprint": reconstruction[
            "calibrationInputFingerprint"
        ],
        "dataFingerprint": data_fingerprint,
        "artifact": calibration_artifact,
        "totalFrames": 0,
        "resolvedFrames": 0,
        "unresolvedFrames": 0,
    }
    return scene


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


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_roster_binding_api_rejects_a_running_reconstruction(monkeypatch, status):
    scene = _scene(status=status)
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: deepcopy(scene))

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
        json={"external_player_id": "player-home-8"},
    )

    assert response.status_code == 409
    assert "Wait for reconstruction" in response.json()["detail"]


def test_roster_binding_api_queues_the_correction_without_a_current_frame(monkeypatch):
    scene = _scene()
    captured: dict = {}
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: deepcopy(scene))

    def save(
        value,
        canonical_person_id,
        external_player_id,
        *,
        match_snapshot=None,
    ):
        captured.update(
            canonical_person_id=canonical_person_id,
            external_player_id=external_player_id,
            match_snapshot=match_snapshot,
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

    monkeypatch.setattr("app.scene_identity_routes.draft_canonical_roster_binding", save)
    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
        json={"external_player_id": "player-home-8"},
    )

    assert response.status_code == 202
    assert response.json()["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-roster"
    assert captured["canonical_person_id"] == "canonical-offscreen"
    assert captured["external_player_id"] == "player-home-8"
    assert "persist" not in captured
    assert captured["expected_scene_fingerprint"].startswith("sha256:")


def test_roster_clear_api_queues_without_running_frame_analysis(monkeypatch):
    scene = _scene()
    captured: dict = {}
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: deepcopy(scene))

    def clear(value, canonical_person_id):
        captured.update(canonical_person_id=canonical_person_id)
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

    monkeypatch.setattr(
        "app.scene_identity_routes.draft_clear_canonical_roster_binding",
        clear,
    )
    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)
    monkeypatch.setattr(
        "app.scene_identity_routes.analyze_scene_frame",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not analyze a frame")),
    )

    response = _request(
        "DELETE",
        "/api/projects/project-test/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
    )

    assert response.status_code == 202
    assert response.json()["payload"]["videoAsset"]["reconstruction"]["runId"] == "run-clear"
    assert captured["canonical_person_id"] == "canonical-offscreen"
    assert "persist" not in captured
    assert captured["expected_scene_fingerprint"].startswith("sha256:")


def test_stale_roster_binding_api_cannot_partially_persist_the_correction(monkeypatch):
    persisted = _scene()
    original = deepcopy(persisted)
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: deepcopy(persisted))

    def stale(*_args, **_kwargs):
        raise StaleReconstructionRun("superseded")

    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", stale)

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/roster-binding-scene/canonical-people/canonical-offscreen/roster-binding",
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
    ready = isolated_store.documents.put(ready)
    projects = ProjectStore(isolated_store.sessions)
    resources = ProjectResourceRepository(isolated_store.sessions)
    projects.create_project(ProjectCreate(id="project-test", title="Roster test"))
    resources.link_scene("project-test", ready["id"], role="segment")
    stale_worker_result = deepcopy(ready)
    stale_worker_result["payload"]["tracks"] = [{"id": "stale-track"}]

    edited = isolated_store.documents.get(ready["id"])
    assert edited is not None
    set_canonical_roster_binding(
        edited,
        "canonical-offscreen",
        "player-home-8"
    )
    monkeypatch.setattr("app.reconstruction_queue.reconstruction_runs", isolated_store.runs)
    monkeypatch.setattr(
        "app.reconstruction_queue.frame_paths",
        lambda *_args, **_kwargs: [],
    )
    queued = queue_reconstruction(
        edited,
        match_snapshot=None,
        expected_scene_fingerprint=old_fingerprint,
    )

    assert queued["payload"]["videoAsset"]["reconstruction"]["runId"] != "run-ready"
    assert isolated_store.runs.put_if_reconstruction_run(
        stale_worker_result,
        "run-ready",
        old_fingerprint,
    ) is False
    saved = isolated_store.documents.get(ready["id"])
    assert saved is not None
    annotations = saved["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"]
    assert len(annotations) == 1
    assert annotations[0]["externalPlayerId"] == "player-home-8"
    assert saved["payload"]["tracks"] == []
