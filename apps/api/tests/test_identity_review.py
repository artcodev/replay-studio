import asyncio
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

import app.identity_review_routes as review_routes
from app.identity_review_contract import IdentityReviewResponse
from app.identity_review_crop_service import identity_observation_crop
from app.identity_review_errors import IdentityReviewError
from app.identity_review_http_presenter import present_identity_review
from app.identity_review_projection import build_identity_review_projection
from app.identity_review_routes import router as identity_review_router
from app.project_match_persistence_contract import MatchSnapshotDocument
from app.project_resource_repository import ProjectResourceConflict
from app.artifact_store import FilesystemArtifactStore
from app.reconstruction_identity_artifacts import publish_identity_diagnostics


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    def owned_scene(_project_id: str, scene_id: str):
        value = review_routes.scenes.get(scene_id)
        if value is None:
            raise HTTPException(status_code=404, detail="Scene not found in project")
        return value

    with patch.object(review_routes, "_owned_scene", side_effect=owned_scene):
        return asyncio.run(_async_request(application, method, path, **kwargs))


def scene() -> dict:
    return {
        "id": "scene-identity",
        "revision": 7,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "generationKey": "identity-test-generation",
                "reconstruction": {},
            },
            "canonicalPeople": [
                {
                    "canonicalPersonId": "canonical-1",
                    "displayName": "Away person",
                    "identityStatus": "provisional",
                    "teamId": "away",
                    "observationCount": 2,
                    "observations": [
                        {
                            "observationId": "obs-1",
                            "frameIndex": 1,
                            "sceneTime": 0.0,
                            "confidence": 0.8,
                            "bbox": {"x": 10, "y": 10, "width": 20, "height": 40},
                        },
                        {
                            "observationId": "obs-2",
                            "frameIndex": 5,
                            "sceneTime": 0.5,
                            "confidence": 0.9,
                            "bbox": {"x": 30, "y": 8, "width": 30, "height": 60},
                        },
                    ],
                    "rosterCandidates": [
                        {"externalPlayerId": "p8", "rank": 1, "score": 0.91}
                    ],
                    "conflicts": [],
                },
                {
                    "canonicalPersonId": "canonical-2",
                    "identityStatus": "provisional",
                    "observations": [],
                    "rosterCandidates": [],
                    "conflicts": [{"code": "jersey-ocr-conflict"}],
                },
            ],
        },
    }


def match_snapshot() -> MatchSnapshotDocument:
    return MatchSnapshotDocument(
        id="snapshot-review",
        project_id="project-review",
        match_id="match-review",
        provider="test-provider",
        external_event_id="event-1",
        schema_version=1,
        fetched_at="2026-07-18T10:00:00Z",
        content_hash="sha256:review",
        is_current=True,
        payload={
            "roster": [{"id": f"p{index}"} for index in range(1, 6)],
            "rosterQuality": {
                "automaticIdentityEligible": False,
                "manualIdentityEligible": True,
                "reasons": ["canonical-roster-incomplete"],
            },
            "sync": {
                "warnings": [
                    "The free source returned only the first five lineup entries."
                ]
            },
        },
    )


def _identity_diagnostics() -> dict:
    return {
        "reid": {
            "crops": [
                {
                    "observationId": "obs-1",
                    "usable": False,
                    "status": "rejected",
                    "rejectionReasons": ["too-small"],
                }
            ]
        },
        "jerseyOcr": {
            "crops": [
                {
                    "observationId": "obs-2",
                    "status": "recognized",
                    "number": "8",
                }
            ]
        },
    }


def _scene_with_identity_artifact(tmp_path: Path) -> tuple[dict, FilesystemArtifactStore]:
    value = scene()
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    manifest, summary = publish_identity_diagnostics(_identity_diagnostics(), store=store)
    reconstruction = value["payload"]["videoAsset"]["reconstruction"]
    reconstruction["artifactManifest"] = manifest
    reconstruction["diagnostics"] = {"identity": summary}
    return value, store


def test_build_identity_review_prioritizes_conflicts_and_keeps_crop_evidence(tmp_path):
    value, artifact_store = _scene_with_identity_artifact(tmp_path)
    result = present_identity_review(
        build_identity_review_projection(
            value,
            match_snapshot=match_snapshot(),
            worker_health={"identity": {"status": "ready"}},
            artifact_store=artifact_store,
        ),
        project_id="project-review",
        scene_id="scene-identity",
    )

    assert result["summary"] == {
        "canonicalPersonCount": 2,
        "boundCount": 0,
        "suggestedCount": 1,
        "conflictCount": 1,
        "anonymousCount": 0,
        "excludedCount": 0,
    }
    assert result["matchSnapshot"]["roster"]["status"] == "incomplete"
    assert result["items"][0]["canonicalPersonId"] == "canonical-2"
    suggested = result["items"][1]
    assert suggested["resolutionState"] == "suggested"
    assert suggested["representativeObservations"][0]["reid"]["rejectionReasons"] == [
        "too-small"
    ]
    assert suggested["representativeObservations"][1]["jerseyOcr"]["number"] == "8"
    assert suggested["representativeObservations"][1]["cropUrl"].endswith(
        "/obs-2/crop"
    )


def test_identity_review_is_empty_but_available_to_read_before_reconstruction():
    result = build_identity_review_projection(
        scene(),
        match_snapshot=match_snapshot(),
    )

    assert result["availability"] == {"state": "not-started", "available": False}
    assert result["items"] == []
    assert result["summary"] == {
        "canonicalPersonCount": 0,
        "boundCount": 0,
        "suggestedCount": 0,
        "conflictCount": 0,
        "anonymousCount": 0,
        "excludedCount": 0,
    }
    assert result["matchSnapshot"]["roster"]["status"] == "incomplete"


@pytest.mark.parametrize("status", ("queued", "processing", "failed", "cancelled"))
def test_identity_review_exposes_non_ready_reconstruction_state(status: str):
    value = scene()
    value["payload"]["videoAsset"]["reconstruction"] = {"status": status}

    result = build_identity_review_projection(value, match_snapshot=match_snapshot())

    assert result["availability"] == {"state": status, "available": False}
    assert result["items"] == []


def test_identity_review_marks_ready_scene_without_diagnostics_as_unavailable():
    value = scene()
    value["payload"]["videoAsset"]["reconstruction"] = {"status": "ready"}

    result = build_identity_review_projection(value, match_snapshot=match_snapshot())

    assert result["availability"] == {
        "state": "unavailable",
        "available": False,
        "reasonCode": "identity-diagnostics-not-published",
    }


def test_identity_observation_crop_uses_persisted_bbox(tmp_path: Path):
    frames = (
        tmp_path
        / "asset-1"
        / ".pipeline-runs"
        / "identity-test-generation"
        / "frames"
    )
    frames.mkdir(parents=True)
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[8:80, 8:75] = (10, 180, 240)
    assert cv2.imwrite(str(frames / "frame_00001.jpg"), image)

    content = identity_observation_crop(scene(), "obs-1", media_root=tmp_path)
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape[0] > 40
    assert decoded.shape[1] > 20


def test_identity_observation_crop_rejects_unknown_observation(tmp_path: Path):
    with pytest.raises(IdentityReviewError, match="not found"):
        identity_observation_crop(scene(), "missing", media_root=tmp_path)


def test_source_frame_zero_is_preserved_in_projection_and_crop(tmp_path: Path):
    value, artifact_store = _scene_with_identity_artifact(tmp_path)
    observation = value["payload"]["canonicalPeople"][0]["observations"][0]
    observation["frameIndex"] = 9
    observation["sourceFrameIndex"] = 0

    review = build_identity_review_projection(
        value,
        match_snapshot=match_snapshot(),
        artifact_store=artifact_store,
    )
    projected = next(
        item
        for item in review["items"]
        if item["canonicalPersonId"] == "canonical-1"
    )["representativeObservations"][0]
    assert projected["frameIndex"] == 9
    assert projected["sourceFrameIndex"] == 0

    frames = (
        tmp_path
        / "asset-1"
        / ".pipeline-runs"
        / "identity-test-generation"
        / "frames"
    )
    frames.mkdir(parents=True)
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[8:80, 8:75] = (10, 180, 240)
    assert cv2.imwrite(str(frames / "frame_00000.jpg"), image)

    crop_scene = scene()
    crop_observation = crop_scene["payload"]["canonicalPeople"][0]["observations"][0]
    crop_observation["frameIndex"] = 9
    crop_observation["sourceFrameIndex"] = 0
    content = identity_observation_crop(crop_scene, "obs-1", media_root=tmp_path)
    assert cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    ) is not None


def test_roster_status_uses_structured_quality_not_warning_prose(tmp_path: Path):
    value, artifact_store = _scene_with_identity_artifact(tmp_path)
    snapshot = match_snapshot().model_copy(
        update={
            "payload": {
                "roster": [{"id": f"p{index}"} for index in range(12)],
                "rosterQuality": {},
                "sync": {
                    "warnings": [
                        "The free source returned only the first five lineup entries."
                    ]
                },
            }
        }
    )

    review = build_identity_review_projection(
        value,
        match_snapshot=snapshot,
        artifact_store=artifact_store,
    )

    assert review["matchSnapshot"]["roster"]["status"] == "review"
    assert review["matchSnapshot"]["roster"]["warnings"]


def test_identity_review_response_rejects_unknown_fields(tmp_path: Path):
    value, artifact_store = _scene_with_identity_artifact(tmp_path)
    payload = present_identity_review(
        build_identity_review_projection(
            value,
            match_snapshot=match_snapshot(),
            worker_health={"identity": {"status": "ready"}},
            artifact_store=artifact_store,
        ),
        project_id="project-review",
        scene_id="scene-identity",
    )
    assert IdentityReviewResponse.model_validate(payload).scene_id == "scene-identity"

    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        IdentityReviewResponse.model_validate(payload)


def test_identity_review_route_exposes_worker_readiness(monkeypatch, tmp_path):
    application = FastAPI()
    application.include_router(identity_review_router)
    value, artifact_store = _scene_with_identity_artifact(tmp_path)
    monkeypatch.setattr("app.identity_review_routes.scenes.get", lambda _: value)
    monkeypatch.setattr(
        "app.reconstruction_artifact_hydration.reconstruction_artifact_store",
        lambda: artifact_store,
    )
    monkeypatch.setattr(
        "app.reconstruction_identity_artifacts.reconstruction_artifact_store",
        lambda: artifact_store,
    )
    monkeypatch.setattr(
        "app.identity_review_routes.identity_worker_readiness",
        lambda **_: {"configured": True, "status": "ready", "backend": "reid"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.jersey_ocr_worker_readiness",
        lambda **_: {"configured": True, "status": "unavailable", "backend": None},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.project_matches.current_snapshot",
        lambda _: match_snapshot(),
    )

    response = _request(
        application,
        "GET",
        "/api/projects/project-test/scenes/scene-identity/identity-review",
    )

    assert response.status_code == 200
    assert response.json()["workers"] == {
        "reid": {"configured": True, "status": "ready", "backend": "reid"},
        "jerseyOcr": {
            "configured": True,
            "status": "unavailable",
            "backend": None,
        },
    }


def test_identity_review_route_returns_readiness_without_diagnostics_artifact(monkeypatch):
    application = FastAPI()
    application.include_router(identity_review_router)
    monkeypatch.setattr("app.identity_review_routes.scenes.get", lambda _: scene())
    monkeypatch.setattr(
        "app.identity_review_routes.identity_worker_readiness",
        lambda **_: {"configured": False, "status": "unavailable"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.jersey_ocr_worker_readiness",
        lambda **_: {"configured": False, "status": "unavailable"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.project_matches.current_snapshot",
        lambda _: match_snapshot(),
    )

    response = _request(
        application,
        "GET",
        "/api/projects/project-test/scenes/scene-identity/identity-review",
    )

    assert response.status_code == 200
    assert response.json()["availability"] == {
        "state": "not-started",
        "available": False,
    }
    assert response.json()["items"] == []


def test_identity_review_route_returns_503_for_missing_referenced_artifact(monkeypatch):
    application = FastAPI()
    application.include_router(identity_review_router)
    value = scene()
    value["payload"]["videoAsset"]["reconstruction"] = {
        "status": "ready",
        "artifactManifest": {
            "schemaVersion": 1,
            "artifacts": {
                "identityDiagnostics": {
                    "id": "sha256:" + "0" * 64,
                    "kind": "reconstruction.identity-diagnostics",
                    "schemaVersion": 1,
                    "uri": "artifact://sha256/" + "0" * 64,
                    "sha256": "0" * 64,
                    "byteSize": 1,
                    "contentType": "application/json",
                }
            },
        },
    }
    monkeypatch.setattr("app.identity_review_routes.scenes.get", lambda _: value)
    monkeypatch.setattr(
        "app.identity_review_routes.identity_worker_readiness",
        lambda **_: {"configured": False, "status": "unavailable"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.jersey_ocr_worker_readiness",
        lambda **_: {"configured": False, "status": "unavailable"},
    )
    monkeypatch.setattr(
        "app.identity_review_routes.project_matches.current_snapshot",
        lambda _: match_snapshot(),
    )

    response = _request(
        application,
        "GET",
        "/api/projects/project-test/scenes/scene-identity/identity-review",
    )

    assert response.status_code == 503
    assert "Referenced artifact is missing" in response.json()["detail"]


def test_owned_scene_reports_ownership_invariant_as_server_error(monkeypatch):
    def conflicting_owner(_: str) -> str:
        raise ProjectResourceConflict("Scene has multiple owning projects")

    monkeypatch.setattr(
        "app.identity_review_routes.project_resources.scene_owner",
        conflicting_owner,
    )

    with pytest.raises(HTTPException) as error:
        review_routes._owned_scene("project-test", "scene-identity")

    assert error.value.status_code == 500
    assert error.value.detail == "Scene ownership is inconsistent"
