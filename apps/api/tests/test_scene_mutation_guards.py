from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.scene_repository as scene_repository_module
from app.database import Base
from app.main import app
from app.scene_document import reconstruction_input_fingerprint
from app.scene_repository import SceneRepository


def _scene(*, status: str = "ready") -> dict:
    scene = {
        "id": "mutation-guard-scene",
        "title": "Mutation guard",
        "version": 1,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "selectedSegmentId": "segment-1",
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "analysisFps": 10.0,
                "processingState": "tracks-ready",
                "reconstruction": {
                    "status": status,
                    "model": "yolo26m.pt",
                    "runId": "run-current",
                    "runRevision": 3,
                    "frameAnnotations": [],
                },
            },
            "teams": [
                {"id": "home", "name": "Old Home", "color": "#f00"},
                {"id": "away", "name": "Old Away", "color": "#00f"},
            ],
            "canonicalPeople": [],
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(scene)
    return scene


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


@pytest.fixture
def isolated_store(monkeypatch) -> SceneRepository:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(scene_repository_module, "SessionLocal", session_local)
    monkeypatch.setattr(
        "app.project_resource_access.project_resources.scene_owner",
        lambda scene_id: "project-test" if scene_id == "mutation-guard-scene" else None,
    )
    monkeypatch.setattr(
        "app.project_resource_access.project_matches.current_summary",
        lambda _project_id: None,
    )
    return SceneRepository()


def test_whole_scene_put_rejects_stale_or_running_reconstruction(
    isolated_store: SceneRepository,
) -> None:
    current = isolated_store.put(_scene())
    stale = deepcopy(current)
    stale["payload"]["videoAsset"]["reconstruction"]["runId"] = "run-stale"

    stale_response = _request(
        "PUT",
        "/api/projects/project-test/scenes/mutation-guard-scene",
        json=stale,
    )
    assert stale_response.status_code == 409
    assert "reconstruction changed" in stale_response.json()["detail"]

    processing = deepcopy(current)
    processing["payload"]["videoAsset"]["reconstruction"]["status"] = "processing"
    isolated_store.put(processing)
    running_response = _request(
        "PUT",
        "/api/projects/project-test/scenes/mutation-guard-scene",
        json=processing,
    )
    assert running_response.status_code == 409
    assert "Wait for reconstruction" in running_response.json()["detail"]


def test_whole_scene_put_rejects_sequential_stale_client_on_unrelated_change(
    isolated_store: SceneRepository,
) -> None:
    initial = isolated_store.put(_scene())
    stale_client = deepcopy(initial)

    concurrent = isolated_store.get(initial["id"])
    assert concurrent is not None
    concurrent["title"] = "Concurrent title"
    isolated_store.put(concurrent)

    # Runtime fields and reconstruction inputs are intentionally identical;
    # only the document revision can detect this otherwise invisible stale PUT.
    stale_client["payload"]["tracks"] = [{"id": "stale-client-track"}]
    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/mutation-guard-scene",
        json=stale_client,
    )

    assert response.status_code == 409
    assert "reload and retry" in response.json()["detail"]
    saved = isolated_store.get(initial["id"])
    assert saved is not None
    assert saved["title"] == "Concurrent title"
    assert saved["payload"]["tracks"] == []
