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


SCENE_PATH = "/api/projects/project-test/scenes/mutation-guard-scene"


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
            "eventBindings": [],
            "tracks": [{"id": "auto-home-02", "label": "Home track 02", "number": 2}],
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


def test_the_whole_document_write_path_no_longer_exists(
    isolated_store: SceneRepository,
) -> None:
    # The legacy path was removed in the same cutover that introduced the
    # dedicated commands: no client may publish a scene document again.
    current = isolated_store.put(_scene())
    response = _request("PUT", SCENE_PATH, json=current)
    assert response.status_code == 405


def test_a_stale_editor_saves_without_clobbering_a_concurrent_change(
    isolated_store: SceneRepository,
) -> None:
    initial = isolated_store.put(_scene())

    # Something else advances the scene while the editor holds an older copy.
    concurrent = isolated_store.get(initial["id"])
    assert concurrent is not None
    concurrent["title"] = "Concurrent title"
    isolated_store.put(concurrent)

    # The stale editor edits an unrelated domain. Under the old whole-document
    # write this either lost the revision race or reverted the concurrent
    # title; a command touches only its own field, so it does neither.
    response = _request(
        "PUT",
        f"{SCENE_PATH}/tracks/auto-home-02/metadata",
        json={"number": 9},
    )

    assert response.status_code == 200
    saved = isolated_store.get(initial["id"])
    assert saved is not None
    assert saved["payload"]["tracks"][0]["number"] == 9
    assert saved["title"] == "Concurrent title"


def test_commands_still_refuse_to_edit_a_running_reconstruction(
    isolated_store: SceneRepository,
) -> None:
    processing = deepcopy(_scene(status="processing"))
    isolated_store.put(processing)

    for path, body in (
        ("/title", {"title": "Renamed"}),
        ("/event-bindings", {"bindings": []}),
        ("/tracks/auto-home-02/metadata", {"number": 7}),
    ):
        response = _request("PUT", f"{SCENE_PATH}{path}", json=body)
        assert response.status_code == 409, path
        assert "Wait for reconstruction" in response.json()["detail"]


def test_commands_fail_closed_on_targets_a_rebuild_retired(
    isolated_store: SceneRepository,
) -> None:
    isolated_store.put(_scene())
    response = _request(
        "PUT",
        f"{SCENE_PATH}/tracks/auto-away-99/metadata",
        json={"number": 4},
    )
    assert response.status_code == 409
    assert "Unknown track" in response.json()["detail"]
