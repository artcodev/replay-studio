import asyncio
from copy import deepcopy
from unittest.mock import patch

import httpx
from fastapi import FastAPI

import app.identity_decision_routes as routes
from app.identity_decision_routes import router
from app.project_match_persistence_contract import MatchSnapshotDocument


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    with patch.object(
        routes.project_resources,
        "scene_owner",
        return_value="project-test",
    ), patch.object(
        routes.project_matches,
        "current_snapshot",
        return_value=_match_snapshot(),
    ):
        return asyncio.run(_async_request(application, method, path, **kwargs))


def _match_snapshot() -> MatchSnapshotDocument:
    return MatchSnapshotDocument(
        id="snapshot-review",
        project_id="project-test",
        match_id="match-test",
        provider="test",
        schema_version=1,
        content_hash="sha256:review",
        is_current=True,
        payload={"roster": [{"id": "player-8", "name": "Player Eight"}]},
    )


def _scene() -> dict:
    return {
        "id": "scene-review",
        "revision": 3,
        "duration": 4.0,
        "payload": {
            "canonicalPeople": [
                {
                    "canonicalPersonId": "canonical-1",
                    "externalPlayerId": None,
                    "observations": [{"observationId": "obs-1"}],
                    "rosterCandidates": [{"externalPlayerId": "player-8"}],
                }
            ],
        },
    }


def test_reject_route_persists_a_durable_review_decision(monkeypatch):
    current = _scene()
    saved: list[dict] = []
    monkeypatch.setattr(
        "app.identity_decision_routes.scenes.get",
        lambda _: deepcopy(current),
    )
    monkeypatch.setattr(
        "app.identity_decision_routes.scenes.put",
        lambda value: saved.append(deepcopy(value)) or value,
    )
    application = FastAPI()
    application.include_router(router)

    response = _request(
        application,
        "POST",
        "/api/projects/project-test/scenes/scene-review/canonical-people/canonical-1/roster-rejections",
        json={"external_player_id": "player-8"},
    )

    assert response.status_code == 202
    assert saved[0]["payload"]["identityReviewDecisions"]["rosterRejections"][0][
        "externalPlayerId"
    ] == "player-8"


def test_reject_route_refuses_an_unpublished_player(monkeypatch):
    monkeypatch.setattr(
        "app.identity_decision_routes.scenes.get",
        lambda _: _scene(),
    )
    application = FastAPI()
    application.include_router(router)

    response = _request(
        application,
        "POST",
        "/api/projects/project-test/scenes/scene-review/canonical-people/canonical-1/roster-rejections",
        json={"external_player_id": "player-99"},
    )

    assert response.status_code == 422
    assert "published" in response.json()["detail"]


def _queueable_scene() -> dict:
    scene = _scene()
    scene["payload"]["videoAsset"] = {
        "id": "asset-review",
        "selectedSegmentId": "segment-review",
        "reconstruction": {"status": "ready"},
    }
    return scene


def _fake_queue(scene: dict, **_kwargs) -> dict:
    queued = deepcopy(scene)
    queued["payload"]["videoAsset"]["reconstruction"] = {
        "status": "queued",
        "runId": "run-review",
        "inputFingerprint": "sha256:review",
    }
    return queued


def test_identity_decision_only_persists_queue_for_dedicated_runner(monkeypatch):
    monkeypatch.setattr(
        "app.identity_decision_routes.scenes.get",
        lambda _: _queueable_scene(),
    )
    monkeypatch.setattr(
        "app.identity_decision_routes.queue_reconstruction",
        _fake_queue,
    )
    application = FastAPI()
    application.include_router(router)

    response = _request(
        application,
        "POST",
        "/api/projects/project-test/scenes/scene-review/canonical-people/canonical-1/roster-rejections",
        json={"external_player_id": "player-8"},
    )

    assert response.status_code == 202
    assert response.json()["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"
