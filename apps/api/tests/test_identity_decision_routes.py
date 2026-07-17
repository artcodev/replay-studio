import asyncio
from copy import deepcopy

import httpx
from fastapi import FastAPI

from app.identity_decision_routes import router


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    return asyncio.run(_async_request(application, method, path, **kwargs))


def _scene() -> dict:
    return {
        "id": "scene-review",
        "revision": 3,
        "duration": 4.0,
        "payload": {
            "matchBinding": {
                "players": [{"id": "player-8", "name": "Player Eight"}],
            },
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
        "app.identity_decision_routes.scene_store.get",
        lambda _: deepcopy(current),
    )
    monkeypatch.setattr(
        "app.identity_decision_routes.scene_store.put",
        lambda value: saved.append(deepcopy(value)) or value,
    )
    application = FastAPI()
    application.include_router(router)

    response = _request(
        application,
        "POST",
        "/api/scenes/scene-review/canonical-people/canonical-1/roster-rejections",
        json={"external_player_id": "player-8"},
    )

    assert response.status_code == 202
    assert saved[0]["payload"]["identityReviewDecisions"]["rosterRejections"][0][
        "externalPlayerId"
    ] == "player-8"


def test_reject_route_refuses_an_unpublished_player(monkeypatch):
    monkeypatch.setattr(
        "app.identity_decision_routes.scene_store.get",
        lambda _: _scene(),
    )
    application = FastAPI()
    application.include_router(router)

    response = _request(
        application,
        "POST",
        "/api/scenes/scene-review/canonical-people/canonical-1/roster-rejections",
        json={"external_player_id": "player-99"},
    )

    assert response.status_code == 422
    assert "published" in response.json()["detail"]
