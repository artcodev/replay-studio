from __future__ import annotations

import asyncio
from copy import deepcopy

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.store as store_module
from app.database import Base
from app.main import app
from app.schemas import EventBundle
from app.store import SceneStore, reconstruction_input_fingerprint


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


def _bundle(*, include_player_8: bool = True) -> EventBundle:
    players = [
        {
            "id": "player-8",
            "name": "Player Eight",
            "team_id": "team-home",
            "team_name": "New Home",
            "number": "8",
        }
    ] if include_player_8 else []
    return EventBundle.model_validate(
        {
            "source": "thesportsdb",
            "event": {
                "id": "event-new",
                "name": "New Home vs New Away",
                "home": {"id": "team-home", "name": "New Home"},
                "away": {"id": "team-away", "name": "New Away"},
            },
            "players": players,
            "timeline": [],
            "fetched_at": "2026-07-17T12:00:00Z",
            "warnings": [],
        }
    )


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


def test_match_binding_rejects_running_reconstruction(
    isolated_store: SceneStore,
) -> None:
    isolated_store.put(_scene(status="processing"))

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding",
        json={"event_id": "event-new"},
    )

    assert response.status_code == 409
    assert "Wait for reconstruction" in response.json()["detail"]


def test_match_binding_queues_a_new_guarded_reconstruction(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    scene = _scene()
    isolated_store.put(scene)
    async def event_bundle(_event_id: str):
        return _bundle()
    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding",
        json={"event_id": "event-new"},
    )

    assert response.status_code == 200
    saved = response.json()["scene"]
    reconstruction = saved["payload"]["videoAsset"]["reconstruction"]
    assert saved["payload"]["matchBinding"]["eventId"] == "event-new"
    assert reconstruction["status"] == "queued"
    assert reconstruction["runId"] != "run-current"
    assert reconstruction["inputFingerprint"] == reconstruction_input_fingerprint(saved)


def test_match_binding_refresh_migrates_legacy_snapshot_with_same_guards(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    scene = _scene()
    scene["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "event-new",
        "fetchedAt": None,
    }
    scene["payload"]["videoAsset"]["reconstruction"][
        "inputFingerprint"
    ] = reconstruction_input_fingerprint(scene)
    isolated_store.put(scene)
    requested: list[str] = []

    async def event_bundle(event_id: str):
        requested.append(event_id)
        return _bundle()

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding/refresh",
    )

    assert response.status_code == 200
    assert requested == ["event-new"]
    saved = response.json()["scene"]
    binding = saved["payload"]["matchBinding"]
    assert binding["schemaVersion"] == 2
    assert binding["event"]["name"] == "New Home vs New Away"
    assert binding["players"][0]["id"] == "player-8"
    assert binding["lineup"] == []
    assert binding["timeline"] == []
    assert binding["substitutions"] == []
    assert binding["rosterQuality"]["automaticIdentityEligible"] is False
    assert binding["rosterQuality"]["manualIdentityEligible"] is True
    assert saved["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"


def test_match_binding_refresh_requires_a_legacy_event_id(
    isolated_store: SceneStore,
) -> None:
    scene = _scene()
    scene["payload"]["matchBinding"] = {"source": "thesportsdb"}
    isolated_store.put(scene)

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding/refresh",
    )

    assert response.status_code == 409
    assert "no saved match event id" in response.json()["detail"]


def test_match_binding_refresh_rejects_running_reconstruction_before_fetch(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    scene = _scene(status="processing")
    scene["payload"]["matchBinding"] = {
        "source": "thesportsdb",
        "eventId": "event-new",
        "fetchedAt": None,
    }
    isolated_store.put(scene)

    async def event_bundle(_event_id: str):
        raise AssertionError("provider must not be called while reconstruction is running")

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding/refresh",
    )

    assert response.status_code == 409
    assert "Wait for reconstruction" in response.json()["detail"]


def test_match_binding_rejects_roster_that_orphans_manual_binding(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    scene = _scene()
    scene["payload"]["canonicalPeople"] = [
        {
            "id": "canonical-8",
            "canonicalPersonId": "canonical-8",
            "teamId": "home",
            "role": "player",
            "externalPlayerId": "player-8",
        }
    ]
    isolated_store.put(scene)
    async def event_bundle(_event_id: str):
        return _bundle(include_player_8=False)
    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding",
        json={"event_id": "event-new"},
    )

    assert response.status_code == 409
    assert "Unbind canonical roster player player-8" in response.json()["detail"]
    assert isolated_store.get(scene["id"])["payload"].get("matchBinding") is None


def test_match_binding_checks_durable_binding_when_published_people_are_stale(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["frameAnnotations"] = [
        {
            "id": "roster-binding-stale-output",
            "sceneTime": 0.0,
            "frameIndex": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "home-player",
            "action": "confirm",
            "scope": "identity",
            "canonicalPersonId": "canonical-stale",
            "externalPlayerId": "player-8",
            "correctionKind": "canonical-roster-binding-v1",
            "rosterBindingState": "bound",
        }
    ]
    reconstruction["inputFingerprint"] = reconstruction_input_fingerprint(scene)
    isolated_store.put(scene)

    async def event_bundle(_event_id: str):
        return _bundle(include_player_8=False)

    monkeypatch.setattr("app.main.sports_provider.event_bundle", event_bundle)

    response = _request(
        "POST",
        "/api/scenes/mutation-guard-scene/match-binding",
        json={"event_id": "event-new"},
    )

    assert response.status_code == 409
    assert "Unbind canonical roster player player-8" in response.json()["detail"]
    assert isolated_store.get(scene["id"])["payload"].get("matchBinding") is None


def test_whole_scene_put_rejects_stale_or_running_reconstruction(
    isolated_store: SceneStore,
) -> None:
    current = _scene()
    isolated_store.put(current)
    stale = deepcopy(current)
    stale["payload"]["videoAsset"]["reconstruction"]["runId"] = "run-stale"

    stale_response = _request(
        "PUT",
        "/api/scenes/mutation-guard-scene",
        json=stale,
    )
    assert stale_response.status_code == 409
    assert "reconstruction changed" in stale_response.json()["detail"]

    processing = deepcopy(current)
    processing["payload"]["videoAsset"]["reconstruction"]["status"] = "processing"
    isolated_store.put(processing)
    running_response = _request(
        "PUT",
        "/api/scenes/mutation-guard-scene",
        json=processing,
    )
    assert running_response.status_code == 409
    assert "Wait for reconstruction" in running_response.json()["detail"]


def test_whole_scene_put_rejects_sequential_stale_client_on_unrelated_change(
    isolated_store: SceneStore,
) -> None:
    initial = _scene()
    isolated_store.put(initial)
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
        "/api/scenes/mutation-guard-scene",
        json=stale_client,
    )

    assert response.status_code == 409
    assert "reload and retry" in response.json()["detail"]
    saved = isolated_store.get(initial["id"])
    assert saved is not None
    assert saved["title"] == "Concurrent title"
    assert saved["payload"]["tracks"] == []
