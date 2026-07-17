from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.store as store_module
from app.database import Base
from app.main import _manual_match_bundle, app
from app.schemas import ManualMatchImportRequest
from app.store import SceneStore, reconstruction_input_fingerprint


def _scene(*, status: str = "ready") -> dict:
    scene = {
        "id": "manual-import-scene",
        "title": "Manual import",
        "version": 1,
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-manual",
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


def _request_body(*, player_count: int = 22) -> dict:
    players = []
    for index in range(player_count):
        home = index < (player_count + 1) // 2
        players.append(
            {
                "id": f"player-{index}",
                "name": f"Player {index}",
                "team_id": "team-home" if home else "team-away",
                "number": str(index + 1 if home else index + 1 - (player_count + 1) // 2),
                "position": "Midfielder",
                "lineup_role": "starter",
            }
        )
    return {
        "event": {
            "id": "manual-event-1",
            "name": "Manual Home vs Manual Away",
            "date": "2026-07-17",
            "league": "Community Cup",
        },
        "teams": {
            "home": {"id": "team-home", "name": "Manual Home"},
            "away": {"id": "team-away", "name": "Manual Away"},
        },
        "players": players,
        "provenance": {
            "label": "Official match sheet",
            "reference": "local://match-sheet.json",
            "capturedAt": "2026-07-17T12:30:00Z",
            "notes": "Noncommercial research import",
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


def test_manual_json_import_persists_v2_snapshot_and_queues_rebuild(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    isolated_store.put(_scene())
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])
    body = _request_body()
    body["timeline"] = [
        {
            "id": "timeline-sub-1",
            "minute": 61,
            "type": "substitution",
            "label": "Substitution",
            "player_id": "player-0",
            "secondary_player_id": "player-1",
        }
    ]
    body["substitutions"] = [
        {
            "id": "sub-1",
            "minute": 61,
            "player_out_id": "player-0",
            "player_in_id": "player-1",
            "label": "Substitution",
        }
    ]

    response = _request(
        "POST",
        "/api/scenes/manual-import-scene/match-binding/import",
        json=body,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    binding = payload["scene"]["payload"]["matchBinding"]
    assert binding["schemaVersion"] == 2
    assert binding["source"] == "manual"
    assert binding["eventId"] == "manual-event-1"
    assert binding["event"]["home"]["id"] == "team-home"
    assert len(binding["players"]) == 22
    assert len(binding["lineup"]) == 22
    assert binding["lineup"][0]["id"] == "manual-lineup-player-0"
    assert binding["timeline"][0]["player_name"] == "Player 0"
    assert binding["timeline"][0]["team_id"] == "team-home"
    assert binding["substitutions"][0]["player_in_name"] == "Player 1"
    assert binding["rosterQuality"] == {
        "status": "automatic-ready",
        "playerCount": 22,
        "homePlayerCount": 11,
        "awayPlayerCount": 11,
        "automaticIdentityEligible": True,
        "manualIdentityEligible": True,
        "reasons": [],
    }
    assert binding["fetchedAt"] == binding["provenance"]["importedAt"]
    assert binding["provenance"]["kind"] == "manual-json"
    assert binding["provenance"]["capturedAt"] == "2026-07-17T12:30:00+00:00"
    assert payload["bundle"]["source"] == "manual"
    assert payload["bundle"]["roster_quality"]["automatic_identity_eligible"] is True
    reconstruction = payload["scene"]["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "queued"
    assert reconstruction["runId"] != "run-current"
    assert reconstruction["inputFingerprint"] == reconstruction_input_fingerprint(
        payload["scene"]
    )


def test_bundled_spain_belgium_roster_stays_strict_and_automatic_ready() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "matches"
        / "spain-belgium-2026-qf.json"
    )
    request = ManualMatchImportRequest.model_validate(
        json.loads(fixture_path.read_text(encoding="utf-8"))
    )

    bundle, provenance = _manual_match_bundle(request)

    assert len(bundle.players) == 52
    assert len(bundle.lineup) == 52
    assert len(bundle.timeline) == 7
    assert len(bundle.substitutions) == 9
    assert bundle.roster_quality.automatic_identity_eligible is True
    assert provenance["kind"] == "manual-json"


def test_minimal_partial_manual_roster_remains_available_for_manual_binding(
    isolated_store: SceneStore,
    monkeypatch,
) -> None:
    isolated_store.put(_scene())
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [("frame", 0.0)])
    body = _request_body(player_count=2)

    response = _request(
        "POST",
        "/api/scenes/manual-import-scene/match-binding/import",
        json=body,
    )

    assert response.status_code == 200, response.text
    binding = response.json()["scene"]["payload"]["matchBinding"]
    assert binding["rosterQuality"]["automaticIdentityEligible"] is False
    assert binding["rosterQuality"]["manualIdentityEligible"] is True
    assert binding["rosterQuality"]["reasons"] == [
        "fewer-than-eleven-players-per-team"
    ]
    assert "provider-five-player-cap" not in binding["rosterQuality"]["reasons"]
    assert len(binding["players"]) == 2


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (lambda body: body.update({"rosterQuality": {}}), "extra_forbidden"),
        (
            lambda body: body["players"][0].update({"unexpected": True}),
            "extra_forbidden",
        ),
        (
            lambda body: body["players"][1].update({"id": "player-0"}),
            "Duplicate player id",
        ),
        (
            lambda body: body["players"][0].update({"team_id": "unknown-team"}),
            "references an unknown team",
        ),
    ],
)
def test_manual_import_is_strict_and_never_partially_persists(
    isolated_store: SceneStore,
    mutation,
    detail: str,
) -> None:
    initial = isolated_store.put(_scene())
    body = _request_body()
    mutation(body)

    response = _request(
        "POST",
        "/api/scenes/manual-import-scene/match-binding/import",
        json=body,
    )

    assert response.status_code == 422
    assert detail in response.text
    saved = isolated_store.get("manual-import-scene")
    assert saved is not None
    assert saved["revision"] == initial["revision"]
    assert saved["payload"].get("matchBinding") is None


def test_manual_import_rejects_running_reconstruction_before_mutation(
    isolated_store: SceneStore,
) -> None:
    isolated_store.put(_scene(status="processing"))

    response = _request(
        "POST",
        "/api/scenes/manual-import-scene/match-binding/import",
        json=_request_body(),
    )

    assert response.status_code == 409
    assert "Wait for reconstruction" in response.json()["detail"]


def test_manual_import_cannot_orphan_an_existing_roster_binding(
    isolated_store: SceneStore,
) -> None:
    scene = _scene()
    scene["payload"]["canonicalPeople"] = [
        {
            "id": "canonical-0",
            "canonicalPersonId": "canonical-0",
            "teamId": "home",
            "role": "player",
            "externalPlayerId": "player-0",
        }
    ]
    isolated_store.put(scene)
    body = _request_body()
    body["players"][0]["team_id"] = "team-away"
    body["players"][0]["number"] = "99"

    response = _request(
        "POST",
        "/api/scenes/manual-import-scene/match-binding/import",
        json=body,
    )

    assert response.status_code == 409
    assert "Unbind canonical roster player player-0" in response.json()["detail"]
    saved = isolated_store.get("manual-import-scene")
    assert saved is not None
    assert saved["payload"].get("matchBinding") is None
