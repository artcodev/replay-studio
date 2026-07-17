import asyncio
from copy import deepcopy

import httpx
import pytest
from pydantic import ValidationError

from app.main import app
from app.player_actions import (
    PlayerActionError,
    delete_player_action,
    upsert_player_action,
)
from app.schemas import PlayerActionUpsertRequest


def _automatic_action(identifier: str = "suggestion-1") -> dict:
    return {
        "id": identifier,
        "canonicalPersonId": "canonical-away-8",
        "type": "pass",
        "startTime": 1.1,
        "endTime": 1.8,
        "keypoints": [{"kind": "contact", "time": 1.4}],
        "confidence": 0.73,
        "status": "suggested",
        "source": "automatic",
        "evidence": {"model": "action-baseline", "reasons": ["ball acceleration"]},
    }


def _scene() -> dict:
    return {
        "id": "action-scene",
        "title": "Action scene",
        "version": 1,
        "revision": 4,
        "duration": 5.0,
        "payload": {
            "videoAsset": {
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "ready"},
            },
            "canonicalPeople": [
                {"canonicalPersonId": "canonical-home-7", "displayName": "Home 7"},
                {"canonicalPersonId": "canonical-away-8", "displayName": "Away 8"},
            ],
            "tracks": [
                {"id": "home-track-1", "canonicalPersonId": "canonical-home-7"},
                {"id": "away-track-1", "canonicalPersonId": "canonical-away-8"},
            ],
            "playerActions": [_automatic_action()],
            "ball": {"keyframes": []},
        },
    }


def _manual_request(**overrides) -> dict:
    request = {
        "id": "manual-action-1",
        "canonical_person_id": "canonical-home-7",
        "type": "shot",
        "start_time": 2.0,
        "end_time": 3.0,
        "keypoints": [
            {"kind": "recovery", "time": 2.9},
            {"kind": "contact", "time": 2.50049},
            {"kind": "wind-up", "time": 2.1},
            {"kind": "contact", "time": 2.5004},
        ],
    }
    request.update(overrides)
    return request


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


def test_manual_action_is_normalized_and_preserves_automatic_suggestions(monkeypatch):
    scene = _scene()
    persisted = []
    monkeypatch.setattr(
        "app.player_actions.scene_store.put",
        lambda value: persisted.append(deepcopy(value)) or value,
    )

    action = upsert_player_action(scene, _manual_request())

    assert action == scene["payload"]["playerActions"][1]
    assert action["id"] == "manual-action-1"
    assert action["canonicalPersonId"] == "canonical-home-7"
    assert action["source"] == "manual"
    assert action["status"] == "confirmed"
    assert action["confidence"] == 1.0
    assert action["createdAt"] == action["updatedAt"]
    assert action["keypoints"] == [
        {"kind": "wind-up", "time": 2.1},
        {"kind": "contact", "time": 2.5},
        {"kind": "recovery", "time": 2.9},
    ]
    assert scene["payload"]["playerActions"][0] == _automatic_action()
    assert len(persisted) == 1


def test_generated_action_id_is_nonempty_and_stable_across_update():
    scene = _scene()
    request = _manual_request(id=None, keypoints=[])
    created = upsert_player_action(scene, request, persist=False)
    assert created["id"].startswith("action-")

    created_at = created["createdAt"]
    updated = upsert_player_action(
        scene,
        {
            **request,
            "id": created["id"],
            "type": "header",
            "start_time": 2.2,
            "end_time": 2.8,
        },
        persist=False,
    )
    assert updated["id"] == created["id"]
    assert updated["type"] == "header"
    assert updated["createdAt"] == created_at
    assert len(scene["payload"]["playerActions"]) == 2


def test_legacy_scene_track_id_is_a_valid_canonical_key():
    scene = _scene()
    scene["payload"].pop("canonicalPeople")
    scene["payload"]["tracks"] = [{"id": "legacy-player"}]

    action = upsert_player_action(
        scene,
        _manual_request(canonical_person_id="legacy-player"),
        persist=False,
    )
    assert action["canonicalPersonId"] == "legacy-player"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"canonical_person_id": "missing"}, "canonical person no longer exists"),
        ({"start_time": 3.0, "end_time": 3.0}, "start time must be before"),
        ({"start_time": 3.0004, "end_time": 3.00049}, "start time must be before"),
        ({"start_time": -0.1}, "between 0 and the scene duration"),
        ({"end_time": 5.1}, "between 0 and the scene duration"),
        (
            {"keypoints": [{"kind": "contact", "time": 1.9}]},
            "outside its action interval",
        ),
        ({"type": "teleport"}, "Unsupported player action type"),
        (
            {"keypoints": [{"kind": "unknown", "time": 2.5}]},
            "Unsupported player action keypoint kind",
        ),
        ({"id": "unsafe/action"}, "invalid format"),
    ],
)
def test_manual_action_domain_validation_is_fail_closed(overrides, message):
    with pytest.raises(PlayerActionError, match=message):
        upsert_player_action(
            _scene(),
            _manual_request(**overrides),
            persist=False,
        )


def test_action_id_cannot_change_owner():
    scene = _scene()
    upsert_player_action(scene, _manual_request(), persist=False)
    with pytest.raises(PlayerActionError, match="cannot be reassigned"):
        upsert_player_action(
            scene,
            _manual_request(canonical_person_id="canonical-away-8"),
            persist=False,
        )


def test_manual_endpoint_cannot_overwrite_or_delete_automatic_suggestion():
    scene = _scene()
    with pytest.raises(PlayerActionError, match="cannot be overwritten"):
        upsert_player_action(
            scene,
            _manual_request(id="suggestion-1", canonical_person_id="canonical-away-8"),
            persist=False,
        )
    with pytest.raises(PlayerActionError, match="cannot be deleted"):
        delete_player_action(scene, "suggestion-1", persist=False)
    assert scene["payload"]["playerActions"] == [_automatic_action()]


def test_corrupt_duplicate_saved_ids_are_rejected_without_mutation():
    scene = _scene()
    scene["payload"]["playerActions"].append(deepcopy(_automatic_action()))
    before = deepcopy(scene)
    with pytest.raises(PlayerActionError, match="ids are not unique"):
        upsert_player_action(scene, _manual_request(), persist=False)
    with pytest.raises(PlayerActionError, match="ids are not unique"):
        delete_player_action(scene, "suggestion-1", persist=False)
    assert scene == before


def test_delete_removes_only_requested_manual_action(monkeypatch):
    scene = _scene()
    upsert_player_action(scene, _manual_request(), persist=False)
    persisted = []
    monkeypatch.setattr(
        "app.player_actions.scene_store.put",
        lambda value: persisted.append(deepcopy(value)) or value,
    )

    deleted = delete_player_action(scene, "manual-action-1")

    assert deleted["source"] == "manual"
    assert scene["payload"]["playerActions"] == [_automatic_action()]
    assert persisted[0]["payload"]["playerActions"] == [_automatic_action()]


def test_request_schema_accepts_camel_and_snake_case_and_forbids_server_fields():
    camel = PlayerActionUpsertRequest.model_validate(
        {
            "id": "action.client-1",
            "canonicalPersonId": "canonical-home-7",
            "type": "slide-tackle",
            "startTime": 0.5,
            "endTime": 1.1,
            "keypoints": [{"kind": "impact", "time": 0.8}],
        }
    )
    assert camel.canonical_person_id == "canonical-home-7"
    snake = PlayerActionUpsertRequest.model_validate(_manual_request())
    assert snake.start_time == 2.0
    long_highlight = PlayerActionUpsertRequest.model_validate(
        {
            "canonicalPersonId": "canonical-home-7",
            "type": "run",
            "startTime": 140.0,
            "endTime": 150.0,
            "keypoints": [{"kind": "apex", "time": 145.0}],
        }
    )
    assert long_highlight.end_time == 150.0

    with pytest.raises(ValidationError):
        PlayerActionUpsertRequest.model_validate(
            {
                **_manual_request(),
                "source": "automatic",
            }
        )
    with pytest.raises(ValidationError):
        PlayerActionUpsertRequest.model_validate(
            _manual_request(start_time=float("nan"))
        )


def test_scene_duration_is_the_only_action_upper_time_bound():
    scene = _scene()
    scene["duration"] = 180.0
    action = upsert_player_action(
        scene,
        _manual_request(
            start_time=140.0,
            end_time=150.0,
            keypoints=[{"kind": "contact", "time": 145.0}],
        ),
        persist=False,
    )
    assert action["startTime"] == 140.0
    assert action["endTime"] == 150.0


def test_player_action_post_api_persists_camel_case_contract(monkeypatch):
    scene = _scene()
    persisted = []
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(scene))
    monkeypatch.setattr(
        "app.player_actions.scene_store.put",
        lambda value: persisted.append(deepcopy(value)) or value,
    )

    response = _request(
        "POST",
        "/api/scenes/action-scene/player-actions",
        json={
            "id": "client-action-8",
            "canonicalPersonId": "canonical-home-7",
            "type": "shot",
            "startTime": 2.0,
            "endTime": 3.0,
            "keypoints": [{"kind": "contact", "time": 2.5}],
        },
    )

    assert response.status_code == 200
    action = response.json()["payload"]["playerActions"][-1]
    assert action["canonicalPersonId"] == "canonical-home-7"
    assert action["startTime"] == 2.0
    assert action["status"] == "confirmed"
    assert persisted[0]["payload"]["playerActions"][-1] == action


@pytest.mark.parametrize("status", ["queued", "processing"])
@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_player_action_api_rejects_edits_during_reconstruction(
    monkeypatch,
    status,
    method,
):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["status"] = status
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)
    path = "/api/scenes/action-scene/player-actions"
    if method == "POST":
        response = _request(
            method,
            path,
            json={
                "id": "manual-action-1",
                "canonicalPersonId": "canonical-home-7",
                "type": "run",
                "startTime": 1.0,
                "endTime": 2.0,
            },
        )
    else:
        response = _request(method, f"{path}/manual-action-1")
    assert response.status_code == 409
    assert "reconstruction" in response.json()["detail"].lower()


def test_player_action_delete_api_returns_updated_scene(monkeypatch):
    scene = _scene()
    upsert_player_action(scene, _manual_request(), persist=False)
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)
    monkeypatch.setattr("app.player_actions.scene_store.put", lambda value: value)

    response = _request(
        "DELETE",
        "/api/scenes/action-scene/player-actions/manual-action-1",
    )
    assert response.status_code == 200
    assert response.json()["payload"]["playerActions"] == [_automatic_action()]


def test_player_action_api_reports_missing_scene_person_and_action(monkeypatch):
    monkeypatch.setattr("app.main.scene_store.get", lambda _: None)
    missing_scene = _request(
        "POST",
        "/api/scenes/missing/player-actions",
        json={
            "canonicalPersonId": "canonical-home-7",
            "type": "run",
            "startTime": 1.0,
            "endTime": 2.0,
        },
    )
    assert missing_scene.status_code == 404

    scene = _scene()
    monkeypatch.setattr("app.main.scene_store.get", lambda _: deepcopy(scene))
    monkeypatch.setattr("app.player_actions.scene_store.put", lambda value: value)
    missing_person = _request(
        "POST",
        "/api/scenes/action-scene/player-actions",
        json={
            "canonicalPersonId": "canonical-missing",
            "type": "run",
            "startTime": 1.0,
            "endTime": 2.0,
        },
    )
    assert missing_person.status_code == 404
    missing_action = _request(
        "DELETE",
        "/api/scenes/action-scene/player-actions/manual-missing",
    )
    assert missing_action.status_code == 404
