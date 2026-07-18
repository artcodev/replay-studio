import asyncio
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from app.main import app
import app.project_resource_access as resource_access
from app.reconstruction_errors import ReconstructionError
from app.reconstruction_ball_trajectory import (
    edit_scene_ball_trajectory,
    publish_automatic_ball_trajectory as _publish_automatic_ball_trajectory,
)
from app.reconstruction_ball_trajectory_command import (
    set_scene_ball_trajectory,
)
from app.reconstruction_artifact_hydration import hydrate_scene_reconstruction


def _scene() -> dict:
    return {
        "id": "manual-ball-scene",
        "title": "Manual ball",
        "version": 1,
        "duration": 5.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "videoAsset": {
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "ready"},
            },
            "ball": {
                "mode": "automatic",
                "automaticKeyframes": [
                    {"t": 0.0, "x": -2.0, "y": 0.22, "z": 1.0, "confidence": 0.7},
                    {"t": 1.0, "x": 0.0, "y": 0.22, "z": 2.0, "confidence": 0.8},
                ],
                "manualKeyframes": [],
                "automaticDiagnostics": {"observedCoverage": 0.5},
                "keyframes": [
                    {"t": 0.0, "x": -2.0, "y": 0.22, "z": 1.0, "confidence": 0.7},
                    {"t": 1.0, "x": 0.0, "y": 0.22, "z": 2.0, "confidence": 0.8},
                ],
                "diagnostics": {"observedCoverage": 0.5},
            },
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

    with patch.object(resource_access, "project_scene_or_404", side_effect=owned_scene):
        return asyncio.run(_async_request(method, path, **kwargs))


def test_automatic_track_survives_manual_edit_and_mode_switch():
    scene = _scene()
    original = deepcopy(scene["payload"]["ball"]["keyframes"])

    edit_scene_ball_trajectory(scene, "manual", [])
    ball = scene["payload"]["ball"]
    assert ball["mode"] == "manual"
    assert ball["keyframes"] == []
    assert ball["manualKeyframes"] == []
    assert ball["automaticKeyframes"] == original

    edit_scene_ball_trajectory(
        scene,
        "manual",
        [{"t": 3.0, "x": 12.0, "z": -4.0}],
    )
    manual = deepcopy(scene["payload"]["ball"]["keyframes"])
    assert manual[0]["id"] == "manual-ball-003000"

    edit_scene_ball_trajectory(scene, "automatic")
    ball = scene["payload"]["ball"]
    assert ball["keyframes"] == original
    assert ball["manualKeyframes"] == manual
    assert ball["diagnostics"]["trajectoryMode"] == "automatic"


def test_manual_keyframes_are_sorted_deduplicated_and_authoritative():
    scene = _scene()
    edit_scene_ball_trajectory(
        scene,
        "manual",
        [
            {"t": 2.0, "x": 10.0, "z": 3.0, "y": 1.2},
            {"t": 1.0004, "x": 4.0, "z": 5.0},
            # Millisecond-normalized timestamps are deduplicated; the latest
            # supplied value is authoritative.
            {"t": 1.00049, "x": 6.0, "z": 7.0},
        ],
    )

    ball = scene["payload"]["ball"]
    assert [item["t"] for item in ball["manualKeyframes"]] == [1.0, 2.0]
    assert ball["manualKeyframes"][0]["x"] == 6.0
    assert ball["manualKeyframes"][0]["id"] == "manual-ball-001000"
    assert ball["manualKeyframes"][0]["y"] == 0.22
    assert ball["manualKeyframes"][0]["confidence"] == 1.0
    assert ball["manualKeyframes"][0]["confidenceKind"] == "manual-authoritative"
    assert ball["manualKeyframes"][0]["provenance"] == {
        "source": "manual",
        "method": "user-pitch-keypoint",
    }
    assert ball["manualKeyframes"][1]["heightSource"] == "manual"
    assert ball["diagnostics"]["source"] == "manual-keypoints"
    assert ball["diagnostics"]["interpolationSegmentCount"] == 1
    assert ball["diagnostics"]["observedCoverage"] is None


@pytest.mark.parametrize(
    "keyframe, message",
    [
        ({"t": 5.01, "x": 0, "z": 0}, "time must be between"),
        ({"t": 1, "x": 52.51, "z": 0}, "x must be within"),
        ({"t": 1, "x": 0, "z": -34.01}, "z must be within"),
        ({"t": 1, "x": float("nan"), "z": 0}, "x must be a finite"),
        ({"t": 1, "x": 0, "z": 0, "y": float("inf")}, "y must be a finite"),
    ],
)
def test_manual_keyframe_validation_is_fail_closed(keyframe, message):
    with pytest.raises(ReconstructionError, match=message):
        edit_scene_ball_trajectory(
            _scene(),
            "manual",
            [keyframe],
        )


def test_automatic_reconstruction_updates_automatic_track_without_overwriting_manual():
    scene = _scene()
    edit_scene_ball_trajectory(
        scene,
        "manual",
        [{"t": 2.0, "x": 8.0, "z": -3.0}],
    )
    manual = deepcopy(scene["payload"]["ball"]["keyframes"])
    latest_automatic = [{"t": 0.5, "x": -10.0, "y": 0.22, "z": 4.0}]

    published = _publish_automatic_ball_trajectory(
        scene,
        latest_automatic,
        {"observedCoverage": 1.0, "worldProjectionStatus": "published"},
    )

    assert published["mode"] == "manual"
    assert published["keyframes"] == manual
    assert published["manualKeyframes"] == manual
    assert published["automaticKeyframes"] == latest_automatic
    assert published["diagnostics"]["trajectoryMode"] == "manual"
    assert published["automaticDiagnostics"]["trajectoryMode"] == "automatic"
    assert published["automaticDiagnostics"]["observedCoverage"] == 1.0

    edit_scene_ball_trajectory(scene, "automatic")
    assert scene["payload"]["ball"]["keyframes"] == latest_automatic


def test_ball_trajectory_api_persists_manual_contract(monkeypatch):
    scene = _scene()
    persisted = []
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)
    monkeypatch.setattr(
        "app.reconstruction_ball_trajectory_command.scenes.put",
        lambda value: persisted.append(deepcopy(value)) or value,
    )

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/manual-ball-scene/ball-trajectory",
        json={
            "mode": "manual",
            "keyframes": [
                {"t": 4, "x": 20, "z": 10},
                {"t": 1, "x": -20, "z": -10, "y": 2.5},
            ],
        },
    )

    assert response.status_code == 200
    ball = response.json()["payload"]["ball"]
    assert ball["mode"] == "manual"
    assert "keyframes" not in ball
    assert ball["keyframeCount"] == 2
    assert len(persisted) == 1
    hydrate_scene_reconstruction(persisted[0], names=("ballTrajectory",))
    assert [item["t"] for item in persisted[0]["payload"]["ball"]["keyframes"]] == [
        1.0,
        4.0,
    ]


def test_ball_trajectory_api_rejects_keyframes_in_automatic_mode(monkeypatch):
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: _scene())

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/manual-ball-scene/ball-trajectory",
        json={"mode": "automatic", "keyframes": [{"t": 1, "x": 0, "z": 0}]},
    )

    assert response.status_code == 422
    assert "only be supplied for manual" in response.json()["detail"]


def test_ball_trajectory_api_rejects_edits_during_reconstruction(monkeypatch):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["status"] = "processing"
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)

    response = _request(
        "PUT",
        "/api/projects/project-test/scenes/manual-ball-scene/ball-trajectory",
        json={"mode": "manual", "keyframes": []},
    )

    assert response.status_code == 409
