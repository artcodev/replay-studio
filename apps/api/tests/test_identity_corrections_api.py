import asyncio
from copy import deepcopy
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi import HTTPException

from app.main import app
import app.project_resource_access as resource_access
import app.scene_identity_routes as scene_identity_routes
from app.reconstruction_errors import ReconstructionError


def _scene():
    return {
        "id": "identity-scene",
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "ready"},
            }
        },
    }


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
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
        return_value=None,
    ):
        return asyncio.run(_async_request(method, path, **kwargs))


def test_frame_annotation_api_passes_identity_merge_contract(monkeypatch):
    captured = {}
    scene = _scene()
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)

    def save(_scene, values):
        captured.update(values)
        return {"sceneTime": 0.0}

    monkeypatch.setattr("app.scene_identity_routes.draft_frame_person_annotation_upsert", save)
    monkeypatch.setattr(
        "app.scene_identity_routes.analyze_scene_frame",
        lambda *_: {"sceneId": scene["id"], "preview": "merged"},
    )
    def queue(value, **kwargs):
        captured["expected_scene_fingerprint"] = kwargs.get("expected_scene_fingerprint")
        queued = deepcopy(value)
        queued["payload"]["videoAsset"]["reconstruction"] = {
            "status": "queued",
            "model": "yolo26n.pt",
            "runId": "run-correction",
            "runRevision": 2,
            "inputFingerprint": "sha256:new",
        }
        return queued

    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "annotation_id": "annotation-a",
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
                "kind": "home-player",
                "action": "merge",
                "scope": "identity",
                "merge_target_id": "auto-home-02",
            "source_track_id": "auto-home-01",
        },
    )

    assert response.status_code == 200
    assert response.json()["preview"] == "merged"
    assert captured["action"] == "merge"
    assert captured["scope"] == "identity"
    assert captured["merge_target_id"] == "auto-home-02"
    assert captured["source_track_id"] == "auto-home-01"
    assert captured["expected_scene_fingerprint"].startswith("sha256:")
    assert response.json()["reconstruction"]["status"] == "queued"
    assert response.json()["reconstruction"]["runId"] == "run-correction"


def test_frame_annotation_api_queues_split_range_atomically(monkeypatch):
    captured = {}
    scene = _scene()
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)

    def save(_scene, values):
        captured.update(values)
        return {"sceneTime": 1.5}

    def queue(value, **kwargs):
        captured["expected_scene_fingerprint"] = kwargs.get("expected_scene_fingerprint")
        queued = deepcopy(value)
        queued["payload"]["videoAsset"]["reconstruction"] = {
            "status": "queued",
            "model": "yolo26m.pt",
            "runId": "run-split",
            "runRevision": 7,
            "inputFingerprint": "sha256:split",
        }
        return queued

    monkeypatch.setattr("app.scene_identity_routes.draft_frame_person_annotation_upsert", save)
    monkeypatch.setattr(
        "app.scene_identity_routes.analyze_scene_frame",
        lambda *_: {"sceneId": scene["id"], "preview": "split"},
    )
    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "annotation_id": "split-a",
            "scene_time": 1.5,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "home-player",
            "action": "split",
            "scope": "range",
            "canonical_person_id": "canonical-a",
            "target_observation_id": "observation-stable-a",
            "range_start": 1.5,
            "range_end": 3.0,
        },
    )

    assert response.status_code == 200
    assert captured["action"] == "split"
    assert captured["scope"] == "range"
    assert captured["canonical_person_id"] == "canonical-a"
    assert captured["target_observation_id"] == "observation-stable-a"
    assert captured["range_start"] == 1.5
    assert captured["range_end"] == 3.0
    assert captured["expected_scene_fingerprint"].startswith("sha256:")
    assert response.json()["reconstruction"]["runId"] == "run-split"


def test_frame_annotation_api_returns_validation_error_for_invalid_merge(monkeypatch):
    scene = _scene()
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)

    def reject(*_args, **_kwargs):
        raise ReconstructionError("A person cannot be merged into itself")

    monkeypatch.setattr("app.scene_identity_routes.draft_frame_person_annotation_upsert", reject)

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "annotation_id": "annotation-a",
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
                "kind": "home-player",
                "action": "merge",
                "scope": "identity",
                "merge_target_id": "annotation-a",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "A person cannot be merged into itself"


def test_frame_annotation_analysis_failure_does_not_queue_or_persist(monkeypatch):
    scene = _scene()
    calls = {"queue": 0}
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)

    def mutate(_scene, _values):
        return {"sceneTime": 0.0}

    def fail_analysis(*_args):
        raise ReconstructionError("frame analysis failed")

    def queue(*_args, **_kwargs):
        calls["queue"] += 1
        raise AssertionError("queue must not run after failed analysis")

    monkeypatch.setattr("app.scene_identity_routes.draft_frame_person_annotation_upsert", mutate)
    monkeypatch.setattr("app.scene_identity_routes.analyze_scene_frame", fail_analysis)
    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
                "kind": "home-player",
                "action": "confirm",
                "scope": "identity",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "frame analysis failed"
    assert calls["queue"] == 0


def test_identity_correction_api_requires_explicit_action_and_scope(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((120, 200, 3), dtype=np.uint8))
    scene = _scene()
    scene["payload"]["videoAsset"]["sourceStart"] = 0.0
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)]
    )

    def analyze(value, _scene_time):
        annotation = value["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"][0]
        return {"sceneId": value["id"], "annotation": annotation}

    def queue(value, **_kwargs):
        queued = deepcopy(value)
        queued["payload"]["videoAsset"]["reconstruction"].update(
            {
                "status": "queued",
                "runId": "legacy-ignore-run",
                "runRevision": 1,
                "inputFingerprint": "sha256:legacy-ignore",
            }
        )
        return queued

    monkeypatch.setattr("app.scene_identity_routes.analyze_scene_frame", analyze)
    monkeypatch.setattr("app.scene_identity_routes.queue_reconstruction", queue)

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "ignore",
        },
    )

    assert response.status_code == 422
    missing_fields = {item["loc"][-1] for item in response.json()["detail"]}
    assert missing_fields == {"action", "scope"}


def test_explicit_identity_exclude_api_still_requires_source_track(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((120, 200, 3), dtype=np.uint8))
    scene = _scene()
    scene["payload"]["videoAsset"]["sourceStart"] = 0.0
    monkeypatch.setattr("app.project_resource_access.scenes.get", lambda _: scene)
    monkeypatch.setattr(
        "app.reconstruction_frame_annotation_target.frame_paths", lambda _: [(frame, 0.0)]
    )

    response = _request(
        "POST",
        "/api/projects/project-test/scenes/identity-scene/frame-annotations",
        json={
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "ignore",
            "action": "exclude",
            "scope": "identity",
        },
    )

    assert response.status_code == 422
    assert "tracked identity" in response.json()["detail"]
