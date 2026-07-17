import asyncio
from copy import deepcopy

import cv2
import httpx
import numpy as np

from app.main import app
from app.reconstruction import ReconstructionError


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
    return asyncio.run(_async_request(method, path, **kwargs))


def test_frame_annotation_api_passes_identity_merge_contract(monkeypatch):
    captured = {}
    scene = _scene()
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)

    def save(_scene, values, *, persist=True):
        captured.update(values)
        captured["persist"] = persist
        return {"sceneTime": 0.0}

    monkeypatch.setattr("app.main.upsert_frame_person_annotation", save)
    monkeypatch.setattr(
        "app.main.analyze_scene_frame",
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

    monkeypatch.setattr("app.main.queue_reconstruction", queue)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
        json={
            "annotation_id": "annotation-a",
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "home-player",
            "action": "merge",
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
    assert captured["persist"] is False
    assert captured["expected_scene_fingerprint"].startswith("sha256:")
    assert response.json()["reconstruction"]["status"] == "queued"
    assert response.json()["reconstruction"]["runId"] == "run-correction"


def test_frame_annotation_api_queues_split_range_atomically(monkeypatch):
    captured = {}
    scene = _scene()
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)

    def save(_scene, values, *, persist=True):
        captured.update(values)
        captured["persist"] = persist
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

    monkeypatch.setattr("app.main.upsert_frame_person_annotation", save)
    monkeypatch.setattr(
        "app.main.analyze_scene_frame",
        lambda *_: {"sceneId": scene["id"], "preview": "split"},
    )
    monkeypatch.setattr("app.main.queue_reconstruction", queue)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
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
    assert captured["persist"] is False
    assert captured["expected_scene_fingerprint"].startswith("sha256:")
    assert response.json()["reconstruction"]["runId"] == "run-split"


def test_frame_annotation_api_returns_validation_error_for_invalid_merge(monkeypatch):
    scene = _scene()
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)

    def reject(*_args, **_kwargs):
        raise ReconstructionError("A person cannot be merged into itself")

    monkeypatch.setattr("app.main.upsert_frame_person_annotation", reject)

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
        json={
            "annotation_id": "annotation-a",
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "home-player",
            "action": "merge",
            "merge_target_id": "annotation-a",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "A person cannot be merged into itself"


def test_frame_annotation_analysis_failure_does_not_queue_or_persist(monkeypatch):
    scene = _scene()
    calls = {"queue": 0}
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)

    def mutate(_scene, _values, *, persist=True):
        assert persist is False
        return {"sceneTime": 0.0}

    def fail_analysis(*_args):
        raise ReconstructionError("frame analysis failed")

    def queue(*_args, **_kwargs):
        calls["queue"] += 1
        raise AssertionError("queue must not run after failed analysis")

    monkeypatch.setattr("app.main.upsert_frame_person_annotation", mutate)
    monkeypatch.setattr("app.main.analyze_scene_frame", fail_analysis)
    monkeypatch.setattr("app.main.queue_reconstruction", queue)

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
        json={
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "home-player",
            "action": "confirm",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "frame analysis failed"
    assert calls["queue"] == 0


def test_legacy_ignore_api_defaults_to_observation_scope(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((120, 200, 3), dtype=np.uint8))
    scene = _scene()
    scene["payload"]["videoAsset"]["sourceStart"] = 0.0
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])

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

    monkeypatch.setattr("app.main.analyze_scene_frame", analyze)
    monkeypatch.setattr("app.main.queue_reconstruction", queue)
    monkeypatch.setattr("app.main.reconstruct_scene_by_id", lambda *_: None)

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
        json={
            "scene_time": 0,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 60},
            "kind": "ignore",
        },
    )

    assert response.status_code == 200
    annotation = response.json()["annotation"]
    assert annotation["action"] == "exclude"
    assert annotation["scope"] == "observation"
    assert annotation["sourceTrackId"] is None
    assert response.json()["reconstruction"]["runId"] == "legacy-ignore-run"


def test_explicit_identity_exclude_api_still_requires_source_track(monkeypatch, tmp_path):
    frame = tmp_path / "frame_00001.jpg"
    cv2.imwrite(str(frame), np.zeros((120, 200, 3), dtype=np.uint8))
    scene = _scene()
    scene["payload"]["videoAsset"]["sourceStart"] = 0.0
    monkeypatch.setattr("app.main.scene_store.get", lambda _: scene)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(frame, 0.0)])

    response = _request(
        "POST",
        "/api/scenes/identity-scene/frame-annotations",
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
