import json
from types import SimpleNamespace

import httpx
import pytest

from app.identity_worker import (
    IdentityWorkerError,
    embed_identity_frames,
    identity_worker_readiness,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _settings(url="http://identity-worker:8091", batch_size=2):
    return SimpleNamespace(
        identity_worker_url=url,
        identity_worker_timeout=900,
        identity_worker_batch_size=batch_size,
    )


def _embedding():
    vector = [0.0] * 256
    vector[0] = 1.0
    return vector


def _quality():
    return {
        "cropWidth": 30,
        "cropHeight": 60,
        "sourceBoxWidth": 30.0,
        "sourceBoxHeight": 60.0,
        "borderClipped": False,
        "sharpness": 42.0,
    }


def _fingerprint(identifier: str = "crop") -> str:
    return f"pixel-evidence-v1:{identifier}"


def test_readiness_is_disabled_without_url(monkeypatch):
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings(url=""))
    assert identity_worker_readiness() == {
        "configured": False,
        "status": "disabled",
        "backend": None,
    }


def test_readiness_requires_loaded_normalized_model(monkeypatch):
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker.httpx.get",
        lambda url, timeout: FakeResponse(
            {
                "status": "ready",
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "device": "cpu",
                "batchSize": 8,
                "modelVersion": "model-v1",
                "modelLoadSeconds": 4.2,
                "soccerNetCommit": "reference-commit",
            }
        ),
    )
    result = identity_worker_readiness(timeout=1.5)
    assert result == {
        "configured": True,
        "status": "ready",
        "backend": "prtreid-bpbreid-soccernet",
        "device": "cpu",
        "batchSize": 8,
        "dimension": 256,
        "normalized": True,
        "evidenceFingerprintVersion": "pixel-evidence-v1",
        "modelVersion": "model-v1",
        "modelLoadSeconds": 4.2,
        "soccerNetCommit": "reference-commit",
    }


def test_readiness_is_nonfatal_when_worker_is_offline(monkeypatch):
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())

    def offline(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("app.identity_worker.httpx.get", offline)
    result = identity_worker_readiness()
    assert result["status"] == "unavailable"
    assert "offline" in result["detail"]


def test_client_batches_frames_and_preserves_rejected_items(monkeypatch, tmp_path):
    frames = []
    for index in range(1, 4):
        path = tmp_path / f"frame_{index:05d}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(
            (
                index,
                path,
                [
                    {
                        "observationId": f"track-1:{index}",
                        "bbox": {"x": 1, "y": 2, "width": 30, "height": 60},
                    }
                ],
            )
        )
    calls = []

    def fake_post(url, data, files, timeout):
        manifest = json.loads(data["manifest"])
        calls.append([item["frameIndex"] for item in manifest["frames"]])
        items = []
        for frame in manifest["frames"]:
            observation_id = frame["observations"][0]["observationId"]
            usable = frame["frameIndex"] != 2
            items.append(
                {
                    "observationId": observation_id,
                    "frameIndex": frame["frameIndex"],
                    "usable": usable,
                    "quality": _quality(),
                    "rejectionReasons": [] if usable else ["crop-too-blurry"],
                    "embedding": _embedding() if usable else None,
                    "visibilityScores": None,
                    "role": None,
                    "roleConfidence": None,
                    "evidenceFingerprint": _fingerprint(observation_id),
                }
            )
        return FakeResponse(
            {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "modelVersion": "model-v1",
                "items": items,
                "diagnostics": {
                    "requestedObservationCount": len(items),
                    "usableObservationCount": sum(item["usable"] for item in items),
                    "rejectedObservationCount": sum(not item["usable"] for item in items),
                    "cacheHitCount": 1,
                    "cacheMissCount": max(0, len(items) - 1),
                    "deduplicatedObservationCount": 0,
                    "concurrentDeduplicatedCount": 0,
                    "providerInferenceCount": max(0, len(items) - 1),
                    "corruptCacheMissCount": 0,
                    "expiredCacheMissCount": 0,
                },
            }
        )

    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings(batch_size=2))
    monkeypatch.setattr("app.identity_worker.httpx.post", fake_post)
    updates = []
    result = embed_identity_frames(
        frames,
        on_progress=lambda completed, total, usable: updates.append((completed, total, usable)),
    )
    assert calls == [[1, 2], [3]]
    assert updates == [(2, 3, 1), (3, 3, 2)]
    assert set(result) == {"track-1:1", "track-1:2", "track-1:3"}
    assert result["track-1:1"]["provider"] == "prtreid-bpbreid-soccernet"
    assert result["track-1:2"]["usable"] is False
    assert result["track-1:2"]["embedding"] is None
    assert result.diagnostics["requestedObservationCount"] == 3
    assert result.diagnostics["cacheHitCount"] == 2
    assert result.diagnostics["providerInferenceCount"] == 1
    assert result.diagnostics["modelContract"] == {
        "backend": "prtreid-bpbreid-soccernet",
        "modelVersion": "model-v1",
        "dimension": 256,
        "normalized": True,
        "evidenceFingerprintVersion": "pixel-evidence-v1",
    }


def test_client_rejects_model_version_change_between_http_batches(
    monkeypatch, tmp_path
):
    frames = []
    for frame_index in (1, 2):
        path = tmp_path / f"frame-{frame_index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(
            (
                frame_index,
                path,
                [
                    {
                        "observationId": f"obs-{frame_index}",
                        "bbox": {"x": 1, "y": 2, "width": 30, "height": 60},
                    }
                ],
            )
        )

    call_count = 0

    def fake_post(_url, data, **_kwargs):
        nonlocal call_count
        call_count += 1
        frame = json.loads(data["manifest"])["frames"][0]
        observation_id = frame["observations"][0]["observationId"]
        return FakeResponse(
            {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "modelVersion": f"model-v{call_count}",
                "items": [
                    {
                        "observationId": observation_id,
                        "frameIndex": frame["frameIndex"],
                        "usable": True,
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "embedding": _embedding(),
                        "visibilityScores": None,
                        "role": None,
                        "roleConfidence": None,
                        "evidenceFingerprint": _fingerprint(observation_id),
                    }
                ],
            }
        )

    monkeypatch.setattr(
        "app.identity_worker.get_settings", lambda: _settings(batch_size=1)
    )
    monkeypatch.setattr("app.identity_worker.httpx.post", fake_post)

    with pytest.raises(
        IdentityWorkerError,
        match="changed model contract between batches: modelVersion",
    ):
        embed_identity_frames(frames)

    assert call_count == 2


def test_client_rejects_non_normalized_embedding(monkeypatch, tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    frames = [
        (
            1,
            path,
            [{"observationId": "obs", "bbox": {"x": 1, "y": 2, "width": 30, "height": 60}}],
        )
    ]
    vector = _embedding()
    vector[0] = 2.0
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "modelVersion": "model-v1",
                "items": [
                    {
                        "observationId": "obs",
                        "frameIndex": 1,
                        "usable": True,
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "embedding": vector,
                        "visibilityScores": None,
                        "role": None,
                        "roleConfidence": None,
                        "evidenceFingerprint": _fingerprint(),
                    }
                ],
            }
        ),
    )
    with pytest.raises(IdentityWorkerError, match="non-normalized"):
        embed_identity_frames(frames)


def test_readiness_rejects_non_object_json_without_raising(monkeypatch):
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker.httpx.get", lambda *_args, **_kwargs: FakeResponse([])
    )

    assert identity_worker_readiness()["status"] == "invalid-response"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"usable": None}, "explicit usable"),
        ({"role": "spectator", "roleConfidence": 0.9}, "unknown role"),
        ({"role": "player", "roleConfidence": float("nan")}, "roleConfidence"),
        ({"evidenceFingerprint": None}, "evidence fingerprint"),
    ],
)
def test_client_rejects_malformed_identity_item(
    monkeypatch, tmp_path, mutation, message
):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    item = {
        "observationId": "obs",
        "frameIndex": 1,
        "usable": True,
        "quality": _quality(),
        "rejectionReasons": [],
        "embedding": _embedding(),
        "visibilityScores": [1.0, 0.5],
        "role": "player",
        "roleConfidence": 0.9,
        "evidenceFingerprint": _fingerprint(),
        **mutation,
    }
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "modelVersion": "model-v1",
                "items": [item],
            }
        ),
    )

    with pytest.raises(IdentityWorkerError, match=message):
        embed_identity_frames(
            [(1, path, [{"observationId": "obs", "bbox": {"x": 1, "y": 2, "width": 30, "height": 60}}])]
        )


def test_client_rejects_non_object_embedding_response(monkeypatch, tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr("app.identity_worker.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker.httpx.post", lambda *_args, **_kwargs: FakeResponse([])
    )

    with pytest.raises(IdentityWorkerError, match="top-level JSON"):
        embed_identity_frames(
            [(1, path, [{"observationId": "obs", "bbox": {"x": 1, "y": 2, "width": 30, "height": 60}}])]
        )


def test_client_reports_duplicate_pixel_evidence_across_http_batches(
    monkeypatch, tmp_path
):
    frames = []
    for frame_index in (1, 2):
        path = tmp_path / f"frame-{frame_index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(
            (
                frame_index,
                path,
                [
                    {
                        "observationId": f"obs-{frame_index}",
                        "bbox": {"x": 1, "y": 2, "width": 30, "height": 60},
                    }
                ],
            )
        )

    def fake_post(_url, data, **_kwargs):
        frame = json.loads(data["manifest"])["frames"][0]
        observation_id = frame["observations"][0]["observationId"]
        return FakeResponse(
            {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v1",
                "modelVersion": "model-v1",
                "items": [
                    {
                        "observationId": observation_id,
                        "frameIndex": frame["frameIndex"],
                        "usable": True,
                        "quality": _quality(),
                        "rejectionReasons": [],
                        "embedding": _embedding(),
                        "visibilityScores": None,
                        "role": None,
                        "roleConfidence": None,
                        "evidenceFingerprint": _fingerprint("same-pixels"),
                    }
                ],
            }
        )

    monkeypatch.setattr(
        "app.identity_worker.get_settings", lambda: _settings(batch_size=1)
    )
    monkeypatch.setattr("app.identity_worker.httpx.post", fake_post)

    result = embed_identity_frames(frames)

    assert result.diagnostics["uniqueEvidenceFingerprintCount"] == 1
    assert result.diagnostics["duplicateEvidenceFingerprintCount"] == 1
