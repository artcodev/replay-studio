import json
from types import SimpleNamespace

import httpx

from app.calibration_worker import (
    calibrate_frames_with_worker,
    calibration_worker_readiness,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_worker_client_reports_progress_after_each_small_frame_batch(monkeypatch, tmp_path):
    frames = []
    for index in range(1, 6):
        path = tmp_path / f"frame_{index:05d}.jpg"
        path.write_bytes(b"jpeg")
        frames.append((index, path))
    calls = []

    def fake_post(url, data, files, timeout):
        indices = json.loads(data["frame_indices"])
        calls.append(indices)
        return FakeResponse(
            {
                "backend": "pnlcalib-points-lines",
                "diagnostics": {
                    "modelVersion": "test-model-v1",
                    "cacheHitCount": 1,
                    "totalSeconds": 0.012,
                },
                "frames": [
                    {
                        "frameIndex": index,
                        "method": "pnlcalib-points-lines",
                        "confidence": 0.9,
                        "keypointCount": 8,
                        "inlierCount": 8,
                        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
                    }
                    for index in indices
                ],
            }
        )

    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(
            calibration_worker_url="http://worker:8090",
            calibration_worker_timeout=900,
            calibration_worker_batch_size=2,
        ),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.post", fake_post)
    updates = []

    result = calibrate_frames_with_worker(
        frames,
        on_progress=lambda completed, total, valid: updates.append((completed, total, valid)),
    )

    assert calls == [[1, 2], [3, 4], [5]]
    assert updates == [(2, 5, 2), (4, 5, 4), (5, 5, 5)]
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert result[1].backend_diagnostics == {
        "modelVersion": "test-model-v1",
        "cacheHitCount": 1,
        "totalSeconds": 0.012,
    }


def test_worker_readiness_reports_disabled_without_a_configured_url(monkeypatch):
    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(calibration_worker_url=""),
    )

    assert calibration_worker_readiness() == {
        "configured": False,
        "status": "disabled",
        "backend": None,
    }


def test_worker_readiness_uses_model_loading_endpoint(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return FakeResponse(
            {
                "status": "ready",
                "backend": "pnlcalib-points-lines",
                "device": "cpu",
                "batchSize": 2,
                "modelVersion": "test-model-v1",
                "modelLoadSeconds": 5.2,
                "cacheEntryCount": 3,
            }
        )

    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(calibration_worker_url="http://worker:8090/"),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.get", fake_get)

    assert calibration_worker_readiness(timeout=1.5) == {
        "configured": True,
        "status": "ready",
        "backend": "pnlcalib-points-lines",
        "device": "cpu",
        "batchSize": 2,
        "modelVersion": "test-model-v1",
        "modelLoadSeconds": 5.2,
        "cacheEntryCount": 3,
    }
    assert calls == [("http://worker:8090/health/ready", 1.5)]


def test_worker_readiness_is_nonfatal_when_worker_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(calibration_worker_url="http://worker:8090"),
    )

    def unavailable(*_args, **_kwargs):
        raise httpx.ConnectError("worker is offline")

    monkeypatch.setattr("app.calibration_worker.httpx.get", unavailable)

    result = calibration_worker_readiness()

    assert result["configured"] is True
    assert result["status"] == "unavailable"
    assert result["backend"] == "pnlcalib-points-lines"
    assert "worker is offline" in result["detail"]
