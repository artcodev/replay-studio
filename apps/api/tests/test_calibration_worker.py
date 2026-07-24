import json
from types import SimpleNamespace

import httpx
import pytest

from app.calibration_worker import (
    CalibrationWorkerError,
    calibrate_frames_with_worker,
    calibration_worker_readiness,
    recalibrate_frames_with_worker,
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
            calibration_anchor_cache_enabled=False,
        ),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.post", fake_post)
    updates = []
    batches = []

    result = calibrate_frames_with_worker(
        frames,
        on_progress=lambda completed, total, valid: updates.append((completed, total, valid)),
        on_batch=batches.append,
    )

    assert calls == [[1, 2], [3, 4], [5]]
    assert updates == [(2, 5, 2), (4, 5, 4), (5, 5, 5)]
    assert [(item.completed, item.total, item.batch_size) for item in batches] == [
        (2, 5, 2),
        (4, 5, 2),
        (5, 5, 1),
    ]
    assert batches[0].diagnostics["modelVersion"] == "test-model-v1"
    assert sorted(result) == [1, 2, 3, 4, 5]
    assert result[1].backend_diagnostics == {
        "modelVersion": "test-model-v1",
        "cacheHitCount": 1,
        "totalSeconds": 0.012,
    }


def _cached_settings(tmp_path, batch_size=2):
    return SimpleNamespace(
        calibration_worker_url="http://worker:8090",
        calibration_worker_timeout=900,
        calibration_worker_batch_size=batch_size,
        calibration_anchor_cache_enabled=True,
        media_root=str(tmp_path / "media"),
    )


def _ready_get(url, timeout):
    return FakeResponse(
        {
            "status": "ready",
            "backend": "pnlcalib-points-lines",
            "modelVersion": "test-model-v1",
        }
    )


def test_anchor_results_are_memoized_on_disk_including_no_solution(
    monkeypatch, tmp_path
):
    frames = []
    for index in (1, 2):
        path = tmp_path / f"frame_{index:05d}.jpg"
        path.write_bytes(f"jpeg-{index}".encode())
        frames.append((index, path))
    post_calls = []

    def fake_post(url, data, files, timeout):
        indices = json.loads(data["frame_indices"])
        post_calls.append(indices)
        return FakeResponse(
            {
                "backend": "pnlcalib-points-lines",
                "diagnostics": {"modelVersion": "test-model-v1"},
                # Frame 2 stays unsolved: the worker returns no row for it.
                "frames": [
                    {
                        "frameIndex": index,
                        "method": "pnlcalib-points-lines",
                        "confidence": 0.9,
                        "keypointCount": 8,
                        "inlierCount": 8,
                        "imageToPitch": [
                            [0.1, 0.0, -48.0],
                            [0.0, 0.1, -27.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                    for index in indices
                    if index != 2
                ],
            }
        )

    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: _cached_settings(tmp_path),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.get", _ready_get)
    monkeypatch.setattr("app.calibration_worker.httpx.post", fake_post)

    first = calibrate_frames_with_worker(frames)
    assert sorted(first) == [1]
    assert post_calls == [[1, 2]]

    def forbidden_post(*_args, **_kwargs):
        raise AssertionError("a warm rebuild must not re-upload cached anchors")

    monkeypatch.setattr("app.calibration_worker.httpx.post", forbidden_post)
    second = calibrate_frames_with_worker(frames)

    # Frame 1 resolves from disk; the cached no-solution for frame 2 is
    # authoritative and is not re-asked.
    assert sorted(second) == [1]
    assert second[1].confidence == first[1].confidence


def test_valid_worker_response_may_report_no_direct_solution(monkeypatch, tmp_path):
    path = tmp_path / "frame_00001.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(
            calibration_worker_url="http://worker:8090",
            calibration_worker_timeout=900,
            calibration_worker_batch_size=2,
            calibration_anchor_cache_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "app.calibration_worker.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "backend": "pnlcalib-points-lines",
                "diagnostics": {"modelVersion": "test-model-v1"},
                "frames": [],
            }
        ),
    )

    assert calibrate_frames_with_worker([(1, path)]) == {}


def test_recalibration_uses_forced_refresh_endpoint_and_skips_disk_cache(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "frame_00001.jpg"
    path.write_bytes(b"jpeg")
    calls = []

    def fake_post(url, data, files, timeout):
        calls.append(url)
        return FakeResponse(
            {
                "backend": "pnlcalib-points-lines",
                "diagnostics": {"inferenceMode": "forced-refresh"},
                "frames": [],
            }
        )

    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: _cached_settings(tmp_path),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.get", _ready_get)
    monkeypatch.setattr("app.calibration_worker.httpx.post", fake_post)

    recalibrate_frames_with_worker([(1, path)])
    recalibrate_frames_with_worker([(1, path)])

    assert calls == [
        "http://worker:8090/v1/recalibrate",
        "http://worker:8090/v1/recalibrate",
    ]


def test_calibration_request_requires_pnlcalib_configuration(monkeypatch, tmp_path):
    path = tmp_path / "frame_00001.jpg"
    path.write_bytes(b"jpeg")
    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(calibration_worker_url=""),
    )

    with pytest.raises(CalibrationWorkerError, match="required"):
        calibrate_frames_with_worker([(1, path)])


def test_worker_transport_error_identifies_endpoint_and_source_frames(
    monkeypatch,
    tmp_path,
):
    frames = []
    for index in (126, 128):
        path = tmp_path / f"frame_{index:05d}.jpg"
        path.write_bytes(b"jpeg")
        frames.append((index, path))
    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: SimpleNamespace(
            calibration_worker_url="http://worker:8090",
            calibration_worker_timeout=900,
            calibration_worker_batch_size=2,
            calibration_anchor_cache_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "app.calibration_worker.httpx.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.RemoteProtocolError("server disconnected")
        ),
    )

    with pytest.raises(
        CalibrationWorkerError,
        match=r"recalibrate request failed for source frame\(s\) \[126,128\] "
        r"\(HTTP batch 2\)",
    ):
        recalibrate_frames_with_worker(frames)


def test_changed_worker_model_version_invalidates_the_anchor_cache(
    monkeypatch, tmp_path
):
    path = tmp_path / "frame_00001.jpg"
    path.write_bytes(b"jpeg")
    frames = [(1, path)]
    post_calls = []

    def fake_post(url, data, files, timeout):
        post_calls.append(json.loads(data["frame_indices"]))
        return FakeResponse(
            {
                "backend": "pnlcalib-points-lines",
                "diagnostics": {},
                "frames": [
                    {
                        "frameIndex": 1,
                        "method": "pnlcalib-points-lines",
                        "confidence": 0.9,
                        "keypointCount": 8,
                        "inlierCount": 8,
                        "imageToPitch": [
                            [0.1, 0.0, -48.0],
                            [0.0, 0.1, -27.0],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        )

    model_version = {"value": "model-v1"}

    def versioned_get(url, timeout):
        return FakeResponse(
            {
                "status": "ready",
                "backend": "pnlcalib-points-lines",
                "modelVersion": model_version["value"],
            }
        )

    monkeypatch.setattr(
        "app.calibration_worker.get_settings",
        lambda: _cached_settings(tmp_path),
    )
    monkeypatch.setattr("app.calibration_worker.httpx.get", versioned_get)
    monkeypatch.setattr("app.calibration_worker.httpx.post", fake_post)

    calibrate_frames_with_worker(frames)
    model_version["value"] = "model-v2"
    calibrate_frames_with_worker(frames)

    # A new model identity is a different cache contract: the frame is
    # re-inferred instead of silently reusing the old model's evidence.
    assert post_calls == [[1], [1]]


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
            "architecture": "arm64",
            "torchVersion": "2.2.2",
            "torchThreadCount": 10,
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
        "architecture": "arm64",
        "torchVersion": "2.2.2",
        "torchThreadCount": 10,
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
