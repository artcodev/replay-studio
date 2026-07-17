from __future__ import annotations

import base64
import io
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from ball_worker_service.main import create_app
from ball_worker_service.providers import (
    BallCandidate,
    ProviderUnavailable,
    WasbSoccerProvider,
)


class FakeProvider:
    backend = "wasb-sbdt-soccer"
    frames_in = 3
    frames_out = 3

    def __init__(self) -> None:
        self._loaded = False
        self.windows: list[list[int]] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "modelVersion": "wasb-soccer@sha256:fake",
            "device": "cpu",
            "framesIn": 3,
            "framesOut": 3,
            "inputSize": [512, 288],
            "scoreThreshold": 0.5,
        }

    def detect_window(self, frames_rgb, *, max_candidates):
        self.windows.append([int(frame[0, 0, 0]) for frame in frames_rgb])
        return [
            [
                BallCandidate(
                    x=10.0 + offset,
                    y=20.0 + offset,
                    confidence=0.9 - offset * 0.1,
                    heatmap_peak=0.9 - offset * 0.1,
                    component_score=3.5,
                    component_area=4,
                    metadata={"fakeOffset": offset},
                )
            ][:max_candidates]
            for offset in range(3)
        ]


class UnavailableProvider(FakeProvider):
    def load(self) -> None:
        raise ProviderUnavailable("verified WASB checkpoint is missing")


def _jpeg(value: int, size: tuple[int, int] = (64, 36)) -> bytes:
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def test_real_provider_rejects_missing_checkpoint_before_importing_torch(monkeypatch, tmp_path):
    monkeypatch.setenv("WASB_WEIGHTS", str(tmp_path / "missing.pth.tar"))
    monkeypatch.setenv("WASB_HRNET_SOURCE", str(tmp_path / "missing.py"))
    provider = WasbSoccerProvider()

    with np.testing.assert_raises_regex(ProviderUnavailable, "checkpoint is missing"):
        provider.load()

    assert provider.loaded is False


def test_liveness_is_independent_but_readiness_is_honest():
    app = create_app(UnavailableProvider(), preload=True)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert "checkpoint is missing" in response.json()["detail"]


def test_ready_reports_backend_and_exact_model_version():
    app = create_app(FakeProvider(), preload=True)
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["backend"] == "wasb-sbdt-soccer"
    assert response.json()["modelVersion"] == "wasb-soccer@sha256:fake"


def test_multipart_batch_returns_per_frame_candidates_and_pads_only_tail():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    manifest = {
        "contractVersion": 1,
        "maxCandidates": 5,
        "frames": [
            {"fileIndex": index, "frameIndex": 100 + index, "timestampMs": index * 40}
            for index in range(4)
        ],
    }
    files = [
        ("frames", (f"frame-{index}.png", _jpeg(value), "image/png"))
        for index, value in enumerate((10, 20, 30, 40))
    ]
    with TestClient(app) as client:
        response = client.post(
            "/v1/detections",
            files=files,
            data={"manifest": json.dumps(manifest)},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["modelVersion"] == "wasb-soccer@sha256:fake"
    assert payload["metadata"]["windowCount"] == 2
    assert len(payload["frames"]) == 4
    assert provider.windows == [[10, 20, 30], [40, 40, 40]]
    first = payload["frames"][0]["candidates"][0]
    assert first["x"] == 10.0
    assert first["y"] == 20.0
    assert first["confidence"] == 0.9
    assert first["backend"] == provider.backend
    assert first["modelVersion"] == "wasb-soccer@sha256:fake"
    assert first["sourceFrameIndex"] == 100
    assert payload["frames"][0]["temporalPadding"] is False
    assert payload["frames"][3]["temporalPadding"] is True


def test_json_detect_contract_matches_existing_api_adapter_and_preserves_bgr():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    bgr_frames = []
    for value in (11, 22):
        array = np.full((12, 20, 3), value, dtype=np.uint8)
        bgr_frames.append(
            {
                "encoding": "numpy-base64",
                "shape": list(array.shape),
                "dtype": "uint8",
                "colorSpace": "BGR",
                "dataBase64": base64.b64encode(array.tobytes()).decode("ascii"),
            }
        )
    with TestClient(app) as client:
        response = client.post(
            "/detect",
            json={
                "contractVersion": 1,
                "frames": bgr_frames,
                "targetIndex": 1,
                "frameIndex": 77,
                "maxCandidates": 4,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["imageSize"] == [20, 12]
    assert provider.windows == [[11, 11, 22]]
    assert payload["metadata"]["sourceWindowIndices"] == [0, 0, 1]
    assert payload["metadata"]["temporalPadding"] is True
    candidate = payload["candidates"][0]
    assert candidate["sourceFrameIndex"] == 77
    assert candidate["x"] == 12.0
    assert candidate["heatmapPeak"] == 0.7


def test_invalid_numpy_byte_count_is_rejected_before_provider_runs():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    with TestClient(app) as client:
        response = client.post(
            "/detect",
            json={
                "frames": [
                    {
                        "encoding": "numpy-base64",
                        "shape": [10, 20, 3],
                        "dtype": "uint8",
                        "dataBase64": base64.b64encode(b"short").decode("ascii"),
                    }
                ],
                "targetIndex": 0,
            },
        )

    assert response.status_code == 422
    assert "byte count" in response.json()["detail"]
    assert provider.windows == []

