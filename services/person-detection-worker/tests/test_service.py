from __future__ import annotations

import asyncio
import io
import json

import httpx
import numpy as np
from PIL import Image
import pytest

from person_detection_worker_service.main import create_app
from person_detection_worker_service.ultralytics_engine import (
    DetectionEngineUnavailable,
    resolve_torch_device,
)


class FakeEngine:
    backend = "ultralytics-yolo"

    def __init__(self) -> None:
        self._loaded = False
        self.received = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def info(self) -> dict:
        return {
            "schemaVersion": 1,
            "backend": self.backend,
            "providerVersion": "8.4.95",
            "modelVersion": "a" * 16,
            "checkpoint": {
                "name": "yolo26m.pt",
                "size": 42_000_000,
                "sha256": "a" * 64,
            },
            "device": "mps",
            "batchSize": 1,
            "torchVersion": "2.2.2",
            "mpsFallbackEnabled": False,
            "modelLoadSeconds": 1.25,
        }

    def predict(self, image, policy: dict):
        self.received += 1
        assert image.shape == (48, 64, 3)
        assert policy["imageSize"] == 1280
        return (
            {0: "person", 32: "sports ball"},
            [
                {
                    "classId": 0,
                    "confidence": 0.9,
                    "x1": 10.0,
                    "y1": 5.0,
                    "x2": 30.0,
                    "y2": 40.0,
                }
            ],
            0.05,
            0,
        )


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _manifest(**updates) -> str:
    return json.dumps(
        {
            "contractVersion": 1,
            "imageSize": 1280,
            "confidence": 0.035,
            "nmsIou": 0.7,
            "maxDetections": 300,
            **updates,
        }
    )


def _call(app, method: str, path: str, **kwargs):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


def test_worker_keeps_engine_loaded_and_returns_strict_raw_boxes():
    engine = FakeEngine()
    app = create_app(engine, preload=False)

    ready = _call(app, "GET", "/health/ready")
    response = _call(
        app,
        "POST",
        "/v1/detections",
        files={"frame": ("frame.jpg", _jpeg(), "image/jpeg")},
        data={"manifest": _manifest()},
    )

    assert ready.status_code == response.status_code == 200
    assert ready.json()["device"] == "mps"
    assert response.json()["boxes"][0]["classId"] == 0
    assert response.json()["diagnostics"]["inferenceSeconds"] == 0.05
    assert response.json()["image"] == {"width": 64, "height": 48}
    assert engine.received == 1


def test_unknown_request_fields_fail_closed():
    app = create_app(FakeEngine(), preload=False)

    response = _call(
        app,
        "POST",
        "/v1/detections",
        files={"frame": ("frame.jpg", _jpeg(), "image/jpeg")},
        data={"manifest": _manifest(extra="not-allowed")},
    )

    assert response.status_code == 422
    assert "unknown fields" in response.json()["detail"]


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_built(self):
        return True

    def is_available(self):
        return self.available


class _FakeTorch:
    def __init__(self, available: bool) -> None:
        self.backends = type("Backends", (), {"mps": _FakeMps(available)})()
        self.cuda = type("Cuda", (), {"is_available": lambda _self: False})()

    @staticmethod
    def device(value):
        return value


def test_requested_mps_never_silently_becomes_cpu():
    with pytest.raises(DetectionEngineUnavailable, match="Metal/MPS"):
        resolve_torch_device(_FakeTorch(False), "mps")

    assert resolve_torch_device(_FakeTorch(True), "mps") == "mps"
