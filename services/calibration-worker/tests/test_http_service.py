from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient, Response

from calibration_worker_service.calibration_service import (
    CalibrationInferenceError,
    CalibrationRequestError,
)
from calibration_worker_service.main import create_app


class FakeCalibrationService:
    def __init__(self) -> None:
        self.request: tuple[str, list[bytes]] | None = None

    def readiness(self) -> dict:
        return {
            "backend": "pnlcalib-points-lines",
            "device": "cpu",
            "batchSize": 2,
            "modelVersion": "fake-v1",
            "modelLoadSeconds": 0.01,
            "cacheMaxEntries": 8,
            "cacheTtlSeconds": 60.0,
            "cacheEntryCount": 0,
        }

    def calibrate(self, frame_indices: str, payloads: list[bytes]) -> dict:
        self.request = (frame_indices, payloads)
        return {
            "backend": "pnlcalib-points-lines",
            "requestedFrameCount": len(payloads),
            "calibratedFrameCount": 1,
            "diagnostics": {},
            "frames": [{"frameIndex": 7}],
        }


async def _request(application, method: str, path: str, **kwargs) -> Response:
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)


def test_http_root_preserves_health_and_batch_contract() -> None:
    service = FakeCalibrationService()
    application = create_app(service, preload=False)
    assert asyncio.run(_request(application, "GET", "/health/live")).json() == {
        "status": "ok",
        "service": "pnlcalib-worker",
    }
    assert (
        asyncio.run(_request(application, "GET", "/health/ready")).json()[
            "modelVersion"
        ]
        == "fake-v1"
    )
    health = asyncio.run(_request(application, "GET", "/health"))
    assert health.json()["status"] == "ready"
    response = asyncio.run(
        _request(
            application,
            "POST",
            "/v1/calibrate",
            files=[("frames", ("frame.jpg", b"image-bytes", "image/jpeg"))],
            data={"frame_indices": "[7]"},
        )
    )

    assert response.status_code == 200
    assert response.json()["frames"] == [{"frameIndex": 7}]
    assert service.request == ("[7]", [b"image-bytes"])


def test_http_root_maps_request_and_inference_failures() -> None:
    service = FakeCalibrationService()
    application = create_app(service, preload=False)
    service.calibrate = lambda *_args: (_ for _ in ()).throw(
        CalibrationRequestError("bad frame mapping")
    )
    invalid = asyncio.run(
        _request(
            application,
            "POST",
            "/v1/calibrate",
            files=[("frames", ("frame.jpg", b"bad", "image/jpeg"))],
            data={"frame_indices": "[]"},
        )
    )
    service.calibrate = lambda *_args: (_ for _ in ()).throw(
        CalibrationInferenceError("model failed")
    )
    failed = asyncio.run(
        _request(
            application,
            "POST",
            "/v1/calibrate",
            files=[("frames", ("frame.jpg", b"bad", "image/jpeg"))],
            data={"frame_indices": "[0]"},
        )
    )

    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "bad frame mapping"
    assert failed.status_code == 500
    assert failed.json()["detail"] == "model failed"
