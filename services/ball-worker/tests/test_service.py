from __future__ import annotations

import asyncio
import io
import json

from httpx import ASGITransport, AsyncClient, Response
import numpy as np
from PIL import Image

from ball_worker_service.main import create_app
from ball_worker_service.provider_contract import (
    BallCandidate,
    BallProviderInfo,
    ProviderUnavailable,
)
from ball_worker_service.wasb_provider import WasbSoccerProvider


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

    def info(self) -> BallProviderInfo:
        return BallProviderInfo(
            backend=self.backend,
            model_version="wasb-soccer@sha256:fake",
            checkpoint_sha256="fake",
            device="cpu",
            frames_in=3,
            frames_out=3,
            input_size=(512, 288),
            score_threshold=0.5,
            model_load_seconds=None,
        )

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


async def _request(application, method: str, path: str, **kwargs) -> Response:
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)


def test_real_provider_rejects_missing_checkpoint_before_importing_torch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("WASB_WEIGHTS", str(tmp_path / "missing.pth.tar"))
    monkeypatch.setenv("WASB_HRNET_SOURCE", str(tmp_path / "missing.py"))
    provider = WasbSoccerProvider()

    with np.testing.assert_raises_regex(ProviderUnavailable, "checkpoint is missing"):
        provider.load()

    assert provider.loaded is False


def test_real_provider_rejects_checkpoint_with_wrong_sha_before_model_import(
    monkeypatch,
    tmp_path,
):
    weights_path = tmp_path / "wrong.pth.tar"
    source_path = tmp_path / "hrnet.py"
    weights_path.write_bytes(b"not-the-pinned-checkpoint")
    source_path.write_text("class HRNet: pass\n", encoding="utf-8")
    monkeypatch.setenv("WASB_WEIGHTS", str(weights_path))
    monkeypatch.setenv("WASB_HRNET_SOURCE", str(source_path))
    monkeypatch.setenv("WASB_WEIGHTS_SHA256", "0" * 64)
    provider = WasbSoccerProvider()

    with np.testing.assert_raises_regex(ProviderUnavailable, "checksum mismatch"):
        provider.load()

    assert provider.loaded is False


def test_real_provider_rejects_disabled_checkpoint_verification(monkeypatch):
    monkeypatch.setenv("WASB_WEIGHTS_SHA256", "")

    with np.testing.assert_raises_regex(ProviderUnavailable, "complete 64-character"):
        WasbSoccerProvider()


def test_liveness_is_independent_but_readiness_is_honest():
    app = create_app(UnavailableProvider(), preload=True)
    assert asyncio.run(_request(app, "GET", "/health/live")).status_code == 200
    response = asyncio.run(_request(app, "GET", "/health/ready"))
    assert response.status_code == 503
    assert "checkpoint is missing" in response.json()["detail"]


def test_ready_reports_backend_and_exact_model_version():
    app = create_app(FakeProvider(), preload=True)
    response = asyncio.run(_request(app, "GET", "/health/ready"))

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
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/detections",
            files=files,
            data={"manifest": json.dumps(manifest)},
        )
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


def test_multipart_manifest_reuses_upload_for_causal_padding():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    manifest = {
        "contractVersion": 1,
        "targetIndex": 2,
        "maxCandidates": 4,
        "frames": [
            {"fileIndex": 0, "frameIndex": 76},
            {"fileIndex": 0, "frameIndex": 76},
            {"fileIndex": 1, "frameIndex": 77},
        ],
    }
    files = [
        ("frames", ("previous.png", _jpeg(11, (20, 12)), "image/png")),
        ("frames", ("current.png", _jpeg(22, (20, 12)), "image/png")),
    ]
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/detections",
            files=files,
            data={"manifest": json.dumps(manifest)},
        )
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert provider.windows == [[11, 11, 22]]
    assert payload["metadata"]["targetIndex"] == 2
    target = payload["frames"][2]
    assert target["imageSize"] == [20, 12]
    assert target["temporalPadding"] is True
    candidate = target["candidates"][0]
    assert candidate["sourceFrameIndex"] == 77
    assert candidate["x"] == 12.0
    assert candidate["heatmapPeak"] == 0.7


def test_legacy_json_detection_route_is_absent():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    response = asyncio.run(_request(app, "POST", "/detect", json={"frames": []}))

    assert response.status_code == 404
    assert provider.windows == []


def test_manifest_rejects_unknown_fields_before_provider_runs():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    manifest = {
        "contractVersion": 1,
        "frames": [{"fileIndex": 0, "frameIndex": 0}],
        "legacyPayload": True,
    }
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/detections",
            files=[("frames", ("frame.png", _jpeg(11), "image/png"))],
            data={"manifest": json.dumps(manifest)},
        )
    )

    assert response.status_code == 422
    assert "unsupported fields" in response.json()["detail"]
    assert provider.windows == []
