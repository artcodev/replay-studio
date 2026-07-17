from __future__ import annotations

import io
import json
import asyncio

import httpx
import numpy as np
from PIL import Image

from jersey_ocr_worker_service.main import QualityPolicy, create_app
from jersey_ocr_worker_service.providers import (
    OcrResult,
    OcrSample,
    ProviderUnavailable,
    RawTextCandidate,
)


class FakeProvider:
    backend = "fake-ocr"

    def __init__(self, candidates: dict[str, list[RawTextCandidate]] | None = None) -> None:
        self._loaded = False
        self.candidates = candidates or {}
        self.received: list[str] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "providerVersion": "test",
            "modelVersion": "fake-v1",
            "device": "cpu",
            "batchSize": 8,
            "modelLoadSeconds": 0.01,
        }

    def recognize(self, samples: list[OcrSample]) -> list[OcrResult]:
        self.received.extend(sample.crop_id for sample in samples)
        return [
            OcrResult(sample.crop_id, tuple(self.candidates.get(sample.crop_id, [])))
            for sample in samples
        ]


class UnavailableProvider(FakeProvider):
    def load(self) -> None:
        raise ProviderUnavailable("OCR checkpoint is missing")


def _jpeg(pattern: str = "sharp", size: tuple[int, int] = (80, 120)) -> bytes:
    width, height = size
    if pattern == "sharp":
        yy, xx = np.indices((height, width))
        values = ((xx // 3 + yy // 3) % 2 * 255).astype(np.uint8)
    else:
        values = np.full((height, width), 128, dtype=np.uint8)
    image = np.stack([values, values, values], axis=-1)
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _call(app, method: str, path: str, **kwargs):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


def _request(app, items: list[dict], images: list[bytes] | None = None):
    payloads = images or [
        _jpeg(size=(80 + index, 120)) for index, _item in enumerate(items)
    ]
    return _call(
        app,
        "POST",
        "/v1/analyze",
        files=[
            ("crops", (f"crop-{index}.jpg", value, "image/jpeg"))
            for index, value in enumerate(payloads)
        ],
        data={
            "manifest": json.dumps(
                {
                    "contractVersion": "jersey-ocr.v1",
                    "items": items,
                }
            )
        },
    )


def _policy(**overrides) -> QualityPolicy:
    values = {
        "minimum_width": 12,
        "minimum_height": 20,
        "minimum_sharpness": 1,
        "minimum_contrast": 1,
        "minimum_confidence": 0.25,
        "ambiguity_margin": 0.05,
    }
    values.update(overrides)
    return QualityPolicy(**values)


def test_live_and_ready_expose_a_provider_neutral_contract():
    app = create_app(FakeProvider(), quality_policy=_policy(), preload=True)
    assert _call(app, "GET", "/health/live").json() == {
        "status": "ok",
        "service": "replay-studio-jersey-ocr-worker",
    }
    response = _call(app, "GET", "/health/ready")
    assert response.status_code == 200
    assert response.json()["contractVersion"] == "jersey-ocr.v1"
    assert response.json()["capabilities"] == {
        "digitsOnly": True,
        "maxDigits": 2,
        "evidenceFingerprintVersion": "pixel-evidence-v1",
        "inputScopes": ["crop", "tracklet"],
    }
    assert response.json()["backend"] == "fake-ocr"


def test_unavailable_model_keeps_liveness_but_not_readiness():
    app = create_app(UnavailableProvider(), quality_policy=_policy(), preload=True)
    assert _call(app, "GET", "/health/live").status_code == 200
    ready = _call(app, "GET", "/health/ready")
    response = _request(
        app,
        [{"cropId": "crop-1", "fileIndex": 0, "observationId": "obs-1"}],
    )
    assert ready.status_code == 503
    assert "checkpoint is missing" in ready.json()["detail"]
    assert response.status_code == 503


def test_batch_recognizes_digits_and_preserves_identity_metadata():
    provider = FakeProvider(
        {
            "crop-1": [RawTextCandidate("ESP #12", 0.91, [[1, 2], [3, 4]])],
            "crop-2": [],
        }
    )
    app = create_app(provider, quality_policy=_policy(), preload=False)
    items = [
        {
            "cropId": "crop-1",
            "fileIndex": 0,
            "observationId": "obs-1",
            "trackletId": "tracklet-1",
            "frameIndex": 7,
            "timestamp": 0.7,
        },
        {"cropId": "crop-2", "fileIndex": 1, "observationId": "obs-2"},
    ]
    response = _request(app, items)
    assert response.status_code == 200
    payload = response.json()
    first, second = payload["items"]
    assert first["number"] == "12"
    assert first["confidence"] == 0.91
    assert first["status"] == "recognized"
    assert first["observationId"] == "obs-1"
    assert first["trackletId"] == "tracklet-1"
    assert first["candidates"][0]["rawText"] == "ESP #12"
    assert second["status"] == "no-number"
    assert second["decisionReasons"] == ["no-numeric-text"]
    assert provider.received == ["crop-1", "crop-2"]
    assert payload["diagnostics"]["recognizedCropCount"] == 1


def test_close_competing_numbers_fail_closed_as_ambiguous():
    provider = FakeProvider(
        {
            "crop-1": [
                RawTextCandidate("8", 0.80),
                RawTextCandidate("6", 0.77),
            ]
        }
    )
    app = create_app(provider, quality_policy=_policy(ambiguity_margin=0.05), preload=False)
    response = _request(app, [{"cropId": "crop-1", "fileIndex": 0}])
    item = response.json()["items"][0]
    assert item["status"] == "ambiguous"
    assert item["number"] is None
    assert item["decisionReasons"] == ["competing-numbers"]


def test_bad_quality_crop_is_retained_without_provider_inference():
    provider = FakeProvider()
    app = create_app(
        provider,
        quality_policy=_policy(
            minimum_width=100,
            minimum_height=150,
            minimum_sharpness=20,
            minimum_contrast=10,
        ),
        preload=False,
    )
    response = _request(
        app,
        [{"cropId": "crop-1", "fileIndex": 0}],
        [_jpeg("flat", (60, 90))],
    )
    item = response.json()["items"][0]
    assert item["usable"] is False
    assert item["status"] == "rejected"
    assert "crop-too-small" in item["rejectionReasons"]
    assert "crop-too-blurry" in item["rejectionReasons"]
    assert provider.received == []


def test_manifest_rejects_duplicate_crop_identity():
    app = create_app(FakeProvider(), quality_policy=_policy(), preload=False)
    response = _request(
        app,
        [
            {"cropId": "duplicate", "fileIndex": 0},
            {"cropId": "duplicate", "fileIndex": 1},
        ],
    )
    assert response.status_code == 422
    assert "Duplicate cropId" in response.json()["detail"]


def test_invalid_policy_keeps_liveness_and_fails_readiness_explicitly():
    app = create_app(
        FakeProvider(),
        quality_policy=_policy(minimum_confidence=1.5),
        preload=False,
    )
    assert _call(app, "GET", "/health/live").status_code == 200
    response = _call(app, "GET", "/health/ready")
    assert response.status_code == 503
    assert "MIN_CONFIDENCE" in response.json()["detail"]


def test_identical_crop_inference_is_deduplicated_and_cached(monkeypatch):
    monkeypatch.setenv("JERSEY_OCR_CACHE_MAX_ENTRIES", "8")
    monkeypatch.setenv("JERSEY_OCR_CACHE_TTL_SECONDS", "60")
    provider = FakeProvider({"first": [RawTextCandidate("23", 0.95)]})
    app = create_app(provider, quality_policy=_policy(), preload=False)
    image = _jpeg()
    response = _request(
        app,
        [
            {"cropId": "first", "fileIndex": 0},
            {"cropId": "same-pixels", "fileIndex": 1},
        ],
        [image, image],
    )
    assert response.status_code == 200
    assert provider.received == ["first"]
    assert [item["number"] for item in response.json()["items"]] == ["23", "23"]
    assert response.json()["diagnostics"]["requestDeduplicatedCount"] == 1
    assert len({item["evidenceFingerprint"] for item in response.json()["items"]}) == 1
    assert response.json()["diagnostics"]["duplicateEvidenceFingerprintCount"] == 1
    assert response.json()["diagnostics"]["uniqueEvidenceFingerprintCount"] == 1

    cached = _request(
        app,
        [{"cropId": "later-request", "fileIndex": 0}],
        [image],
    )
    assert cached.json()["items"][0]["number"] == "23"
    assert cached.json()["items"][0]["cacheHit"] is True
    assert cached.json()["diagnostics"]["cacheHitCount"] == 1
    assert provider.received == ["first"]
