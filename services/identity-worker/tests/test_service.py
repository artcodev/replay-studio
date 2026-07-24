from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import io
import json
import time

import httpx
import numpy as np
from PIL import Image
import pytest

from identity_worker_service.cache import IdentityEmbeddingCache
from identity_worker_service.main import create_app
from identity_worker_service.provider_contract import (
    EmbeddingSample,
    ProviderEmbedding,
    ProviderUnavailable,
)
from identity_worker_service.prtreid_provider import (
    PRTReIDProvider,
    resolve_torch_device,
)


class FakeProvider:
    backend = "prtreid-bpbreid-soccernet"
    dimension = 256

    def __init__(self, *, delay: float = 0.0) -> None:
        self._loaded = False
        self.received: list[str] = []
        self.model_version = "fake-model-v1"
        self.delay = delay

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "dimension": self.dimension,
            "normalized": True,
            "device": "cpu",
            "batchSize": 8,
            "modelVersion": self.model_version,
            "checkpointSha256": f"checkpoint-{self.model_version}",
            "modelLoadSeconds": 0.01,
            "soccerNetCommit": "test",
        }

    def embed(self, samples: list[EmbeddingSample]) -> list[ProviderEmbedding]:
        if self.delay:
            time.sleep(self.delay)
        self.received.extend(item.observation_id for item in samples)
        vector = np.zeros(256, dtype=np.float32)
        vector[0] = 3.0
        vector[1] = 4.0
        return [
            ProviderEmbedding(
                observation_id=item.observation_id,
                embedding=vector.copy(),
                visibility_scores=np.array([1.0, 0.5], dtype=np.float32),
                role="player",
                role_confidence=0.9,
            )
            for item in samples
        ]


class UnavailableProvider(FakeProvider):
    def load(self) -> None:
        raise ProviderUnavailable("checkpoint is missing")


class _FakeMpsBackend:
    def __init__(self, *, built: bool, available: bool) -> None:
        self._built = built
        self._available = available

    def is_built(self) -> bool:
        return self._built

    def is_available(self) -> bool:
        return self._available


class _FakeTorchRuntime:
    def __init__(self, *, built: bool, available: bool) -> None:
        self.backends = type(
            "Backends",
            (),
            {"mps": _FakeMpsBackend(built=built, available=available)},
        )()
        self.cuda = type("Cuda", (), {"is_available": lambda _self: False})()

    @staticmethod
    def device(value: str) -> str:
        return value


def test_mps_device_fails_closed_when_metal_is_unavailable():
    runtime = _FakeTorchRuntime(built=True, available=False)

    with pytest.raises(ProviderUnavailable, match="Metal/MPS is unavailable"):
        resolve_torch_device(runtime, "mps")


def test_mps_device_is_preserved_when_metal_is_available():
    runtime = _FakeTorchRuntime(built=True, available=True)

    assert resolve_torch_device(runtime, "mps") == "mps"


def test_production_provider_never_loads_without_verified_assets(monkeypatch, tmp_path):
    missing_primary = tmp_path / "missing-prtreid.pth.tar"
    missing_hrnet = tmp_path / "missing-hrnet.pth"
    monkeypatch.setenv("PRTREID_WEIGHTS", str(missing_primary))
    monkeypatch.setenv("PRTREID_HRNET_WEIGHTS", str(missing_hrnet))
    provider = PRTReIDProvider()
    with np.testing.assert_raises_regex(ProviderUnavailable, "checkpoint is missing"):
        provider.load()
    assert provider.loaded is False


class _TensorResult:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _InferenceTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


def _loaded_production_provider(role_logits: np.ndarray) -> PRTReIDProvider:
    provider = PRTReIDProvider()
    provider._loaded = True
    provider._torch = _InferenceTorch()
    provider._feature_extractor = lambda images, external_parts_masks=None: images
    vector = np.zeros((len(role_logits), 256), dtype=np.float32)
    vector[:, 0] = 3.0
    vector[:, 1] = 4.0
    provider._extract_test_embeddings = lambda _raw, _parts: (
        _TensorResult(vector),
        None,
        None,
        None,
        {"globl": _TensorResult(role_logits)},
    )
    return provider


def test_production_role_logits_use_stable_softmax_without_changing_embedding() -> None:
    # Direct exp(logit) overflows here; subtracting the maximum must still
    # produce the same bounded player probability.
    logits = np.asarray([[996.0, 1000.0, 998.0, 1007.0, 1001.0]], dtype=np.float32)
    provider = _loaded_production_provider(logits)

    [result] = provider.embed(
        [EmbeddingSample("outside-unit-range", np.zeros((80, 40, 3), dtype=np.uint8))]
    )

    expected_weights = np.exp(logits[0] - logits[0].max())
    expected_confidence = float(expected_weights[3] / expected_weights.sum())
    assert result.role == "player"
    assert result.role_confidence is not None
    assert 0.0 < result.role_confidence < 1.0
    assert result.role_confidence == pytest.approx(expected_confidence)
    np.testing.assert_allclose(result.embedding[:2], [0.6, 0.8], atol=1e-6)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)


def test_non_finite_role_logits_omit_role_but_keep_embedding() -> None:
    provider = _loaded_production_provider(
        np.asarray([[0.0, np.nan, 1.0, 2.0, 3.0]], dtype=np.float32)
    )

    [result] = provider.embed(
        [EmbeddingSample("non-finite-role", np.zeros((80, 40, 3), dtype=np.uint8))]
    )

    assert result.role is None
    assert result.role_confidence is None
    np.testing.assert_allclose(result.embedding[:2], [0.6, 0.8], atol=1e-6)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)


def _jpeg(shift: int = 0, size: tuple[int, int] = (44, 76)) -> bytes:
    """One deterministic crop image; different shifts give different bytes."""

    width, height = size
    yy, xx = np.indices((height, width))
    values = (((xx + shift) // 3 + yy // 3) % 2 * 255).astype(np.uint8)
    image = np.stack([values, values, values], axis=-1)
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=95)
    return output.getvalue()


def _quality() -> dict:
    return {
        "cropWidth": 44,
        "cropHeight": 76,
        "sourceBoxWidth": 40.0,
        "sourceBoxHeight": 70.0,
        "borderClipped": False,
        "sharpness": 55.0,
    }


def _call(app, method: str, path: str, **kwargs):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


def _request(app, crops: list[tuple[str, bytes]], manifest: dict | None = None):
    return _call(
        app,
        "POST",
        "/v1/embeddings",
        files=[
            ("crops", (f"crop-{index}.jpg", data, "image/jpeg"))
            for index, (_identifier, data) in enumerate(crops)
        ],
        data={
            "manifest": json.dumps(
                manifest
                if manifest is not None
                else {
                    "contractVersion": 2,
                    "crops": [
                        {
                            "observationId": identifier,
                            "frameIndex": 7,
                            "fileIndex": index,
                            "quality": _quality(),
                        }
                        for index, (identifier, _data) in enumerate(crops)
                    ],
                }
            )
        },
    )


def _cache() -> IdentityEmbeddingCache:
    return IdentityEmbeddingCache(
        dimension=256,
        max_entries=16,
        ttl_seconds=60,
        wait_timeout_seconds=5,
    )


def test_health_ready_requires_a_loaded_real_provider_contract():
    provider = FakeProvider()
    app = create_app(provider, preload=True)
    assert _call(app, "GET", "/health/live").json()["status"] == "ok"
    ready = _call(app, "GET", "/health/ready")
    assert ready.status_code == 200
    assert ready.json()["backend"] == provider.backend
    assert ready.json()["dimension"] == 256
    assert ready.json()["modelVersion"] == "fake-model-v1"
    assert ready.json()["evidenceFingerprintVersion"] == "pixel-evidence-v2"
    assert ready.json()["cache"] == {
        "schemaVersion": "identity-embedding-cache.v3",
        "enabled": True,
        "maxEntries": 4096,
        "ttlSeconds": 86400.0,
        "waitTimeoutSeconds": 900.0,
        "size": 0,
        "inFlight": 0,
        "configurationError": None,
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
        "expirations": 0,
        "corruptMisses": 0,
        "inRequestDeduplicated": 0,
        "concurrentDeduplicated": 0,
        "waitTimeouts": 0,
        "providerFailures": 0,
    }


def test_missing_model_keeps_liveness_but_never_claims_readiness():
    app = create_app(UnavailableProvider(), preload=True)
    assert _call(app, "GET", "/health/live").status_code == 200
    ready = _call(app, "GET", "/health/ready")
    assert ready.status_code == 503
    assert "checkpoint is missing" in ready.json()["detail"]
    response = _request(app, [("obs-1", _jpeg())])
    assert response.status_code == 503


def test_embeddings_are_normalized_and_keep_provider_evidence():
    provider = FakeProvider()
    app = create_app(provider, preload=False)
    response = _request(app, [("obs-1", _jpeg())])
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidenceFingerprintVersion"] == "pixel-evidence-v2"
    item = payload["items"][0]
    assert item["usable"] is True
    assert item["frameIndex"] == 7
    # Extraction quality travels with the manifest and is echoed verbatim.
    assert item["quality"] == _quality()
    assert item["rejectionReasons"] == []
    assert item["evidenceFingerprint"].startswith("pixel-evidence-v2:")
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(item["embedding"])), 1.0, atol=1e-6
    )
    assert item["embedding"][:2] == [0.6000000238418579, 0.800000011920929]
    assert item["role"] == "player"
    assert item["visibilityScores"] == [1.0, 0.5]
    assert provider.received == ["obs-1"]


def test_undecodable_crop_bytes_fail_closed():
    app = create_app(FakeProvider(), preload=False)
    response = _request(app, [("obs-1", b"not-a-jpeg")])
    assert response.status_code == 422
    assert "not a readable image" in response.json()["detail"]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"frames": []}, "contractVersion"),
        ({"contractVersion": 1, "crops": []}, "contractVersion"),
        ({"contractVersion": 2, "crops": []}, "non-empty"),
        (
            {
                "contractVersion": 2,
                "crops": [
                    {"observationId": "obs", "frameIndex": 0, "fileIndex": 4,
                     "quality": {}}
                ],
            },
            "out of range",
        ),
        (
            {
                "contractVersion": 2,
                "crops": [
                    {"observationId": "obs", "frameIndex": 0, "fileIndex": 0},
                ],
            },
            "quality",
        ),
        (
            {
                "contractVersion": 2,
                "crops": [
                    {"observationId": "obs", "frameIndex": 0, "fileIndex": 0,
                     "quality": {}},
                    {"observationId": "obs", "frameIndex": 1, "fileIndex": 0,
                     "quality": {}},
                ],
            },
            "Duplicate observationId",
        ),
    ],
)
def test_manifest_contract_violations_fail_closed(manifest, message):
    app = create_app(FakeProvider(), preload=False)
    response = _request(app, [("obs", _jpeg())], manifest=manifest)
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_identical_second_request_is_served_without_provider_call():
    provider = FakeProvider()
    app = create_app(provider, embedding_cache=_cache(), preload=False)
    first = _request(app, [("first", _jpeg())])
    second = _request(app, [("second", _jpeg())])

    assert first.status_code == second.status_code == 200
    assert provider.received == ["first"]
    assert first.json()["diagnostics"]["providerInferenceCount"] == 1
    assert second.json()["diagnostics"]["cacheHitCount"] == 1
    assert second.json()["diagnostics"]["cacheMissCount"] == 0
    assert second.json()["diagnostics"]["providerInferenceCount"] == 0
    assert first.json()["diagnostics"]["providerCallCount"] == 1
    assert first.json()["diagnostics"]["providerInferenceSeconds"] >= 0
    assert first.json()["diagnostics"]["requestSeconds"] >= 0
    assert second.json()["items"][0]["cacheHit"] is True


def test_different_crop_pixels_are_distinct_cache_entries():
    provider = FakeProvider()
    app = create_app(provider, embedding_cache=_cache(), preload=False)
    _request(app, [("first", _jpeg(shift=0))])
    changed = _request(app, [("changed", _jpeg(shift=1))])

    assert provider.received == ["first", "changed"]
    assert changed.json()["diagnostics"]["cacheHitCount"] == 0
    assert changed.json()["diagnostics"]["cacheMissCount"] == 1


def test_model_change_invalidates_shared_cache():
    provider = FakeProvider()
    shared_cache = _cache()
    app = create_app(provider, embedding_cache=shared_cache, preload=False)
    _request(app, [("model-v1", _jpeg())])

    provider.model_version = "fake-model-v2"
    model_changed = _request(app, [("model-v2", _jpeg())])
    assert model_changed.json()["diagnostics"]["cacheMissCount"] == 1
    assert provider.received == ["model-v1", "model-v2"]


def test_duplicate_crops_in_one_request_are_inferred_once():
    provider = FakeProvider()
    app = create_app(provider, embedding_cache=_cache(), preload=False)
    response = _request(
        app,
        [("duplicate-a", _jpeg()), ("duplicate-b", _jpeg())],
    )

    assert response.status_code == 200
    assert provider.received == ["duplicate-a"]
    assert response.json()["diagnostics"]["cacheMissCount"] == 1
    assert response.json()["diagnostics"]["deduplicatedObservationCount"] == 1
    assert response.json()["diagnostics"]["providerInferenceCount"] == 1
    assert all(item["embedding"] is not None for item in response.json()["items"])
    fingerprints = {
        item["evidenceFingerprint"] for item in response.json()["items"]
    }
    assert len(fingerprints) == 1
    assert response.json()["diagnostics"]["duplicateEvidenceFingerprintCount"] == 1
    assert response.json()["diagnostics"]["uniqueEvidenceFingerprintCount"] == 1


def test_corrupt_cache_entry_is_removed_and_recomputed_as_a_safe_miss():
    provider = FakeProvider()
    cache = _cache()
    app = create_app(provider, embedding_cache=cache, preload=False)
    _request(app, [("first", _jpeg())])
    with cache._lock:
        key = next(iter(cache._entries))
        cache._entries[key].value = {"corrupt": True}

    response = _request(app, [("after-corruption", _jpeg())])

    assert response.status_code == 200
    assert provider.received == ["first", "after-corruption"]
    assert response.json()["diagnostics"]["corruptCacheMissCount"] == 1
    assert response.json()["diagnostics"]["cacheMissCount"] == 1
    assert response.json()["items"][0]["cacheSource"] == "provider"


def test_concurrent_identical_requests_share_one_inflight_provider_call():
    provider = FakeProvider(delay=0.15)
    app = create_app(provider, embedding_cache=_cache(), preload=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_request, app, [("concurrent-a", _jpeg())])
        second = executor.submit(_request, app, [("concurrent-b", _jpeg())])
        responses = [first.result(timeout=5), second.result(timeout=5)]

    assert all(response.status_code == 200 for response in responses)
    assert len(provider.received) == 1
    diagnostics = [response.json()["diagnostics"] for response in responses]
    assert sum(item["providerInferenceCount"] for item in diagnostics) == 1
    assert sum(item["concurrentDeduplicatedCount"] for item in diagnostics) == 1


def test_cache_is_bounded_by_lru_capacity():
    provider = FakeProvider()
    cache = IdentityEmbeddingCache(
        dimension=256,
        max_entries=1,
        ttl_seconds=60,
        wait_timeout_seconds=5,
    )
    app = create_app(provider, embedding_cache=cache, preload=False)
    _request(app, [("first", _jpeg(shift=0))])
    _request(app, [("second", _jpeg(shift=1))])
    evicted = _request(app, [("first-again", _jpeg(shift=0))])

    assert provider.received == ["first", "second", "first-again"]
    assert evicted.json()["diagnostics"]["cacheMissCount"] == 1
    assert cache.stats()["size"] == 1
    assert cache.stats()["evictions"] == 2


def test_expired_entry_is_a_safe_miss():
    provider = FakeProvider()
    cache = IdentityEmbeddingCache(
        dimension=256,
        max_entries=4,
        ttl_seconds=0.01,
        wait_timeout_seconds=5,
    )
    app = create_app(provider, embedding_cache=cache, preload=False)
    _request(app, [("before-expiry", _jpeg())])
    time.sleep(0.03)
    expired = _request(app, [("after-expiry", _jpeg())])

    assert provider.received == ["before-expiry", "after-expiry"]
    assert expired.json()["diagnostics"]["expiredCacheMissCount"] == 1
    assert expired.json()["diagnostics"]["cacheMissCount"] == 1
