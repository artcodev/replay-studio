from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import time

import httpx
import numpy as np
from PIL import Image

from identity_worker_service.cache import IdentityEmbeddingCache
from identity_worker_service.main import QualityPolicy, create_app
from identity_worker_service.providers import (
    EmbeddingSample,
    PRTReIDProvider,
    ProviderEmbedding,
    ProviderUnavailable,
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


def test_production_provider_never_loads_without_verified_assets(monkeypatch, tmp_path):
    missing_primary = tmp_path / "missing-prtreid.pth.tar"
    missing_hrnet = tmp_path / "missing-hrnet.pth"
    monkeypatch.setenv("PRTREID_WEIGHTS", str(missing_primary))
    monkeypatch.setenv("PRTREID_HRNET_WEIGHTS", str(missing_hrnet))
    provider = PRTReIDProvider()
    with np.testing.assert_raises_regex(ProviderUnavailable, "checkpoint is missing"):
        provider.load()
    assert provider.loaded is False


def _jpeg(pattern: str = "sharp", size: tuple[int, int] = (96, 128)) -> bytes:
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


def _request(app, observations: list[dict], image: bytes | None = None):
    return _call(
        app,
        "POST",
        "/v1/embeddings",
        files=[("frames", ("frame.jpg", image or _jpeg(), "image/jpeg"))],
        data={
            "manifest": json.dumps(
                {
                    "frames": [
                        {
                            "frameIndex": 7,
                            "fileIndex": 0,
                            "observations": observations,
                        }
                    ]
                }
            )
        },
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
    assert ready.json()["evidenceFingerprintVersion"] == "pixel-evidence-v1"
    assert ready.json()["cache"]["schemaVersion"] == "identity-embedding-cache.v1"


def test_missing_model_keeps_liveness_but_never_claims_readiness():
    app = create_app(UnavailableProvider(), preload=True)
    assert _call(app, "GET", "/health/live").status_code == 200
    ready = _call(app, "GET", "/health/ready")
    assert ready.status_code == 503
    assert "checkpoint is missing" in ready.json()["detail"]
    response = _request(
        app,
        [{"observationId": "obs-1", "bbox": {"x": 10, "y": 10, "width": 40, "height": 70}}],
    )
    assert response.status_code == 503


def test_embeddings_are_normalized_and_keep_provider_evidence():
    provider = FakeProvider()
    app = create_app(
        provider,
        preload=False,
        quality_policy=QualityPolicy(minimum_width=12, minimum_height=20, minimum_sharpness=1),
    )
    response = _request(
        app,
        [{"observationId": "obs-1", "bbox": {"x": 10, "y": 10, "width": 40, "height": 70}}],
    )
    assert response.status_code == 200
    payload = response.json()
    item = payload["items"][0]
    assert item["usable"] is True
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(item["embedding"])), 1.0, atol=1e-6
    )
    assert item["embedding"][:2] == [0.6000000238418579, 0.800000011920929]
    assert item["role"] == "player"
    assert item["visibilityScores"] == [1.0, 0.5]
    assert provider.received == ["obs-1"]


def test_small_and_blurry_crops_are_rejected_without_calling_provider():
    provider = FakeProvider()
    app = create_app(
        provider,
        preload=False,
        quality_policy=QualityPolicy(minimum_width=20, minimum_height=30, minimum_sharpness=20),
    )
    observations = [
        {"observationId": "small", "bbox": {"x": 5, "y": 5, "width": 10, "height": 20}},
        {"observationId": "blurry", "bbox": {"x": 20, "y": 20, "width": 40, "height": 70}},
    ]
    response = _request(app, observations, _jpeg("flat"))
    assert response.status_code == 200
    items = {item["observationId"]: item for item in response.json()["items"]}
    assert items["small"]["usable"] is False
    assert "crop-too-small" in items["small"]["rejectionReasons"]
    assert items["blurry"]["usable"] is False
    assert "crop-too-blurry" in items["blurry"]["rejectionReasons"]
    assert items["small"]["embedding"] is None
    assert items["blurry"]["embedding"] is None
    assert provider.received == []


def _usable_policy(**values) -> QualityPolicy:
    defaults = {
        "minimum_width": 12,
        "minimum_height": 20,
        "minimum_sharpness": 1,
        "padding_ratio": 0.08,
    }
    defaults.update(values)
    return QualityPolicy(**defaults)


def _observation(identifier: str, *, x: float = 10) -> dict:
    return {
        "observationId": identifier,
        "bbox": {"x": x, "y": 10, "width": 40, "height": 70},
    }


def _cache() -> IdentityEmbeddingCache:
    return IdentityEmbeddingCache(
        dimension=256,
        max_entries=16,
        ttl_seconds=60,
        wait_timeout_seconds=5,
    )


def test_identical_second_request_is_served_without_provider_call():
    provider = FakeProvider()
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=_cache(),
        preload=False,
    )
    first = _request(app, [_observation("first")])
    second = _request(app, [_observation("second")])

    assert first.status_code == second.status_code == 200
    assert provider.received == ["first"]
    assert first.json()["diagnostics"]["providerInferenceCount"] == 1
    assert second.json()["diagnostics"]["cacheHitCount"] == 1
    assert second.json()["diagnostics"]["cacheMissCount"] == 0
    assert second.json()["diagnostics"]["providerInferenceCount"] == 0
    assert second.json()["items"][0]["cacheHit"] is True


def test_changed_bbox_is_a_cache_miss_even_for_same_frame_bytes():
    provider = FakeProvider()
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=_cache(),
        preload=False,
    )
    _request(app, [_observation("first", x=10)])
    changed = _request(app, [_observation("changed", x=11)])

    assert provider.received == ["first", "changed"]
    assert changed.json()["diagnostics"]["cacheHitCount"] == 0
    assert changed.json()["diagnostics"]["cacheMissCount"] == 1


def test_model_and_policy_changes_invalidate_shared_cache():
    provider = FakeProvider()
    shared_cache = _cache()
    first_app = create_app(
        provider,
        quality_policy=_usable_policy(minimum_sharpness=1),
        embedding_cache=shared_cache,
        preload=False,
    )
    _request(first_app, [_observation("model-v1")])

    provider.model_version = "fake-model-v2"
    model_changed = _request(first_app, [_observation("model-v2")])
    assert model_changed.json()["diagnostics"]["cacheMissCount"] == 1

    policy_changed_app = create_app(
        provider,
        quality_policy=_usable_policy(minimum_sharpness=2),
        embedding_cache=shared_cache,
        preload=False,
    )
    policy_changed = _request(policy_changed_app, [_observation("policy-v2")])
    assert policy_changed.json()["diagnostics"]["cacheMissCount"] == 1
    assert provider.received == ["model-v1", "model-v2", "policy-v2"]


def test_duplicate_crops_in_one_request_are_inferred_once():
    provider = FakeProvider()
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=_cache(),
        preload=False,
    )
    response = _request(
        app,
        [_observation("duplicate-a"), _observation("duplicate-b")],
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
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=cache,
        preload=False,
    )
    _request(app, [_observation("first")])
    with cache._lock:
        key = next(iter(cache._entries))
        cache._entries[key].value = {"corrupt": True}

    response = _request(app, [_observation("after-corruption")])

    assert response.status_code == 200
    assert provider.received == ["first", "after-corruption"]
    assert response.json()["diagnostics"]["corruptCacheMissCount"] == 1
    assert response.json()["diagnostics"]["cacheMissCount"] == 1
    assert response.json()["items"][0]["cacheSource"] == "provider"


def test_unusable_qa_result_is_cached_without_provider_inference():
    provider = FakeProvider()
    app = create_app(
        provider,
        quality_policy=QualityPolicy(
            minimum_width=100,
            minimum_height=150,
            minimum_sharpness=20,
        ),
        embedding_cache=_cache(),
        preload=False,
    )
    first = _request(app, [_observation("small-1")], _jpeg("flat"))
    second = _request(app, [_observation("small-2")], _jpeg("flat"))

    assert provider.received == []
    assert first.json()["items"][0]["cacheSource"] == "qa-computed"
    assert second.json()["items"][0]["cacheSource"] == "cache-hit"
    assert second.json()["diagnostics"]["cacheHitCount"] == 1


def test_concurrent_identical_requests_share_one_inflight_provider_call():
    provider = FakeProvider(delay=0.15)
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=_cache(),
        preload=False,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_request, app, [_observation("concurrent-a")])
        second = executor.submit(_request, app, [_observation("concurrent-b")])
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
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=cache,
        preload=False,
    )
    _request(app, [_observation("first", x=10)])
    _request(app, [_observation("second", x=30)])
    evicted = _request(app, [_observation("first-again", x=10)])

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
    app = create_app(
        provider,
        quality_policy=_usable_policy(),
        embedding_cache=cache,
        preload=False,
    )
    _request(app, [_observation("before-expiry")])
    time.sleep(0.03)
    expired = _request(app, [_observation("after-expiry")])

    assert provider.received == ["before-expiry", "after-expiry"]
    assert expired.json()["diagnostics"]["expiredCacheMissCount"] == 1
    assert expired.json()["diagnostics"]["cacheMissCount"] == 1
