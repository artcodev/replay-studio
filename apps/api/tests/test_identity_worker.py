import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.identity_worker_client import (
    embed_identity_frames,
    identity_worker_readiness,
)
from app.identity_worker_contract import IdentityWorkerError
from app.identity_worker_transport import IdentityWorkerTransportError
from app.person_crop_store import PersonCropPolicy, PersonCropRecord


def _settings(
    url="http://identity-worker:8091",
    batch_size=2,
    retry_count=0,
    cache_root=None,
):
    return SimpleNamespace(
        identity_worker_url=url,
        identity_worker_timeout=900,
        identity_worker_batch_size=batch_size,
        identity_worker_batch_retry_count=retry_count,
        identity_embedding_cache_enabled=cache_root is not None,
        media_root=str(cache_root) if cache_root is not None else ".",
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
    return f"pixel-evidence-v2:{identifier}"


def _frame_sha(frame_index: int) -> str:
    return sha256(f"frame-{frame_index}".encode("utf-8")).hexdigest()


def _crop_sha(observation_id: str) -> str:
    return sha256(f"crop-{observation_id}".encode("utf-8")).hexdigest()


def _observation(observation_id: str) -> dict:
    return {
        "observationId": observation_id,
        "cropSha256": _crop_sha(observation_id),
        "quality": _quality(),
    }


def _crop_record(observation_id: str) -> PersonCropRecord:
    return PersonCropRecord(
        observation_id=observation_id,
        crop_sha256=_crop_sha(observation_id),
        crop_jpeg=f"jpeg-{observation_id}".encode("utf-8"),
        bbox={"x": 1.0, "y": 2.0, "width": 30.0, "height": 60.0},
        padded_rect=(0, 0, 33, 66),
        quality=_quality(),
        rejection_reasons=(),
    )


def _install_store(monkeypatch, frames) -> None:
    """Serve crop bytes for every request without touching the filesystem."""

    records = {
        frame_sha: {
            str(observation["observationId"]): _crop_record(
                str(observation["observationId"])
            )
            for observation in observations
        }
        for _frame_index, frame_sha, observations in frames
    }
    monkeypatch.setattr(
        "app.identity_worker_client.person_crop_store_runtime",
        lambda: (Path("/unused-person-crops"), PersonCropPolicy()),
    )
    monkeypatch.setattr(
        "app.identity_worker_client.lookup_person_crop_envelope",
        lambda _directory, *, frame_sha256, policy: records.get(frame_sha256),
    )


def _cache():
    return {
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


def test_readiness_is_disabled_without_url(monkeypatch):
    monkeypatch.setattr(
        "app.identity_worker_client.get_settings", lambda: _settings(url="")
    )
    assert identity_worker_readiness() == {
        "configured": False,
        "status": "disabled",
        "backend": None,
    }


def test_readiness_requires_loaded_normalized_model(monkeypatch):
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.fetch_identity_readiness",
        lambda url, timeout: {
            "status": "ready",
            "backend": "prtreid-bpbreid-soccernet",
            "dimension": 256,
            "normalized": True,
            "evidenceFingerprintVersion": "pixel-evidence-v2",
            "device": "cpu",
            "batchSize": 8,
            "modelVersion": "model-v1",
            "modelLoadSeconds": 4.2,
            "soccerNetCommit": "reference-commit",
            "cache": _cache(),
        },
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
        "evidenceFingerprintVersion": "pixel-evidence-v2",
        "modelVersion": "model-v1",
        "modelLoadSeconds": 4.2,
        "soccerNetCommit": "reference-commit",
        "torchVersion": None,
        "mpsFallbackEnabled": False,
    }


def test_readiness_is_nonfatal_when_worker_is_offline(monkeypatch):
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())

    def offline(*_args, **_kwargs):
        raise IdentityWorkerTransportError("offline")

    monkeypatch.setattr(
        "app.identity_worker_client.fetch_identity_readiness", offline
    )
    result = identity_worker_readiness()
    assert result["status"] == "unavailable"
    assert "offline" in result["detail"]


def test_client_batches_frames_and_preserves_rejected_items(monkeypatch):
    frames = [
        (index, _frame_sha(index), [_observation(f"track-1:{index}")])
        for index in range(1, 4)
    ]
    _install_store(monkeypatch, frames)
    calls = []

    def fake_post(url, *, manifest, files, timeout):
        manifest = json.loads(manifest)
        assert manifest["contractVersion"] == 2
        assert len(files) == len(manifest["crops"])
        assert all(field == "crops" for field, _payload in files)
        calls.append([crop["frameIndex"] for crop in manifest["crops"]])
        items = []
        for crop in manifest["crops"]:
            observation_id = crop["observationId"]
            usable = crop["frameIndex"] != 2
            items.append(
                {
                    "observationId": observation_id,
                    "frameIndex": crop["frameIndex"],
                    "usable": usable,
                    "quality": crop["quality"],
                    "rejectionReasons": [] if usable else ["crop-too-blurry"],
                    "embedding": _embedding() if usable else None,
                    "visibilityScores": None,
                    "role": None,
                    "roleConfidence": None,
                    "evidenceFingerprint": _fingerprint(observation_id),
                }
            )
        return {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v2",
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

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings", lambda: _settings(batch_size=2)
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)
    updates = []
    result = embed_identity_frames(
        frames,
        on_progress=lambda completed, total, usable: updates.append((completed, total, usable)),
    )
    assert calls == [[1, 2], [3]]
    assert updates == [(2, 3, 1), (3, 3, 2)]
    assert not isinstance(result, dict)
    assert set(result.items_by_observation_id) == {
        "track-1:1",
        "track-1:2",
        "track-1:3",
    }
    assert (
        result.items_by_observation_id["track-1:1"]["provider"]
        == "prtreid-bpbreid-soccernet"
    )
    assert result.items_by_observation_id["track-1:2"]["usable"] is False
    assert result.items_by_observation_id["track-1:2"]["embedding"] is None
    assert result.diagnostics["requestedObservationCount"] == 3
    assert result.diagnostics["cacheHitCount"] == 2
    assert result.diagnostics["providerInferenceCount"] == 1
    assert result.diagnostics["modelContract"] == {
        "backend": "prtreid-bpbreid-soccernet",
        "modelVersion": "model-v1",
        "dimension": 256,
        "normalized": True,
        "evidenceFingerprintVersion": "pixel-evidence-v2",
    }


def _single_observation_frames(count: int):
    return [
        (frame_index, _frame_sha(frame_index), [_observation(f"obs-{frame_index}")])
        for frame_index in range(1, count + 1)
    ]


def _usable_batch_payload(manifest: str) -> dict:
    items = [
        {
            "observationId": crop["observationId"],
            "frameIndex": crop["frameIndex"],
            "usable": True,
            "quality": crop["quality"],
            "rejectionReasons": [],
            "embedding": _embedding(),
            "visibilityScores": None,
            "role": None,
            "roleConfidence": None,
            "evidenceFingerprint": _fingerprint(crop["observationId"]),
        }
        for crop in json.loads(manifest)["crops"]
    ]
    return {
        "backend": "prtreid-bpbreid-soccernet",
        "dimension": 256,
        "normalized": True,
        "evidenceFingerprintVersion": "pixel-evidence-v2",
        "modelVersion": "model-v1",
        "items": items,
        "diagnostics": {
            "requestedObservationCount": len(items),
            "usableObservationCount": len(items),
            "rejectedObservationCount": 0,
            "cacheHitCount": 0,
            "cacheMissCount": len(items),
            "deduplicatedObservationCount": 0,
            "concurrentDeduplicatedCount": 0,
            "providerInferenceCount": len(items),
            "corruptCacheMissCount": 0,
            "expiredCacheMissCount": 0,
        },
    }


def test_transient_batch_transport_failure_is_retried_once_and_recovers(
    monkeypatch,
):
    frames = _single_observation_frames(2)
    _install_store(monkeypatch, frames)
    attempts = []

    def flaky_post(_url, *, manifest, **_kwargs):
        attempts.append(json.loads(manifest)["crops"][0]["frameIndex"])
        if len(attempts) == 2:
            raise IdentityWorkerTransportError("connection reset")
        return _usable_batch_payload(manifest)

    sleeps = []
    monkeypatch.setattr(
        "app.identity_worker_client.get_settings",
        lambda: _settings(batch_size=1, retry_count=1),
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", flaky_post)
    monkeypatch.setattr(
        "app.identity_worker_client.sleep", lambda value: sleeps.append(value)
    )

    result = embed_identity_frames(frames)

    assert attempts == [1, 2, 2]
    assert sleeps == [0.5]
    assert set(result.items_by_observation_id) == {"obs-1", "obs-2"}
    assert result.diagnostics["retriedBatchCount"] == 1
    assert "partialFailure" not in result.diagnostics


def test_exhausted_batch_retries_keep_earlier_embeddings_as_partial_result(
    monkeypatch,
):
    frames = _single_observation_frames(3)
    _install_store(monkeypatch, frames)

    def failing_second_batch(_url, *, manifest, **_kwargs):
        frame_index = json.loads(manifest)["crops"][0]["frameIndex"]
        if frame_index >= 2:
            raise IdentityWorkerTransportError("worker went away")
        return _usable_batch_payload(manifest)

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings",
        lambda: _settings(batch_size=1, retry_count=1),
    )
    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch", failing_second_batch
    )
    monkeypatch.setattr("app.identity_worker_client.sleep", lambda _value: None)

    result = embed_identity_frames(frames)

    # The first batch survives; the failure is explicit, not an exception.
    assert set(result.items_by_observation_id) == {"obs-1"}
    assert result.diagnostics["partialFailure"] == {
        "failedFrameIndex": 2,
        "processedFrameCount": 1,
        "requestedFrameCount": 3,
        "attempts": 2,
        "detail": "worker went away",
    }


def test_crop_store_miss_degrades_to_explicit_local_rejection(monkeypatch):
    frames = _single_observation_frames(2)
    # Only frame 1 has bytes in the store; frame 2 is a store miss.
    _install_store(monkeypatch, frames[:1])
    posted = []

    def fake_post(_url, *, manifest, **_kwargs):
        posted.extend(
            crop["observationId"] for crop in json.loads(manifest)["crops"]
        )
        return _usable_batch_payload(manifest)

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings", lambda: _settings(batch_size=2)
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)

    result = embed_identity_frames(frames)

    assert posted == ["obs-1"]
    assert result.items_by_observation_id["obs-1"]["usable"] is True
    missing = result.items_by_observation_id["obs-2"]
    assert missing["usable"] is False
    assert missing["rejectionReasons"] == ["crop-store-unavailable"]
    assert result.diagnostics["cropStoreMissCount"] == 1


def _ready_payload(model_version="model-v1"):
    return {
        "status": "ready",
        "backend": "prtreid-bpbreid-soccernet",
        "device": "cpu",
        "batchSize": 8,
        "dimension": 256,
        "normalized": True,
        "evidenceFingerprintVersion": "pixel-evidence-v2",
        "modelVersion": model_version,
        "modelLoadSeconds": 4.2,
        "soccerNetCommit": "reference-commit",
        "cache": {
            "schemaVersion": "identity-embedding-cache.v3",
            "enabled": True,
            "maxEntries": 4096,
            "ttlSeconds": 86400,
            "waitTimeoutSeconds": 30,
            "size": 0,
            "inFlight": 0,
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
        },
    }


def test_disk_cache_survives_worker_restart_and_model_change_reinfers(
    monkeypatch, tmp_path
):
    frames = _single_observation_frames(2)
    _install_store(monkeypatch, frames)
    posted_batches = []
    model_version = {"value": "model-v1"}

    def fake_post(_url, *, manifest, **_kwargs):
        posted_batches.append(
            [crop["frameIndex"] for crop in json.loads(manifest)["crops"]]
        )
        payload = _usable_batch_payload(manifest)
        payload["modelVersion"] = model_version["value"]
        return payload

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings",
        lambda: _settings(batch_size=2, cache_root=tmp_path / "media"),
    )
    monkeypatch.setattr(
        "app.identity_worker_client.fetch_identity_readiness",
        lambda _url, timeout: _ready_payload(model_version["value"]),
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)

    first = embed_identity_frames(frames)
    assert posted_batches == [[1, 2]]
    assert first.diagnostics["diskCacheHitCount"] == 0
    assert first.diagnostics["diskCacheMissCount"] == 2

    def forbidden_post(*_args, **_kwargs):
        raise AssertionError("a warm rebuild must not re-upload cached crops")

    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch", forbidden_post
    )
    second = embed_identity_frames(frames)

    assert set(second.items_by_observation_id) == {"obs-1", "obs-2"}
    assert second.diagnostics["diskCacheHitCount"] == 2
    assert second.diagnostics["diskCacheMissCount"] == 0
    assert second.diagnostics["modelContract"]["modelVersion"] == "model-v1"
    assert (
        second.items_by_observation_id["obs-1"]["embedding"]
        == first.items_by_observation_id["obs-1"]["embedding"]
    )

    # A new model identity is a different cache contract: every observation
    # is re-embedded instead of silently reusing the old model's evidence.
    model_version["value"] = "model-v2"
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)
    third = embed_identity_frames(frames)

    assert posted_batches == [[1, 2], [1, 2]]
    assert third.diagnostics["diskCacheHitCount"] == 0
    assert third.diagnostics["diskCacheMissCount"] == 2


def test_partially_cached_frame_sends_only_the_missed_observations(
    monkeypatch, tmp_path
):
    first_observation = _observation("obs-a")
    second_observation = _observation("obs-b")

    def frame(observations):
        return [(1, _frame_sha(1), observations)]

    _install_store(
        monkeypatch, frame([first_observation, second_observation])
    )
    manifests = []

    def fake_post(_url, *, manifest, **_kwargs):
        manifests.append(json.loads(manifest))
        return _usable_batch_payload(manifest)

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings",
        lambda: _settings(batch_size=2, cache_root=tmp_path / "media"),
    )
    monkeypatch.setattr(
        "app.identity_worker_client.fetch_identity_readiness",
        lambda _url, timeout: _ready_payload(),
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)

    embed_identity_frames(frame([first_observation]))
    result = embed_identity_frames(frame([first_observation, second_observation]))

    # The second run only uploads the observation that was never embedded.
    assert [
        crop["observationId"] for crop in manifests[-1]["crops"]
    ] == ["obs-b"]
    assert set(result.items_by_observation_id) == {"obs-a", "obs-b"}
    assert result.diagnostics["diskCacheHitCount"] == 1
    assert result.diagnostics["diskCacheMissCount"] == 1


def test_overlapping_and_rejected_crops_never_reach_the_worker():
    from app.reconstruction_reid_evidence import identity_embedding_requests
    from app.reconstruction_person_detection_contract import Detection
    import numpy as np

    def person(observation_id, x, width=30.0, *, reasons=(), stored=True):
        detection = Detection(
            x=x,
            y=100.0,
            width=width,
            height=60.0,
            confidence=0.9,
            feature=np.zeros(12, dtype=np.float32),
        )
        detection.observation_id = observation_id
        detection.crop_frame_sha256 = _frame_sha(1) if stored else None
        detection.crop_sha256 = _crop_sha(observation_id) if stored else None
        detection.crop_quality = _quality()
        detection.crop_rejection_reasons = tuple(reasons)
        return detection

    # Two players almost on top of each other plus one clean detection.
    people = [person("obs-a", 50.0), person("obs-b", 55.0), person("obs-clean", 300.0)]

    requests, local_items, diagnostics = identity_embedding_requests(
        [(Path("frame_00001.jpg"), 0.0)],
        [(people, 0.0)],
        overlap_iou_threshold=0.45,
    )

    assert [item["observationId"] for item in requests[0][2]] == ["obs-clean"]
    assert requests[0][1] == _frame_sha(1)
    assert local_items == {}
    assert diagnostics["overlapSkippedObservationCount"] == 2
    assert {
        item["observationId"] for item in diagnostics["overlapSkippedObservations"]
    } == {"obs-a", "obs-b"}

    # A zero threshold disables the filter entirely.
    unfiltered, _local, disabled = identity_embedding_requests(
        [(Path("frame_00001.jpg"), 0.0)],
        [(people, 0.0)],
        overlap_iou_threshold=0.0,
    )
    assert len(unfiltered[0][2]) == 3
    assert disabled["overlapSkippedObservationCount"] == 0

    # Extraction QA rejections and store faults resolve locally instead of
    # travelling to the worker.
    degraded = [
        person("obs-blurry", 50.0, reasons=("crop-too-blurry",)),
        person("obs-lost", 300.0, stored=False),
    ]
    requests, local_items, diagnostics = identity_embedding_requests(
        [(Path("frame_00001.jpg"), 0.0)],
        [(degraded, 0.0)],
        overlap_iou_threshold=0.45,
    )
    assert requests == []
    assert local_items["obs-blurry"]["rejectionReasons"] == ["crop-too-blurry"]
    assert local_items["obs-lost"]["rejectionReasons"] == [
        "crop-store-unavailable"
    ]
    assert diagnostics["cropRejectedObservationCount"] == 1
    assert diagnostics["cropStoreUnavailableObservationCount"] == 1


def test_reid_phase_reports_partial_evidence_with_an_explicit_warning(monkeypatch):
    from app.reconstruction_reid_phase import extract_reid_evidence
    from app.identity_worker_contract import IdentityWorkerBatchResult
    from app.reconstruction_person_detection_contract import Detection
    import numpy as np

    person = Detection(
        x=1.0,
        y=2.0,
        width=30.0,
        height=60.0,
        confidence=0.9,
        feature=np.zeros(12, dtype=np.float32),
    )
    person.observation_id = "obs-1"
    person_frames = [([person], 0.0)]

    monkeypatch.setattr(
        "app.reconstruction_reid_phase.identity_embedding_requests",
        lambda _frames, _person_frames, **_kwargs: (
            [(0, _frame_sha(0), [_observation("obs-1")])],
            {},
            {"overlapSkippedObservationCount": 0},
        ),
    )
    monkeypatch.setattr(
        "app.reconstruction_reid_phase.identity_worker_readiness",
        lambda timeout: {"configured": True, "status": "ready", "backend": "prtreid"},
    )
    partial = IdentityWorkerBatchResult()
    partial.items_by_observation_id["obs-1"] = {
        "usable": True,
        "frameIndex": 0,
        "quality": {},
        "rejectionReasons": [],
        "embedding": _embedding(),
        "evidenceFingerprint": _fingerprint("obs-1"),
        "provider": "prtreid",
        "modelVersion": "model-v1",
    }
    partial.diagnostics["partialFailure"] = {
        "failedFrameIndex": 5,
        "processedFrameCount": 4,
        "requestedFrameCount": 9,
        "attempts": 2,
        "detail": "worker went away",
    }
    monkeypatch.setattr(
        "app.reconstruction_reid_phase.embed_identity_frames",
        lambda _requests, _progress: partial,
    )

    diagnostics, warnings = extract_reid_evidence(
        [(Path("frame.jpg"), 0.0)],
        person_frames,
        SimpleNamespace(update=lambda *args, **kwargs: None),
    )

    assert diagnostics["status"] == "partial"
    assert diagnostics["partialFailure"]["processedFrameCount"] == 4
    assert person.reid_feature is not None
    assert any("only 4/9 frames" in warning for warning in warnings)


def test_client_rejects_model_version_change_between_http_batches(monkeypatch):
    frames = _single_observation_frames(2)
    _install_store(monkeypatch, frames)

    call_count = 0

    def fake_post(_url, *, manifest, **_kwargs):
        nonlocal call_count
        call_count += 1
        crop = json.loads(manifest)["crops"][0]
        observation_id = crop["observationId"]
        return {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v2",
                "modelVersion": f"model-v{call_count}",
                "items": [
                    {
                        "observationId": observation_id,
                        "frameIndex": crop["frameIndex"],
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

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings", lambda: _settings(batch_size=1)
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)

    with pytest.raises(
        IdentityWorkerError,
        match="changed model contract between batches: modelVersion",
    ):
        embed_identity_frames(frames)

    assert call_count == 2


def test_client_rejects_non_normalized_embedding(monkeypatch):
    frames = [(1, _frame_sha(1), [_observation("obs")])]
    _install_store(monkeypatch, frames)
    vector = _embedding()
    vector[0] = 2.0
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch",
        lambda *_args, **_kwargs: {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v2",
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
            },
    )
    with pytest.raises(IdentityWorkerError, match="non-normalized"):
        embed_identity_frames(frames)


def test_readiness_rejects_non_object_json_without_raising(monkeypatch):
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.fetch_identity_readiness",
        lambda *_args, **_kwargs: [],
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
def test_client_rejects_malformed_identity_item(monkeypatch, mutation, message):
    frames = [(1, _frame_sha(1), [_observation("obs")])]
    _install_store(monkeypatch, frames)
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
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch",
        lambda *_args, **_kwargs: {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v2",
                "modelVersion": "model-v1",
                "items": [item],
            },
    )

    with pytest.raises(IdentityWorkerError, match=message):
        embed_identity_frames(frames)


def test_client_rejects_non_object_embedding_response(monkeypatch):
    frames = [(1, _frame_sha(1), [_observation("obs")])]
    _install_store(monkeypatch, frames)
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(IdentityWorkerError, match="top-level JSON"):
        embed_identity_frames(frames)


def test_client_rejects_unknown_wire_fields(monkeypatch):
    frames = [(1, _frame_sha(1), [_observation("obs")])]
    _install_store(monkeypatch, frames)
    monkeypatch.setattr("app.identity_worker_client.get_settings", lambda: _settings())
    monkeypatch.setattr(
        "app.identity_worker_client.post_identity_batch",
        lambda *_args, **_kwargs: {
            "backend": "prtreid-bpbreid-soccernet",
            "dimension": 256,
            "normalized": True,
            "evidenceFingerprintVersion": "pixel-evidence-v2",
            "modelVersion": "model-v1",
            "items": [
                {
                    "observationId": "obs",
                    "frameIndex": 1,
                    "usable": True,
                    "quality": _quality(),
                    "rejectionReasons": [],
                    "embedding": _embedding(),
                    "visibilityScores": None,
                    "role": None,
                    "roleConfidence": None,
                    "evidenceFingerprint": _fingerprint(),
                    "futureField": "must-fail-closed",
                }
            ],
        },
    )

    with pytest.raises(IdentityWorkerError, match="unsupported fields: futureField"):
        embed_identity_frames(frames)


def test_client_reports_duplicate_pixel_evidence_across_http_batches(monkeypatch):
    frames = _single_observation_frames(2)
    _install_store(monkeypatch, frames)

    def fake_post(_url, *, manifest, **_kwargs):
        crop = json.loads(manifest)["crops"][0]
        return {
                "backend": "prtreid-bpbreid-soccernet",
                "dimension": 256,
                "normalized": True,
                "evidenceFingerprintVersion": "pixel-evidence-v2",
                "modelVersion": "model-v1",
                "items": [
                    {
                        "observationId": crop["observationId"],
                        "frameIndex": crop["frameIndex"],
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

    monkeypatch.setattr(
        "app.identity_worker_client.get_settings", lambda: _settings(batch_size=1)
    )
    monkeypatch.setattr("app.identity_worker_client.post_identity_batch", fake_post)

    result = embed_identity_frames(frames)

    assert result.diagnostics["uniqueEvidenceFingerprintCount"] == 1
    assert result.diagnostics["duplicateEvidenceFingerprintCount"] == 1
