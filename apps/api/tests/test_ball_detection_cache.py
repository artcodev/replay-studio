from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from app import ball_detection_cache as cache
from app import ball_detection_cache_contract as cache_contract


def _detector_input() -> dict:
    return {
        "schemaVersion": 1,
        "backend": "dedicated-ultralytics",
        "checkpoint": {
            "name": "football-ball-detection.pt",
            "size": 130_000_000,
            "mtimeNs": 123456789,
            "sha256": "checkpoint-sha256",
        },
        "confidence": 0.05,
        "tile": {"size": 640, "overlap": 0.2, "batch": 8},
        "adaptiveRoi": {
            "enabled": True,
            "algorithmVersion": "adaptive-roi-v1",
            "fullScanIntervalFrames": 5,
            "maxRegions": 3,
            "paddingPixels": 320,
            "reacquirePolicy": "same-frame-global-on-miss",
        },
    }


def _pipeline_data(backend: str = "dedicated-ultralytics"):
    resolved = [
        (
            [
                {
                    "x": 101.5,
                    "y": 202.5,
                    "confidence": 0.91,
                    "bbox": [99.0, 200.0, 104.0, 205.0],
                    "detectorBackend": backend,
                    "candidateId": "ball-f00000-c01",
                    "provenance": {"backend": backend},
                }
            ],
            0.0,
        ),
        ([], 0.04),
    ]
    batches = [
        {
            "frameIndex": 0,
            "t": 0.0,
            "backend": backend,
            "candidateCount": 1,
            "imageSize": [1920, 1080],
            "fallbackReason": None,
            "metadata": {"tileCount": 8},
        },
        {
            "frameIndex": 1,
            "t": 0.04,
            "backend": backend,
            "candidateCount": 0,
            "imageSize": [1920, 1080],
            "fallbackReason": None,
            "metadata": {"tileCount": 8},
        },
    ]
    return resolved, batches


def test_contract_key_is_order_independent_and_covers_every_detector_field():
    detector_input = _detector_input()
    reordered = dict(reversed(list(detector_input.items())))

    first = cache_contract.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=detector_input,
    )
    equivalent = cache_contract.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=reordered,
    )
    changed_dense = cache_contract.build_ball_detection_cache_contract(
        dense_cache_key="dense-b",
        detector_input=detector_input,
    )
    changed_model = _detector_input()
    changed_model["checkpoint"]["sha256"] = "replacement-checkpoint"
    changed_checkpoint = cache_contract.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=changed_model,
    )
    changed_strategy = _detector_input()
    changed_strategy["adaptiveRoi"]["fullScanIntervalFrames"] = 4
    changed_adaptive = cache_contract.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=changed_strategy,
    )

    assert first["schemaVersion"] == cache_contract.BALL_DETECTION_CACHE_SCHEMA_VERSION
    assert first["detectorInput"] == detector_input
    assert first["detectorInputFingerprint"] == cache_contract.ball_detection_input_fingerprint(
        detector_input
    )
    assert cache_contract.ball_detection_cache_key(first) == cache_contract.ball_detection_cache_key(
        equivalent
    )
    assert cache_contract.ball_detection_cache_key(first) != cache_contract.ball_detection_cache_key(
        changed_dense
    )
    assert cache_contract.ball_detection_cache_key(first) != cache_contract.ball_detection_cache_key(
        changed_checkpoint
    )
    assert cache_contract.ball_detection_cache_key(first) != cache_contract.ball_detection_cache_key(
        changed_adaptive
    )


def test_clean_primary_round_trip_returns_fresh_pipeline_values(tmp_path: Path):
    resolved, batches = _pipeline_data()

    stored = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )

    assert stored is not None
    assert stored.path.is_file()
    assert stored.path.parent == tmp_path.resolve() / "ball-detections"
    loaded = cache.load_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
    )
    assert loaded is not None
    assert loaded.cache_key == stored.cache_key
    assert loaded.primary_backend == "dedicated-ultralytics"
    cached_resolved, cached_batches = loaded.as_pipeline_data()
    assert cached_resolved == resolved
    assert cached_batches == batches

    cached_resolved[0][0][0]["x"] = -1
    second_resolved, _ = loaded.as_pipeline_data()
    assert second_resolved[0][0][0]["x"] == 101.5


def test_clean_prefix_checkpoint_is_resumable_but_never_a_complete_cache(
    tmp_path: Path,
):
    resolved, batches = _pipeline_data()

    checkpoint = cache.store_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved[:1],
        batches=batches[:1],
        expected_frame_count=2,
    )

    assert checkpoint.path.name.endswith(".partial.json")
    assert cache.load_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
    ) is None
    resumed = cache.load_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        expected_frame_count=2,
    )
    assert resumed is not None
    resumed_data, resumed_batches = resumed.as_pipeline_data()
    assert resumed_data == resolved[:1]
    assert resumed_batches == batches[:1]
    assert cache.load_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        expected_frame_count=3,
    ) is None

    cache.delete_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
    )
    assert not checkpoint.path.exists()


def test_checkpoint_publication_never_moves_a_clean_prefix_backwards(tmp_path: Path):
    resolved, batches = _pipeline_data()

    longest = cache.store_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
        expected_frame_count=3,
    )
    stale = cache.store_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved[:1],
        batches=batches[:1],
        expected_frame_count=3,
    )

    assert len(longest.frames) == 2
    assert len(stale.frames) == 2
    loaded = cache.load_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        expected_frame_count=3,
    )
    assert loaded is not None
    assert len(loaded.frames) == 2


def test_concurrent_checkpoint_writers_leave_the_longest_prefix(tmp_path: Path):
    resolved, batches = _pipeline_data()
    barrier = Barrier(2)

    def publish(prefix_length: int):
        barrier.wait(timeout=2)
        return cache.store_ball_detection_checkpoint(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
            primary_backend="dedicated-ultralytics",
            resolved_frames=resolved[:prefix_length],
            batches=batches[:prefix_length],
            expected_frame_count=3,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, length) for length in (1, 2)]
        for future in futures:
            future.result(timeout=5)

    loaded = cache.load_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        expected_frame_count=3,
    )
    assert loaded is not None
    assert len(loaded.frames) == 2


def test_complete_cache_prevents_a_late_worker_recreating_partial_state(
    tmp_path: Path,
):
    resolved, batches = _pipeline_data()
    partial = cache.store_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved[:1],
        batches=batches[:1],
        expected_frame_count=2,
    )
    complete = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )

    assert complete is not None
    assert not partial.path.exists()
    late = cache.store_ball_detection_checkpoint(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved[:1],
        batches=batches[:1],
        expected_frame_count=2,
    )

    assert late.path == complete.path
    assert not partial.path.exists()


@pytest.mark.parametrize(
    ("batch_backend", "fallback_reason", "expected_failed", "expected_fallback"),
    [
        ("generic-coco-fallback", "worker timeout", 1, 1),
        ("dedicated-ultralytics", "worker timeout", 0, 1),
    ],
)
def test_degraded_frames_are_published_with_explicit_markers(
    tmp_path: Path,
    batch_backend: str,
    fallback_reason: str,
    expected_failed: int,
    expected_fallback: int,
):
    resolved, batches = _pipeline_data()
    batches[0]["backend"] = batch_backend
    batches[0]["fallbackReason"] = fallback_reason
    if batch_backend == "generic-coco-fallback":
        resolved[0][0][0]["detectorBackend"] = batch_backend
        resolved[0][0][0]["provenance"] = {"backend": batch_backend}

    stored = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )

    assert stored.failed_frame_count == expected_failed
    assert stored.fallback_frame_count == expected_fallback
    assert stored.is_clean is False
    loaded = cache.load_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
    )
    assert loaded is not None
    assert loaded.failed_frame_count == expected_failed
    assert loaded.fallback_frame_count == expected_fallback
    cached_resolved, cached_batches = loaded.as_pipeline_data()
    assert cached_resolved == resolved
    assert cached_batches == batches


def test_unmarked_foreign_backend_output_is_rejected_before_publication(
    tmp_path: Path,
):
    resolved, batches = _pipeline_data()
    # A frame from another backend without degradation markers must never be
    # accepted as primary evidence.
    batches[0]["backend"] = "wasb-service"
    batches[0]["fallbackReason"] = None

    with pytest.raises(cache_contract.BallDetectionCacheError, match="structural"):
        cache.store_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
            primary_backend="dedicated-ultralytics",
            resolved_frames=resolved,
            batches=batches,
        )

    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_or_tampered_artifact_is_a_cache_miss(tmp_path: Path):
    resolved, batches = _pipeline_data()
    stored = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )
    assert stored is not None

    envelope = json.loads(stored.path.read_text())
    envelope["payload"]["frames"][0]["detections"][0]["confidence"] = 0.01
    stored.path.write_text(json.dumps(envelope))

    assert (
        cache.load_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
        )
        is None
    )
    stored.path.write_text("not json")
    assert (
        cache.load_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
        )
        is None
    )


def test_non_finite_payload_is_a_corrupt_cache_miss(tmp_path: Path):
    resolved, batches = _pipeline_data()
    stored = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )
    assert stored is not None

    envelope = json.loads(stored.path.read_text())
    envelope["payload"]["frames"][0]["detections"][0]["confidence"] = float("nan")
    stored.path.write_text(json.dumps(envelope))

    assert (
        cache.load_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
        )
        is None
    )


def test_failed_atomic_replace_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resolved, batches = _pipeline_data()
    original = cache.store_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
    )
    assert original is not None
    original_bytes = original.path.read_bytes()

    def fail_replace(_: Path, __: Path) -> None:
        raise OSError("simulated publish failure")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    with pytest.raises(cache_contract.BallDetectionCacheError, match="simulated publish failure"):
        cache.store_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
            primary_backend="dedicated-ultralytics",
            resolved_frames=resolved,
            batches=batches,
        )

    assert original.path.read_bytes() == original_bytes
    assert not list(original.path.parent.glob("*.tmp"))
