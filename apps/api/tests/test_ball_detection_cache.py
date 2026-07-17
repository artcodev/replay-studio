from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import ball_detection_cache as cache


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

    first = cache.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=detector_input,
    )
    equivalent = cache.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=reordered,
    )
    changed_dense = cache.build_ball_detection_cache_contract(
        dense_cache_key="dense-b",
        detector_input=detector_input,
    )
    changed_model = _detector_input()
    changed_model["checkpoint"]["sha256"] = "replacement-checkpoint"
    changed_checkpoint = cache.build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=changed_model,
    )

    assert first["schemaVersion"] == cache.BALL_DETECTION_CACHE_SCHEMA_VERSION
    assert first["detectorInput"] == detector_input
    assert first["detectorInputFingerprint"] == cache.ball_detection_input_fingerprint(
        detector_input
    )
    assert cache.ball_detection_cache_key(first) == cache.ball_detection_cache_key(
        equivalent
    )
    assert cache.ball_detection_cache_key(first) != cache.ball_detection_cache_key(
        changed_dense
    )
    assert cache.ball_detection_cache_key(first) != cache.ball_detection_cache_key(
        changed_checkpoint
    )


def test_clean_primary_round_trip_returns_fresh_pipeline_values(tmp_path: Path):
    resolved, batches = _pipeline_data()

    stored = cache.store_clean_ball_detection_cache(
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


@pytest.mark.parametrize(
    ("failed", "fallback", "batch_backend", "fallback_reason"),
    [
        (1, 0, "dedicated-ultralytics", None),
        (0, 1, "dedicated-ultralytics", None),
        (0, 0, "legacy-coco-fallback", None),
        (0, 0, "dedicated-ultralytics", "worker timeout"),
    ],
)
def test_degraded_or_non_primary_output_is_never_published(
    tmp_path: Path,
    failed: int,
    fallback: int,
    batch_backend: str,
    fallback_reason: str | None,
):
    resolved, batches = _pipeline_data()
    batches[0]["backend"] = batch_backend
    batches[0]["fallbackReason"] = fallback_reason

    stored = cache.store_clean_ball_detection_cache(
        tmp_path,
        dense_cache_key="dense-a",
        detector_input=_detector_input(),
        primary_backend="dedicated-ultralytics",
        resolved_frames=resolved,
        batches=batches,
        failed_frame_count=failed,
        fallback_frame_count=fallback,
    )

    assert stored is None
    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_or_tampered_artifact_is_a_cache_miss(tmp_path: Path):
    resolved, batches = _pipeline_data()
    stored = cache.store_clean_ball_detection_cache(
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


def test_failed_atomic_replace_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resolved, batches = _pipeline_data()
    original = cache.store_clean_ball_detection_cache(
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
    with pytest.raises(cache.BallDetectionCacheError, match="simulated publish failure"):
        cache.store_clean_ball_detection_cache(
            tmp_path,
            dense_cache_key="dense-a",
            detector_input=_detector_input(),
            primary_backend="dedicated-ultralytics",
            resolved_frames=resolved,
            batches=batches,
        )

    assert original.path.read_bytes() == original_bytes
    assert not list(original.path.parent.glob("*.tmp"))
