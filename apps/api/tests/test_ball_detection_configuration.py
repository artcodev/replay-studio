from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from app import ball_detection_configuration as configuration
from app.ball_detection_cache_contract import (
    ball_detection_input_fingerprint,
    build_ball_detection_cache_contract,
)
from app.reconstruction_errors import ReconstructionError
from app.scene_document import reconstruction_input_fingerprint


def _settings(checkpoint, *, failure_policy: str = "raise"):
    return SimpleNamespace(
        ball_detection_backend="dedicated-ultralytics",
        ball_detection_model=str(checkpoint),
        ball_detection_confidence=0.05,
        ball_detection_image_size=1280,
        ball_detection_tile_size=640,
        ball_detection_tile_overlap=0.2,
        ball_detection_inference_batch_size=4,
        ball_detection_nms_iou=0.1,
        ball_detection_full_scan_interval=5,
        ball_detection_roi_region_count=3,
        ball_detection_roi_padding=320,
        ball_detection_failure_policy=failure_policy,
        ball_detection_max_candidates=12,
        ball_analysis_frame_rate=25.0,
        ball_wasb_worker_url="http://ball-worker:8092/v1/detections",
        ball_wasb_timeout=30.0,
    )


def _scene(detector_input: dict) -> dict:
    return {
        "payload": {
            "videoAsset": {
                "id": "video-a",
                "sourceStart": 0.0,
                "sourceEnd": 1.0,
                "analysisFps": 10.0,
                "reconstruction": {
                    "model": "person.pt",
                    "ballBackend": "dedicated-ultralytics",
                    "ballDetectionInput": detector_input,
                },
            }
        }
    }


def test_detector_contract_and_cache_key_ignore_checkpoint_mtime(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"same-weights")
    monkeypatch.setattr(
        configuration,
        "get_settings",
        lambda: _settings(checkpoint),
    )

    first = configuration.ball_detection_input()
    first_cache_contract = build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=first,
    )
    stat = checkpoint.stat()
    os.utime(
        checkpoint,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    second = configuration.ball_detection_input()
    second_cache_contract = build_ball_detection_cache_contract(
        dense_cache_key="dense-a",
        detector_input=second,
    )

    assert first == second
    assert first["checkpoint"]["sha256"]
    assert "mtimeNs" not in first["checkpoint"]
    assert ball_detection_input_fingerprint(first) == ball_detection_input_fingerprint(
        second
    )
    assert first_cache_contract == second_cache_contract
    assert reconstruction_input_fingerprint(
        _scene(first)
    ) == reconstruction_input_fingerprint(_scene(second))


def test_detector_contract_changes_when_same_size_checkpoint_content_changes(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "ball.pt"
    checkpoint.write_bytes(b"old-weights")
    monkeypatch.setattr(
        configuration,
        "get_settings",
        lambda: _settings(checkpoint),
    )

    first = configuration.ball_detection_input()
    checkpoint.write_bytes(b"new-weights")
    second = configuration.ball_detection_input()

    assert first["checkpoint"]["size"] == second["checkpoint"]["size"]
    assert first["checkpoint"]["sha256"] != second["checkpoint"]["sha256"]
    assert ball_detection_input_fingerprint(first) != ball_detection_input_fingerprint(
        second
    )
    assert reconstruction_input_fingerprint(
        _scene(first)
    ) != reconstruction_input_fingerprint(_scene(second))


def test_required_checkpoint_must_exist_when_run_is_queued(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pt"
    monkeypatch.setattr(
        configuration,
        "get_settings",
        lambda: _settings(missing),
    )

    with pytest.raises(ReconstructionError, match="does not exist"):
        configuration.ball_detection_input()
