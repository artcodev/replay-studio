from __future__ import annotations

import pytest

from app.reconstruction_errors import ReconstructionError
from app.remote_person_detection_provider import (
    require_person_detection_checkpoint,
    validate_person_detection_payload,
    validate_person_detection_readiness,
)


def _runtime() -> dict:
    return {
        "schemaVersion": 1,
        "backend": "ultralytics-yolo",
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
    }


def test_readiness_preserves_explicit_mps_runtime():
    value = validate_person_detection_readiness(
        {
            **_runtime(),
            "status": "ready",
            "modelLoadSeconds": 1.25,
        }
    )

    assert value["device"] == "mps"
    assert value["mpsFallbackEnabled"] is False


def test_analysis_contract_rejects_unknown_and_broken_boxes():
    payload = {
        **_runtime(),
        "image": {"width": 1920, "height": 1080},
        "names": {"0": "person"},
        "boxes": [
            {
                "classId": 0,
                "confidence": 0.9,
                "x1": 10.0,
                "y1": 5.0,
                "x2": 30.0,
                "y2": 50.0,
            }
        ],
        "diagnostics": {
            "decodeSeconds": 0.01,
            "inferenceSeconds": 0.2,
            "requestSeconds": 0.22,
            "boxCount": 1,
            "degenerateBoxCount": 0,
        },
    }

    assert validate_person_detection_payload(payload)["device"] == "mps"
    with pytest.raises(ReconstructionError, match="unknown box fields"):
        validate_person_detection_payload(
            {
                **payload,
                "boxes": [{**payload["boxes"][0], "tensor": "forbidden"}],
            }
        )
    with pytest.raises(ReconstructionError, match="invalid box"):
        validate_person_detection_payload(
            {
                **payload,
                "boxes": [{**payload["boxes"][0], "x2": 9.0}],
            }
        )


def test_remote_worker_rejects_a_different_selected_checkpoint():
    readiness = validate_person_detection_readiness(
        {
            **_runtime(),
            "status": "ready",
            "modelLoadSeconds": 1.25,
        }
    )

    require_person_detection_checkpoint(readiness, "/models/yolo26m.pt")
    with pytest.raises(ReconstructionError, match="requested football.pt.*yolo26m.pt"):
        require_person_detection_checkpoint(readiness, "/models/football.pt")
