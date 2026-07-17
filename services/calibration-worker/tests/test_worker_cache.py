from collections import OrderedDict
from threading import Lock
from types import MethodType

import torch

from app.main import DecodedFrame, PnLCalibEngine


def _frame(frame_index: int, content_hash: str) -> DecodedFrame:
    return DecodedFrame(
        frame_index=frame_index,
        width=1920,
        height=1080,
        tensor=torch.zeros(1),
        content_sha256=content_hash,
    )


def _engine(results_by_hash: dict[str, dict | None]):
    engine = object.__new__(PnLCalibEngine)
    engine.device = torch.device("cpu")
    engine.batch_size = 2
    engine.cache_max_entries = 8
    engine.cache_ttl_seconds = 3600.0
    engine.model_version = "test-model-v1"
    engine.lock = Lock()
    engine._cache = OrderedDict()
    engine.model_load_seconds = 0.0
    calls: list[list[str]] = []

    def infer_batch(self, frames, timings):
        calls.append([frame.content_sha256 for frame in frames])
        return [results_by_hash[frame.content_sha256] for frame in frames]

    engine._infer_batch = MethodType(infer_batch, engine)
    return engine, calls


def _calibration(frame_index: int) -> dict:
    return {
        "frameIndex": frame_index,
        "method": "pnlcalib-points-lines",
        "confidence": 0.9,
        "imageToPitch": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_identical_content_is_deduplicated_then_served_from_cache():
    engine, calls = _engine(
        {
            "same": _calibration(1),
            "other": _calibration(3),
        }
    )
    diagnostics = {}

    first = engine.calibrate(
        [_frame(1, "same"), _frame(2, "same"), _frame(3, "other")],
        diagnostics,
    )

    assert calls == [["same", "other"]]
    assert [item["frameIndex"] for item in first] == [1, 2, 3]
    assert diagnostics["requestedFrameCount"] == 3
    assert diagnostics["uniqueFrameCount"] == 2
    assert diagnostics["cacheHitCount"] == 0
    assert diagnostics["cacheMissCount"] == 2
    assert diagnostics["deduplicatedFrameCount"] == 1
    assert diagnostics["inferenceBatchCount"] == 1
    assert diagnostics["cacheEntryCount"] == 2

    warm_diagnostics = {}
    warm = engine.calibrate(
        [_frame(11, "other"), _frame(12, "same")],
        warm_diagnostics,
    )

    assert calls == [["same", "other"]]
    assert [item["frameIndex"] for item in warm] == [11, 12]
    assert warm_diagnostics["cacheHitCount"] == 2
    assert warm_diagnostics["cacheMissCount"] == 0
    assert warm_diagnostics["inferenceBatchCount"] == 0
    assert warm_diagnostics["modelInferenceSeconds"] == 0.0


def test_cache_key_includes_model_version():
    engine, calls = _engine({"same": _calibration(1)})

    engine.calibrate([_frame(1, "same")])
    engine.model_version = "test-model-v2"
    diagnostics = {}
    engine.calibrate([_frame(2, "same")], diagnostics)

    assert calls == [["same"], ["same"]]
    assert diagnostics["modelVersion"] == "test-model-v2"
    assert diagnostics["cacheHitCount"] == 0
    assert diagnostics["cacheMissCount"] == 1


def test_failed_calibration_is_cached_too():
    engine, calls = _engine({"unusable": None})

    assert engine.calibrate([_frame(1, "unusable")]) == []
    diagnostics = {}
    assert engine.calibrate([_frame(2, "unusable")], diagnostics) == []

    assert calls == [["unusable"]]
    assert diagnostics["cacheHitCount"] == 1
    assert diagnostics["cacheMissCount"] == 0
    assert diagnostics["inferenceBatchCount"] == 0

