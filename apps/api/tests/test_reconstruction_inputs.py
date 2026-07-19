from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app import reconstruction_inputs
from app.reconstruction_errors import ReconstructionError


def _frame_scene(duration: float = 1.0) -> dict:
    return {
        "duration": duration,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "analysisFps": 10.0,
                "sourceStart": 0.0,
                "sourceEnd": duration,
                "generationKey": "gen-1",
            }
        },
    }


def _materialize_frames(tmp_path, indexes) -> None:
    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        (frames / f"frame_{index:05d}.jpg").write_bytes(b"jpeg")


@pytest.fixture
def frame_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        reconstruction_inputs,
        "video_generation_directory",
        lambda _asset_id, _generation_key: tmp_path,
    )
    monkeypatch.setattr(
        reconstruction_inputs,
        "get_settings",
        lambda: SimpleNamespace(reconstruction_frame_rate=10.0),
    )
    return tmp_path


def test_frame_paths_returns_the_complete_expected_sample(frame_runtime):
    # duration 1.0s at 10 fps → expected indexes 1..11.
    _materialize_frames(frame_runtime, range(1, 12))

    resolved = reconstruction_inputs.frame_paths(_frame_scene())

    assert [path.name for path, _ in resolved] == [
        f"frame_{index:05d}.jpg" for index in range(1, 12)
    ]


def test_frame_paths_tolerates_only_the_rounding_tail(frame_runtime):
    # The final index may not exist because last = end*fps + 1 rounds past
    # the materialized range.
    _materialize_frames(frame_runtime, range(1, 11))

    resolved = reconstruction_inputs.frame_paths(_frame_scene())

    assert len(resolved) == 10


def test_frame_paths_fails_closed_on_a_gap_inside_the_sample(frame_runtime):
    _materialize_frames(
        frame_runtime, [index for index in range(1, 12) if index != 5]
    )

    with pytest.raises(ReconstructionError, match="frame_00005.jpg"):
        reconstruction_inputs.frame_paths(_frame_scene())


def test_frame_paths_fails_closed_on_a_truncated_generation(frame_runtime):
    # Half the range is gone: this is a truncated generation, not rounding.
    _materialize_frames(frame_runtime, range(1, 6))

    with pytest.raises(ReconstructionError, match="missing 6 of 11"):
        reconstruction_inputs.frame_paths(_frame_scene())


def test_frame_paths_fails_closed_when_no_frames_exist(frame_runtime):
    (frame_runtime / "frames").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ReconstructionError, match="No sampled frames exist"):
        reconstruction_inputs.frame_paths(_frame_scene())


def test_local_model_cache_uses_content_identity(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"old-weights")
    loaded: list[tuple[str, bytes]] = []

    ultralytics = ModuleType("ultralytics")

    def load(name: str):
        model = (name, checkpoint.read_bytes())
        loaded.append(model)
        return model

    ultralytics.YOLO = load
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setattr(reconstruction_inputs, "_models", {})

    first = reconstruction_inputs.load_model(str(checkpoint))
    stat = checkpoint.stat()
    os.utime(
        checkpoint,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    after_touch = reconstruction_inputs.load_model(str(checkpoint))

    checkpoint.write_bytes(b"new-weights")
    replacement = reconstruction_inputs.load_model(str(checkpoint))

    assert after_touch is first
    assert replacement is not first
    assert loaded == [
        (str(checkpoint), b"old-weights"),
        (str(checkpoint), b"new-weights"),
    ]
    assert len(reconstruction_inputs._models) == 1
