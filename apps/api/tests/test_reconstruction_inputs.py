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
                "fps": 10.0,
                "analysisFps": 10.0,
                "sourceStart": 0.0,
                "sourceEnd": duration,
                "generationKey": "gen-1",
                "analysisFrameInput": {
                    "schemaVersion": 1,
                    "source": "uploaded-video",
                    "coordinateSpace": "source-video-pixels",
                    "width": 1920,
                    "height": 1080,
                    "resize": "none",
                },
            }
        },
    }


def test_frame_paths_rejects_a_legacy_derived_frame_generation(frame_runtime):
    scene = _frame_scene()
    scene["payload"]["videoAsset"].pop("analysisFrameInput")

    with pytest.raises(ReconstructionError, match="legacy derived-frame generation"):
        reconstruction_inputs.frame_paths(scene)


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
        lambda: SimpleNamespace(
            reconstruction_frame_rate=10.0,
            analysis_frame_rate=0.0,
            analysis_frame_rate_cap=25.0,
        ),
    )
    return tmp_path


def test_frame_paths_returns_the_complete_expected_sample(frame_runtime):
    # duration 1.0s at 10 fps → expected indexes 1..11.
    _materialize_frames(frame_runtime, range(1, 12))

    resolved = reconstruction_inputs.frame_paths(_frame_scene())

    assert [path.name for path, _ in resolved] == [
        f"frame_{index:05d}.jpg" for index in range(1, 12)
    ]


def test_frame_paths_omits_exclusions_but_native_inspection_keeps_them(
    frame_runtime,
):
    _materialize_frames(frame_runtime, range(1, 12))
    scene = _frame_scene()
    scene["payload"]["videoAsset"]["frameExclusions"] = [
        {"sourceFrameIndex": 5, "sceneTime": 0.4},
    ]

    analyzed = reconstruction_inputs.frame_paths(scene)
    inspectable = reconstruction_inputs.native_frame_paths(scene)

    assert "frame_00005.jpg" not in [path.name for path, _ in analyzed]
    assert "frame_00005.jpg" in [path.name for path, _ in inspectable]


def test_first_sample_never_precedes_the_segment_start(frame_runtime):
    # Segments are cut at shot changes: a fractional start (4.079s at 10 fps)
    # must begin at frame 42 (t=4.1s), never at frame 41 (t=4.0s) — that
    # frame belongs to the previous camera shot and used to produce a
    # confidently wrong first-frame calibration.
    _materialize_frames(frame_runtime, range(40, 53))
    scene = _frame_scene(duration=1.0)
    scene["payload"]["videoAsset"]["sourceStart"] = 4.079
    scene["payload"]["videoAsset"]["sourceEnd"] = 5.079

    resolved = reconstruction_inputs.frame_paths(scene)

    assert resolved[0][0].name == "frame_00042.jpg"
    assert all(
        (index - 1) / 10.0 >= 4.079 - 1e-6
        for index in (
            int(path.stem.split("_")[1]) for path, _ in resolved
        )
    )
    # Scene times stay relative to the segment start.
    assert resolved[0][1] == pytest.approx(0.021, abs=1e-6)

    # An exact boundary keeps the frame that sits precisely on it.
    scene["payload"]["videoAsset"]["sourceStart"] = 4.0
    scene["payload"]["videoAsset"]["sourceEnd"] = 5.0
    resolved = reconstruction_inputs.frame_paths(scene)
    assert resolved[0][0].name == "frame_00041.jpg"
    assert resolved[0][1] == pytest.approx(0.0, abs=1e-9)


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


def test_resolve_analysis_frame_rate_always_keeps_the_native_source_cadence():
    assert reconstruction_inputs.resolve_analysis_frame_rate(25.0) == 25.0
    assert reconstruction_inputs.resolve_analysis_frame_rate(29.97) == 29.97
    assert reconstruction_inputs.resolve_analysis_frame_rate(59.94) == 59.94
    assert reconstruction_inputs.resolve_analysis_frame_rate(24.0) == 24.0


def test_native_sampling_rejects_an_older_capped_generation(frame_runtime):
    scene = _frame_scene()
    scene["payload"]["videoAsset"]["fps"] = 29.97
    scene["payload"]["videoAsset"]["analysisFps"] = 25.0

    with pytest.raises(ReconstructionError, match="Regenerate source-resolution"):
        reconstruction_inputs.frame_paths(scene)


def test_full_fps_sampling_consumes_every_materialized_frame(frame_runtime, monkeypatch):
    monkeypatch.setattr(
        reconstruction_inputs, "video_generation_directory",
        lambda _a, _g: frame_runtime,
    )
    _materialize_frames(frame_runtime, range(1, 26))
    scene = _frame_scene(duration=1.0)
    scene["payload"]["videoAsset"]["fps"] = 25.0
    scene["payload"]["videoAsset"]["analysisFps"] = 25.0

    resolved = reconstruction_inputs.frame_paths(scene)

    assert len(resolved) == 25
    assert resolved[1][1] == pytest.approx(0.04, abs=1e-6)  # 1/25 s cadence


def test_reduced_sampling_uses_timestamps_not_a_rounded_integer_stride(
    frame_runtime,
    monkeypatch,
):
    monkeypatch.setattr(
        reconstruction_inputs,
        "video_generation_directory",
        lambda _a, _g: frame_runtime,
    )
    _materialize_frames(frame_runtime, range(1, 27))
    scene = _frame_scene(duration=1.0)
    scene["payload"]["videoAsset"]["fps"] = 25.0
    scene["payload"]["videoAsset"]["analysisFps"] = 25.0

    resolved = reconstruction_inputs.frame_paths(scene, sampling_frame_rate=10.0)

    assert [path.name for path, _ in resolved] == [
        "frame_00001.jpg",
        "frame_00004.jpg",
        "frame_00006.jpg",
        "frame_00009.jpg",
        "frame_00011.jpg",
        "frame_00014.jpg",
        "frame_00016.jpg",
        "frame_00019.jpg",
        "frame_00021.jpg",
        "frame_00024.jpg",
        "frame_00026.jpg",
    ]
