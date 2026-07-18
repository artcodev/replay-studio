from pathlib import Path
from types import SimpleNamespace

import numpy as np

import app.reconstruction_calibration_detection as reconstruction_module
from app.pitch_calibration_contract import PitchCalibration


def _frames(times: list[float]) -> list[tuple[Path, float]]:
    return [
        (Path(f"/tmp/frame_{index:05d}.jpg"), timestamp)
        for index, timestamp in enumerate(times)
    ]


def _calibration(frame_index: int) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.9,
        supported_lines=8,
        mean_line_score=0.8,
        rectangle="test",
        method="pnlcalib-points-lines",
        frame_index=frame_index,
    )


def test_anchor_sampling_keeps_first_last_and_bounds_regular_gaps():
    frames = _frames([index / 10 for index in range(36)])

    selected = reconstruction_module.select_calibration_anchor_frames(frames, 1.0)

    assert [path.stem for path, _ in selected] == [
        "frame_00000",
        "frame_00010",
        "frame_00020",
        "frame_00030",
        "frame_00035",
    ]
    assert max(
        right[1] - left[1] for left, right in zip(selected, selected[1:])
    ) <= 1.0


def test_anchor_sampling_retains_both_sides_of_unavoidable_source_gap():
    frames = _frames([0.0, 0.7, 2.1, 2.8])

    selected = reconstruction_module.select_calibration_anchor_frames(frames, 1.0)

    assert selected == frames


def test_automatic_calibration_sends_only_anchor_frames_to_worker(monkeypatch):
    frames = _frames([index / 10 for index in range(36)])
    worker_calls = []
    progress = []
    monkeypatch.setattr(
        reconstruction_module,
        "get_settings",
        lambda: SimpleNamespace(
            calibration_worker_url="http://calibration-worker:8090",
            calibration_anchor_max_gap_seconds=1.0,
        ),
    )

    def calibrate(indexed, on_progress=None, timeout=None):
        worker_calls.append((indexed, timeout))
        result = {index: _calibration(index) for index, _ in indexed}
        if on_progress is not None:
            on_progress(len(indexed), len(indexed), len(result))
        return result

    monkeypatch.setattr(
        reconstruction_module,
        "calibrate_frames_with_worker",
        calibrate,
    )
    monkeypatch.setattr(
        reconstruction_module,
        "local_frame_calibrations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("complete worker anchors must not invoke local fallback")
        ),
    )

    result, warnings = reconstruction_module.automatic_frame_calibrations(
        frames,
        lambda *update: progress.append(update),
        worker_timeout=123.0,
    )

    assert [index for index, _ in worker_calls[0][0]] == [0, 10, 20, 30, 35]
    assert worker_calls[0][1] == 123.0
    assert sorted(result) == [0, 10, 20, 30, 35]
    assert warnings == []
    assert progress[-1][1:3] == (5, 5)


def test_single_frame_calibration_is_never_sampled_away():
    frames = _frames([4.2])

    assert reconstruction_module.select_calibration_anchor_frames(frames, 1.0) == frames
