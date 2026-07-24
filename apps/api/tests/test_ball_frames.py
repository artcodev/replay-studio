from __future__ import annotations

import fcntl
import json
import multiprocessing
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from app import ball_frames


def _hold_cache_lock(lock_path: str, ready: object, release: object) -> None:
    with ball_frames._cache_lock(Path(lock_path)):
        ready.set()
        release.wait(timeout=10)


def _scene(
    *,
    asset_id: str = "asset-ball-frames",
    source_start: float = 0.0,
    source_end: float | None = 1.0,
    source_fps: float = 30.0,
    duration: float = 1.0,
) -> dict:
    video = {
        "id": asset_id,
        "fps": source_fps,
        "sourceStart": source_start,
    }
    if source_end is not None:
        video["sourceEnd"] = source_end
    return {
        "duration": duration,
        "payload": {"videoAsset": video},
    }


def _source(root: Path, asset_id: str = "asset-ball-frames") -> Path:
    directory = root / asset_id
    directory.mkdir(parents=True)
    source = directory / "source.mp4"
    source.write_bytes(b"deterministic source identity")
    return source


def _settings(root: Path, frame_rate: float = 25.0) -> SimpleNamespace:
    return SimpleNamespace(
        media_root=str(root),
        ball_analysis_frame_rate=frame_rate,
    )


def _successful_ffmpeg(frame_count: int, calls: list[list[str]]):
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        pattern = Path(command[-1])
        for index in range(1, frame_count + 1):
            frame = pattern.parent / f"frame_{index:06d}.jpg"
            frame.write_bytes(f"frame-{index}".encode())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_dense_ball_frames_reuse_deterministic_cache(monkeypatch, tmp_path: Path):
    _source(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(ball_frames, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(ball_frames.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        ball_frames.subprocess,
        "run",
        _successful_ffmpeg(3, calls),
    )

    first = ball_frames.dense_ball_frame_paths(_scene())
    second = ball_frames.dense_ball_frame_paths(_scene())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.frames == second.frames
    assert len(calls) == 1
    assert all(path.is_file() for path, _ in second.frames)


def test_dense_ball_frames_recheck_cache_after_lock_wait(monkeypatch, tmp_path: Path):
    source = _source(tmp_path)
    scene = _scene()
    settings = _settings(tmp_path)
    monkeypatch.setattr(ball_frames, "get_settings", lambda: settings)

    contract = ball_frames._cache_contract(scene, source, settings.ball_analysis_frame_rate)
    key = ball_frames._cache_key(contract)
    directory = source.parent / "ball-frames" / key

    @contextmanager
    def completed_by_other_worker(_: Path) -> Iterator[None]:
        directory.mkdir(parents=True)
        (directory / "frame_000001.jpg").write_bytes(b"published by peer")
        (directory / "manifest.json").write_text(
            json.dumps({"contract": contract, "frameCount": 1})
        )
        yield

    def unexpected_extract(*_: object) -> tuple[Path, ...]:
        raise AssertionError("cache must be rechecked before extracting")

    monkeypatch.setattr(ball_frames, "_cache_lock", completed_by_other_worker)
    monkeypatch.setattr(ball_frames, "_extract", unexpected_extract)

    result = ball_frames.dense_ball_frame_paths(scene)

    assert result.cache_hit is True
    assert [path.name for path, _ in result.frames] == ["frame_000001.jpg"]


def test_dense_ball_frame_lock_is_exclusive_across_processes(tmp_path: Path):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    lock_path = tmp_path / "locks" / "same-cache-key.lock"
    process = context.Process(
        target=_hold_cache_lock,
        args=(str(lock_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=5), "child process did not acquire the cache lock"
        with lock_path.open("a+b") as competing_handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competing_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert process.exitcode == 0


def test_dense_ball_frame_timestamps_are_relative_to_scene_range(monkeypatch, tmp_path: Path):
    _source(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(ball_frames, "get_settings", lambda: _settings(tmp_path, 10.0))
    monkeypatch.setattr(ball_frames.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        ball_frames.subprocess,
        "run",
        _successful_ffmpeg(4, calls),
    )

    result = ball_frames.dense_ball_frame_paths(
        _scene(source_start=7.25, source_end=7.55, source_fps=50.0, duration=0.3)
    )

    assert result.source_start == pytest.approx(7.25)
    assert result.source_end == pytest.approx(7.55)
    assert [timestamp for _, timestamp in result.frames] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3]
    )
    command = calls[0]
    assert command[command.index("-ss") + 1] == "7.250000"
    assert command[command.index("-t") + 1] == "0.300000"


def test_dense_ball_frames_omit_the_moment_of_an_excluded_source_frame(
    monkeypatch,
    tmp_path: Path,
):
    _source(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(ball_frames, "get_settings", lambda: _settings(tmp_path, 10.0))
    monkeypatch.setattr(ball_frames.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        ball_frames.subprocess,
        "run",
        _successful_ffmpeg(4, calls),
    )
    scene = _scene(source_fps=20.0, source_end=0.3, duration=0.3)
    scene["payload"]["videoAsset"]["frameExclusions"] = [
        {"sourceFrameIndex": 3, "sceneTime": 0.1},
    ]

    result = ball_frames.dense_ball_frame_paths(scene)

    assert [timestamp for _, timestamp in result.frames] == pytest.approx(
        [0.0, 0.2, 0.3]
    )
    assert result.cache_key != ball_frames._cache_key(
        ball_frames._cache_contract(
            scene,
            tmp_path / "asset-ball-frames" / "source.mp4",
            10.0,
        )
    )


def test_dense_ball_frame_rate_is_capped_by_source_fps(monkeypatch, tmp_path: Path):
    _source(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(ball_frames, "get_settings", lambda: _settings(tmp_path, 25.0))
    monkeypatch.setattr(ball_frames.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        ball_frames.subprocess,
        "run",
        _successful_ffmpeg(2, calls),
    )

    result = ball_frames.dense_ball_frame_paths(_scene(source_fps=12.0))

    assert result.frame_rate == pytest.approx(12.0)
    command = calls[0]
    assert command[command.index("-vf") + 1] == "fps=12.000000"


def test_ffmpeg_failure_removes_partial_dense_frame_directory(monkeypatch, tmp_path: Path):
    source = _source(tmp_path)
    monkeypatch.setattr(ball_frames, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(ball_frames.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        ball_frames,
        "uuid4",
        lambda: SimpleNamespace(hex="failed-extraction"),
    )

    def fail(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="decoder exploded\n",
        )

    monkeypatch.setattr(ball_frames.subprocess, "run", fail)

    with pytest.raises(ball_frames.DenseBallFramesError, match="decoder exploded"):
        ball_frames.dense_ball_frame_paths(_scene())

    cache_root = source.parent / "ball-frames"
    assert cache_root.is_dir()
    assert list(cache_root.iterdir()) == []
