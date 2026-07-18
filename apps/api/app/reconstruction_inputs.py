from __future__ import annotations

"""Immutable reconstruction input discovery and lazy model loading."""

import os
from pathlib import Path
from threading import Lock

from .config import get_settings
from .reconstruction_errors import ReconstructionError
from .video_media_paths import video_generation_directory


_models: dict[str, object] = {}
_model_lock = Lock()


def load_model(model_name: str | None = None):
    name = model_name or get_settings().reconstruction_model
    with _model_lock:
        if name not in _models:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/replay-studio-matplotlib")
            from ultralytics import YOLO

            _models[name] = YOLO(name)
    return _models[name]


def frame_paths(scene: dict) -> list[tuple[Path, float]]:
    video = scene["payload"]["videoAsset"]
    analysis_fps = float(video.get("analysisFps") or 10.0)
    source_start = float(video.get("sourceStart") or 0.0)
    source_end = float(video.get("sourceEnd") or source_start + scene["duration"])
    sample_fps = min(analysis_fps, get_settings().reconstruction_frame_rate)
    step = max(1, round(analysis_fps / sample_fps))
    first = max(1, int(source_start * analysis_fps) + 1)
    last = max(first, int(source_end * analysis_fps) + 1)
    generation_key = str(video.get("generationKey") or "")
    if not generation_key:
        raise ReconstructionError("Scene has no published video generation")
    frames = video_generation_directory(str(video["id"]), generation_key) / "frames"
    return [
        (
            frames / f"frame_{index:05d}.jpg",
            max(0.0, (index - 1) / analysis_fps - source_start),
        )
        for index in range(first, last + 1, step)
        if (frames / f"frame_{index:05d}.jpg").exists()
    ]


def source_frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])
