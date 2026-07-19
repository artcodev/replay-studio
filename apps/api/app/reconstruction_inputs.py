from __future__ import annotations

"""Immutable reconstruction input discovery and lazy model loading."""

import os
from pathlib import Path
from threading import Lock

from .checkpoint_identity import file_content_sha256
from .config import get_settings
from .reconstruction_errors import ReconstructionError
from .video_media_paths import video_generation_directory


_models: dict[tuple[str, str | None], object] = {}
_model_lock = Lock()


def _model_cache_key(name: str) -> tuple[str, str | None]:
    candidate = Path(name).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    checkpoint = candidate.resolve()
    if checkpoint.is_file():
        return str(checkpoint), file_content_sha256(checkpoint)
    return name, None


def load_model(model_name: str | None = None):
    name = model_name or get_settings().reconstruction_model
    cache_key = _model_cache_key(name)
    with _model_lock:
        if cache_key not in _models:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/replay-studio-matplotlib")
            from ultralytics import YOLO

            source = cache_key[0]
            stale_keys = [key for key in _models if key[0] == source]
            for stale_key in stale_keys:
                _models.pop(stale_key, None)
            _models[cache_key] = YOLO(name)
    return _models[cache_key]


# The sampled range is derived from source_end * fps and may therefore point
# one or two indexes past the final materialized frame. Only that rounding
# suffix may be absent; any other missing frame is a truncated generation.
_MISSING_FRAME_SUFFIX_TOLERANCE = 2


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
    expected_indexes = list(range(first, last + 1, step))
    resolved: list[tuple[Path, float]] = []
    missing: list[int] = []
    for index in expected_indexes:
        path = frames / f"frame_{index:05d}.jpg"
        if path.exists():
            resolved.append(
                (path, max(0.0, (index - 1) / analysis_fps - source_start))
            )
        else:
            missing.append(index)
    if not resolved:
        raise ReconstructionError(
            "No sampled frames exist for the published video generation; "
            "re-import the asset before reconstructing"
        )
    if missing:
        tail_start = expected_indexes[-len(missing)]
        is_rounding_tail = (
            len(missing) <= _MISSING_FRAME_SUFFIX_TOLERANCE
            and missing == expected_indexes[-len(missing) :]
            and tail_start > expected_indexes[0]
        )
        if not is_rounding_tail:
            raise ReconstructionError(
                f"The published video generation is missing "
                f"{len(missing)} of {len(expected_indexes)} sampled frames "
                f"(first missing: frame_{missing[0]:05d}.jpg); a silently "
                "thinned sample is not reconstructed — re-import the asset"
            )
    return resolved


def source_frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])
