"""Dense, scene-range frame cache used only by the ball pipeline.

Players and calibration intentionally keep their cheaper sampled cadence.  A
football can cross many pixels between those samples, so the ball detector
decodes a separate native-rate sequence from the private source video.  The
cache is deterministic for an asset/range/FPS tuple and is safe to reuse on a
later rebuild.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .config import get_settings
from .scene_frame_exclusions import scene_frame_exclusions


class DenseBallFramesError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DenseBallFrameSet:
    frames: tuple[tuple[Path, float], ...]
    frame_rate: float
    source_start: float
    source_end: float
    cache_key: str
    cache_hit: bool


def _source_video(asset_directory: Path) -> Path:
    sources = sorted(
        path
        for path in asset_directory.glob("source.*")
        if path.is_file() and path.suffix.lower() not in {".jpg", ".json"}
    )
    if not sources:
        raise DenseBallFramesError("The private source video is unavailable")
    return sources[0]


def _cache_contract(scene: dict, source: Path, frame_rate: float) -> dict:
    video = scene["payload"]["videoAsset"]
    source_start = max(0.0, float(video.get("sourceStart") or 0.0))
    source_end = float(
        video.get("sourceEnd")
        or source_start + float(scene.get("duration") or 0.0)
    )
    source_end = max(source_start, source_end)
    stat = source.stat()
    return {
        "schemaVersion": 1,
        "assetId": str(video["id"]),
        "sourceName": source.name,
        "sourceSize": int(stat.st_size),
        "sourceMtimeNs": int(stat.st_mtime_ns),
        "sourceStart": round(source_start, 6),
        "sourceEnd": round(source_end, 6),
        "frameRate": round(frame_rate, 6),
    }


def _cache_key(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:20]


def _read_cached(directory: Path, contract: dict) -> tuple[Path, ...] | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("contract") != contract:
        return None
    paths = tuple(sorted(directory.glob("frame_*.jpg")))
    if not paths or int(manifest.get("frameCount") or 0) != len(paths):
        return None
    return paths


@contextmanager
def _cache_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive, cross-process lock for one dense-frame cache key.

    Lock files live outside ``ball-frames`` so a failed extraction still leaves
    that cache directory free of anything except published frame sets.  The
    file itself is intentionally persistent: unlinking a lock file can let two
    processes lock different inodes for the same cache key.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("a+b")
    except OSError as exc:
        raise DenseBallFramesError(
            f"Could not open the dense ball-frame cache lock: {exc}"
        ) from exc
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            acquired = True
        except OSError as exc:
            raise DenseBallFramesError(
                f"Could not acquire the dense ball-frame cache lock: {exc}"
            ) from exc
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Closing the descriptor below also releases flock ownership.
                pass
        handle.close()


def _extract(source: Path, directory: Path, contract: dict) -> tuple[Path, ...]:
    if not shutil.which("ffmpeg"):
        raise DenseBallFramesError("ffmpeg is required for dense ball frames")
    partial = directory.parent / f".{directory.name}.{uuid4().hex}.partial"
    partial.mkdir(parents=True, exist_ok=False)
    duration = max(0.0, contract["sourceEnd"] - contract["sourceStart"])
    if duration <= 0.0:
        shutil.rmtree(partial, ignore_errors=True)
        raise DenseBallFramesError("The scene range for ball analysis is empty")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{contract['sourceStart']:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"fps={contract['frameRate']:.6f}",
        "-q:v",
        "2",
        str(partial / "frame_%06d.jpg"),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            raise DenseBallFramesError(
                detail[-1] if detail else "ffmpeg could not decode dense ball frames"
            )
        frames = tuple(sorted(partial.glob("frame_*.jpg")))
        if not frames:
            raise DenseBallFramesError("ffmpeg produced no dense ball frames")
        (partial / "manifest.json").write_text(
            json.dumps(
                {"contract": contract, "frameCount": len(frames)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if directory.exists():
            shutil.rmtree(directory)
        partial.rename(directory)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return tuple(sorted(directory.glob("frame_*.jpg")))


def dense_ball_frame_paths(scene: dict) -> DenseBallFrameSet:
    """Return cached dense frames with scene-relative timestamps."""

    settings = get_settings()
    video = scene["payload"]["videoAsset"]
    source_fps = max(1.0, float(video.get("fps") or settings.ball_analysis_frame_rate))
    target_fps = min(source_fps, max(1.0, float(settings.ball_analysis_frame_rate)))
    asset_directory = Path(settings.media_root).resolve() / str(video["id"])
    source = _source_video(asset_directory)
    contract = _cache_contract(scene, source, target_fps)
    key = _cache_key(contract)
    directory = asset_directory / "ball-frames" / key
    directory.parent.mkdir(parents=True, exist_ok=True)
    paths = _read_cached(directory, contract)
    cache_hit = paths is not None
    if paths is None:
        lock_path = asset_directory / ".ball-frame-locks" / f"{key}.lock"
        with _cache_lock(lock_path):
            # Another API worker may have populated the cache while this one
            # waited for the per-key lock.  Rechecking here avoids duplicate
            # ffmpeg work and prevents competing publishers.
            paths = _read_cached(directory, contract)
            cache_hit = paths is not None
            if paths is None:
                paths = _extract(source, directory, contract)
    duration = max(0.0, contract["sourceEnd"] - contract["sourceStart"])
    frames = tuple(
        (path, min(duration, (index - 1) / target_fps))
        for index, path in enumerate(paths, start=1)
    )
    exclusions = scene_frame_exclusions(scene)
    excluded_dense_indices = {
        max(0, min(len(frames) - 1, int(float(item["sceneTime"]) * target_fps + 0.5)))
        for item in exclusions
    } if frames else set()
    if excluded_dense_indices:
        frames = tuple(
            frame
            for index, frame in enumerate(frames)
            if index not in excluded_dense_indices
        )
    filtered_key = (
        sha256(
            json.dumps(
                {
                    "denseCacheKey": key,
                    "excludedSourceFrames": [
                        int(item["sourceFrameIndex"]) for item in exclusions
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:20]
        if exclusions
        else key
    )
    return DenseBallFrameSet(
        frames=frames,
        frame_rate=target_fps,
        source_start=contract["sourceStart"],
        source_end=contract["sourceEnd"],
        cache_key=filtered_key,
        cache_hit=cache_hit,
    )
