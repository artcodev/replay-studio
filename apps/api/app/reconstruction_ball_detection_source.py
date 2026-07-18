from __future__ import annotations

from pathlib import Path

from .ball_frames import DenseBallFramesError, dense_ball_frame_paths
from .config import get_settings
from .reconstruction_ball_detection_contract import (
    BallFrameDetections,
    DenseBallDetectionSource,
)


def resolve_dense_ball_detection_source(
    scene: dict,
    sampled_frames: list[tuple[Path, float]],
    detector_input: dict | None,
) -> DenseBallDetectionSource:
    """Resolve dense inputs and make the sampled-frame fallback observable."""

    warnings: list[str] = []
    try:
        dense = dense_ball_frame_paths(scene)
        cache_asset_directory = (
            Path(get_settings().media_root).resolve()
            / str(scene["payload"]["videoAsset"]["id"])
            if isinstance(detector_input, dict)
            else None
        )
        return DenseBallDetectionSource(
            frames=list(dense.frames),
            metadata={
                "source": "dense-source-cache",
                "frameRate": round(dense.frame_rate, 3),
                "frameCount": len(dense.frames),
                "cacheKey": dense.cache_key,
                "cacheHit": dense.cache_hit,
                "sourceStart": dense.source_start,
                "sourceEnd": dense.source_end,
            },
            dense_cache_key=dense.cache_key,
            cache_asset_directory=cache_asset_directory,
            warnings=warnings,
        )
    except (DenseBallFramesError, KeyError, OSError, ValueError) as exc:
        warnings.append(
            f"Dense ball frames were unavailable; sampled frames were used: {exc}"
        )
        return DenseBallDetectionSource(
            frames=sampled_frames,
            metadata={
                "source": "sampled-frame-fallback",
                "frameRate": float(
                    scene.get("payload", {})
                    .get("videoAsset", {})
                    .get("analysisFps")
                    or get_settings().reconstruction_frame_rate
                ),
                "frameCount": len(sampled_frames),
                "cacheHit": False,
                "fallbackReason": str(exc),
            },
            dense_cache_key=None,
            cache_asset_directory=None,
            warnings=warnings,
        )


def index_generic_ball_fallbacks(
    source_frames: list[tuple[Path, float]],
    generic_fallback_ball_frames: BallFrameDetections,
    frame_rate: float,
) -> tuple[dict[int, list[dict]], dict[int, float], float]:
    """Assign every sparse generic observation to at most one dense frame."""

    tolerance = 0.51 / max(1.0, frame_rate)
    items_by_frame: dict[int, list[dict]] = {}
    distance_by_frame: dict[int, float] = {}
    if not source_frames:
        return items_by_frame, distance_by_frame, tolerance
    dense_times = [float(item[1]) for item in source_frames]
    for fallback_items, fallback_time in generic_fallback_ball_frames:
        if not fallback_items:
            continue
        dense_index = min(
            range(len(dense_times)),
            key=lambda index: abs(dense_times[index] - float(fallback_time)),
        )
        distance = abs(dense_times[dense_index] - float(fallback_time))
        if distance > tolerance:
            continue
        items_by_frame.setdefault(dense_index, []).extend(fallback_items)
        distance_by_frame[dense_index] = min(
            distance, distance_by_frame.get(dense_index, distance)
        )
    return items_by_frame, distance_by_frame, tolerance
