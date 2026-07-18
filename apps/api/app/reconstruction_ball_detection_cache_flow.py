from __future__ import annotations

from .ball_detection_cache import (
    delete_ball_detection_checkpoint,
    load_ball_detection_cache,
    load_ball_detection_checkpoint,
    store_clean_ball_detection_cache,
)
from .ball_detection_cache_contract import BallDetectionCacheError
from .ball_detection_contract import BallDetector
from .reconstruction_ball_detection_contract import (
    BallDetectionProgress,
    BallDetectionResult,
    BallFrameDetections,
    DenseBallDetectionSource,
)
from .reconstruction_ball_roi import AdaptiveBallRoiConfig, adaptive_ball_roi_diagnostics


def load_complete_ball_detection_cache(
    source: DenseBallDetectionSource,
    detector: BallDetector,
    detector_input: dict | None,
    adaptive_roi: AdaptiveBallRoiConfig | None,
    on_progress: BallDetectionProgress | None,
) -> BallDetectionResult | None:
    """Return a complete, timestamp-aligned cache hit or explain its rejection."""

    metadata = source.metadata
    if not (
        source.dense_cache_key
        and source.cache_asset_directory is not None
        and isinstance(detector_input, dict)
    ):
        metadata["detectionCacheHit"] = False
        return None
    cached_entry = load_ball_detection_cache(
        source.cache_asset_directory,
        dense_cache_key=source.dense_cache_key,
        detector_input=detector_input,
    )
    if cached_entry is None or cached_entry.primary_backend != detector.backend_name:
        metadata["detectionCacheHit"] = False
        return None
    resolved, batches = cached_entry.as_pipeline_data()
    timestamps_match = len(resolved) == len(source.frames) and all(
        abs(float(cached_time) - float(source_time)) <= 1e-6
        for (_, cached_time), (_, source_time) in zip(resolved, source.frames, strict=True)
    )
    if not timestamps_match or len(batches) != len(source.frames):
        metadata["detectionCacheInvalidReason"] = (
            "cached frame count or timestamps do not match dense frames"
        )
        metadata["detectionCacheHit"] = False
        return None
    backend_names = sorted({str(item.get("backend") or "unknown") for item in batches})
    metadata.update(
        {
            "detectionCacheHit": True,
            "detectionCacheKey": cached_entry.cache_key,
            "failedFrameCount": 0,
            "fallbackFrameCount": 0,
            "circuitBreaker": {"opened": False, "reason": None},
            "backendCounts": {
                backend: sum(item.get("backend") == backend for item in batches)
                for backend in backend_names
            },
            "adaptiveRoi": adaptive_ball_roi_diagnostics(batches, adaptive_roi),
        }
    )
    if on_progress is not None:
        on_progress(
            len(source.frames), len(source.frames), f"{detector.backend_name} · cached detections"
        )
    try:
        delete_ball_detection_checkpoint(
            source.cache_asset_directory,
            dense_cache_key=source.dense_cache_key,
            detector_input=detector_input,
        )
    except (BallDetectionCacheError, OSError):
        pass
    return resolved, metadata, batches, source.warnings


def resume_clean_ball_detection_checkpoint(
    source: DenseBallDetectionSource,
    detector: BallDetector,
    detector_input: dict | None,
    on_progress: BallDetectionProgress | None,
) -> tuple[BallFrameDetections, list[dict]]:
    """Resume only a clean primary-backend prefix with aligned timestamps."""

    resolved: BallFrameDetections = []
    batches: list[dict] = []
    if not (
        source.dense_cache_key
        and source.cache_asset_directory is not None
        and isinstance(detector_input, dict)
        and source.frames
    ):
        source.metadata.setdefault("detectionCheckpointHit", False)
        source.metadata.setdefault("resumedFrameCount", 0)
        return resolved, batches
    checkpoint = load_ball_detection_checkpoint(
        source.cache_asset_directory,
        dense_cache_key=source.dense_cache_key,
        detector_input=detector_input,
        expected_frame_count=len(source.frames),
    )
    if checkpoint is not None and checkpoint.primary_backend == detector.backend_name:
        checkpoint_resolved, checkpoint_batches = checkpoint.as_pipeline_data()
        prefix_length = len(checkpoint_resolved)
        timestamps_match = (
            0 < prefix_length < len(source.frames)
            and len(checkpoint_batches) == prefix_length
            and all(
                abs(float(cached_time) - float(source_time)) <= 1e-6
                for (_, cached_time), (_, source_time) in zip(
                    checkpoint_resolved, source.frames[:prefix_length], strict=True
                )
            )
        )
        if timestamps_match:
            resolved = checkpoint_resolved
            batches = checkpoint_batches
            source.metadata.update(
                {
                    "detectionCheckpointHit": True,
                    "detectionCheckpointKey": checkpoint.cache_key,
                    "resumedFrameCount": prefix_length,
                }
            )
            if on_progress is not None:
                on_progress(
                    prefix_length,
                    len(source.frames),
                    f"{detector.backend_name} · resumed clean checkpoint",
                )
        else:
            source.metadata["detectionCheckpointInvalidReason"] = (
                "checkpoint timestamps do not match the dense-frame prefix"
            )
    source.metadata.setdefault("detectionCheckpointHit", False)
    source.metadata.setdefault("resumedFrameCount", 0)
    return resolved, batches


def initial_adaptive_image_size(
    resolved: BallFrameDetections, batches: list[dict]
) -> tuple[int, int] | None:
    if batches:
        image_size = batches[-1].get("imageSize")
        if (
            isinstance(image_size, list)
            and len(image_size) == 2
            and all(isinstance(value, (int, float)) for value in image_size)
        ):
            return int(image_size[0]), int(image_size[1])
    if resolved and resolved[-1][0]:
        try:
            return (
                int(resolved[-1][0][0]["imageWidth"]),
                int(resolved[-1][0][0]["imageHeight"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
    return None


def publish_clean_ball_detection_cache(
    source: DenseBallDetectionSource,
    detector: BallDetector,
    detector_input: dict | None,
    resolved: BallFrameDetections,
    batches: list[dict],
    *,
    failed_frame_count: int,
    fallback_frame_count: int,
) -> None:
    """Publish only clean cache state; cache IO never invalidates evidence."""

    if not (
        source.dense_cache_key
        and source.cache_asset_directory is not None
        and isinstance(detector_input, dict)
    ):
        return
    try:
        stored_entry = store_clean_ball_detection_cache(
            source.cache_asset_directory,
            dense_cache_key=source.dense_cache_key,
            detector_input=detector_input,
            primary_backend=detector.backend_name,
            resolved_frames=resolved,
            batches=batches,
            failed_frame_count=failed_frame_count,
            fallback_frame_count=fallback_frame_count,
        )
    except (BallDetectionCacheError, OSError) as exc:
        source.metadata["detectionCacheWriteError"] = str(exc)
        return
    if stored_entry is None:
        source.metadata["detectionCacheStored"] = False
        return
    source.metadata["detectionCacheKey"] = stored_entry.cache_key
    source.metadata["detectionCacheStored"] = True
    try:
        delete_ball_detection_checkpoint(
            source.cache_asset_directory,
            dense_cache_key=source.dense_cache_key,
            detector_input=detector_input,
        )
    except (BallDetectionCacheError, OSError) as exc:
        source.metadata["detectionCheckpointDeleteError"] = str(exc)
