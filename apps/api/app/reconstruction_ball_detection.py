from __future__ import annotations

"""State machine coordinating dense ball detection capabilities."""

from copy import deepcopy
from pathlib import Path

from .ball_detection_cache import store_ball_detection_checkpoint
from .ball_detection_cache_contract import BallDetectionCacheError
from .ball_detection_contract import BallDetector
from .config import get_settings
from .reconstruction_ball_detection_attempt import (
    attempt_ball_detection,
    materialize_ball_frame_evidence,
)
from .reconstruction_ball_detection_cache_flow import (
    initial_adaptive_image_size,
    load_complete_ball_detection_cache,
    publish_clean_ball_detection_cache,
    resume_clean_ball_detection_checkpoint,
)
from .reconstruction_ball_detection_contract import (
    BallDetectionProgress,
    BallDetectionResult,
    BallFrameDetections,
)
from .reconstruction_ball_detection_source import (
    index_generic_ball_fallbacks,
    resolve_dense_ball_detection_source,
)
from .reconstruction_ball_roi import (
    adaptive_ball_roi_config,
    adaptive_ball_roi_diagnostics,
    scene_camera_cut_between,
)
from .reconstruction_errors import ReconstructionError


def detect_ball_frames(
    scene: dict,
    detector: BallDetector,
    fallback_detector: BallDetector | None,
    sampled_frames: list[tuple[Path, float]],
    generic_fallback_ball_frames: BallFrameDetections,
    on_progress: BallDetectionProgress | None = None,
    *,
    failure_policy: str | None = None,
    detector_input: dict | None = None,
) -> BallDetectionResult:
    """Run dense ball detection while preserving explicit fallback provenance."""

    source = resolve_dense_ball_detection_source(scene, sampled_frames, detector_input)
    source_frames = source.frames
    source_metadata = source.metadata
    warnings = source.warnings
    adaptive_roi = adaptive_ball_roi_config(detector, detector_input)

    cached_result = load_complete_ball_detection_cache(
        source, detector, detector_input, adaptive_roi, on_progress
    )
    if cached_result is not None:
        return cached_result

    resolved, batches = resume_clean_ball_detection_checkpoint(
        source, detector, detector_input, on_progress
    )
    failed_frame_count = 0
    fallback_frame_count = 0
    settings = get_settings()
    checkpoint_prefix_open = True
    checkpoint_interval = max(1, int(settings.ball_detection_checkpoint_interval))
    resolved_failure_policy = str(
        failure_policy or settings.ball_detection_failure_policy
    )
    if resolved_failure_policy not in {"raise", "fallback"}:
        raise ReconstructionError(
            "BALL_DETECTION_FAILURE_POLICY must be raise or fallback"
        )
    primary_circuit_reason: str | None = None
    source_paths = [Path(path) for path, _ in source_frames]
    (
        generic_fallback_by_frame,
        generic_fallback_distance_by_frame,
        generic_fallback_tolerance,
    ) = index_generic_ball_fallbacks(
        source_frames,
        generic_fallback_ball_frames,
        float(source_metadata.get("frameRate") or 1.0),
    )
    adaptive_seeds = deepcopy(resolved[-1][0]) if resolved else []
    adaptive_image_size = initial_adaptive_image_size(resolved, batches)
    force_global_reason: str | None = None
    previous_dense_time = float(resolved[-1][1]) if resolved else None

    for frame_index, (path, timestamp) in enumerate(
        source_frames[len(resolved) :], start=len(resolved)
    ):
        context_paths = (
            source_paths[max(0, frame_index - 1)],
            source_paths[min(len(source_paths) - 1, frame_index + 1)],
        )
        camera_cut = (
            adaptive_roi is not None
            and previous_dense_time is not None
            and scene_camera_cut_between(
                scene, previous_dense_time, float(timestamp)
            )
        )
        if camera_cut:
            adaptive_seeds = []
            force_global_reason = "camera-cut"

        attempt = attempt_ball_detection(
            detector,
            fallback_detector,
            path,
            frame_index=frame_index,
            frame_count=len(source_frames),
            timestamp=float(timestamp),
            context_paths=context_paths,
            adaptive_roi=adaptive_roi,
            adaptive_seeds=adaptive_seeds,
            adaptive_image_size=adaptive_image_size,
            force_global_reason=force_global_reason,
            failure_policy=resolved_failure_policy,
            circuit_reason=primary_circuit_reason,
        )
        primary_circuit_reason = attempt.circuit_reason
        if attempt.failure_detail is not None:
            fallback_frame_count += 1

        evidence = materialize_ball_frame_evidence(
            attempt,
            frame_index=frame_index,
            generic_fallback_items=generic_fallback_by_frame.get(frame_index, []),
            generic_fallback_distance=generic_fallback_distance_by_frame.get(frame_index),
            generic_fallback_tolerance=generic_fallback_tolerance,
        )
        if evidence.detector_failed:
            failed_frame_count += 1
        resolved.append((evidence.detections, float(timestamp)))
        batches.append(
            {
                "frameIndex": frame_index,
                "t": round(float(timestamp), 4),
                "backend": evidence.backend,
                "candidateCount": len(evidence.detections),
                "imageSize": list(evidence.image_size) if evidence.image_size else None,
                "fallbackReason": attempt.failure_detail,
                "scanMode": evidence.metadata.get("scanMode"),
                "metadata": evidence.metadata,
            }
        )
        clean_primary_frame = (
            attempt.failure_detail is None
            and evidence.backend == detector.backend_name
            and attempt.batch is not None
        )
        if not clean_primary_frame:
            checkpoint_prefix_open = False
            force_global_reason = "previous-frame-fallback"
            adaptive_seeds = []
        else:
            force_global_reason = None
            adaptive_seeds = deepcopy(evidence.detections)
            if evidence.image_size is not None:
                adaptive_image_size = tuple(map(int, evidence.image_size))
            if (
                checkpoint_prefix_open
                and source.dense_cache_key
                and source.cache_asset_directory is not None
                and isinstance(detector_input, dict)
                and len(resolved) < len(source_frames)
                and len(resolved) % checkpoint_interval == 0
            ):
                try:
                    stored_checkpoint = store_ball_detection_checkpoint(
                        source.cache_asset_directory,
                        dense_cache_key=source.dense_cache_key,
                        detector_input=detector_input,
                        primary_backend=detector.backend_name,
                        resolved_frames=resolved,
                        batches=batches,
                        expected_frame_count=len(source_frames),
                    )
                except (BallDetectionCacheError, OSError) as exc:
                    source_metadata["detectionCheckpointWriteError"] = str(exc)
                else:
                    source_metadata["detectionCheckpointKey"] = stored_checkpoint.cache_key
                    source_metadata["checkpointedFrameCount"] = len(resolved)
        if on_progress is not None:
            scan_detail = str(evidence.metadata.get("scanMode") or "")
            on_progress(
                frame_index + 1,
                len(source_frames),
                f"{evidence.backend} · {scan_detail or 'default'} · "
                f"{len(evidence.detections)} candidate(s)",
            )
        previous_dense_time = float(timestamp)

    if failed_frame_count:
        warnings.append(
            f"Ball detector failed on {failed_frame_count}/{len(source_frames)} frames; "
            "explicit generic COCO fallback candidates were retained where available."
        )
    if fallback_frame_count:
        warnings.append(
            f"The requested ball detector used an explicit fallback on "
            f"{fallback_frame_count}/{len(source_frames)} frames; inspect backendCounts "
            "and per-frame fallbackReason."
        )
    source_metadata.update(
        {
            "failedFrameCount": failed_frame_count,
            "fallbackFrameCount": fallback_frame_count,
            "circuitBreaker": {
                "opened": primary_circuit_reason is not None,
                "reason": primary_circuit_reason,
            },
            "backendCounts": {
                backend: sum(item["backend"] == backend for item in batches)
                for backend in sorted({str(item["backend"]) for item in batches})
            },
            "adaptiveRoi": adaptive_ball_roi_diagnostics(batches, adaptive_roi),
        }
    )
    publish_clean_ball_detection_cache(
        source,
        detector,
        detector_input,
        resolved,
        batches,
        failed_frame_count=failed_frame_count,
        fallback_frame_count=fallback_frame_count,
    )
    return resolved, source_metadata, batches, warnings
