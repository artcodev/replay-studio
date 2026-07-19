from __future__ import annotations

"""State machine coordinating dense ball detection capabilities."""

from copy import deepcopy
from pathlib import Path

from .ball_detection_cache import store_ball_detection_checkpoint
from .ball_detection_cache_codec import batch_degradation_counts, frame_degradation
from .ball_detection_cache_contract import BallDetectionCacheError
from .ball_detection_contract import BallDetector
from .config import get_settings
from .reconstruction_ball_detection_attempt import (
    attempt_ball_detection,
    materialize_ball_frame_evidence,
)
from .reconstruction_ball_detection_contract import BallDetectionAttempt
from .reconstruction_ball_detection_cache_flow import (
    ball_degradation_warnings,
    initial_adaptive_image_size,
    load_complete_ball_detection_cache,
    publish_ball_detection_cache,
    resume_ball_detection_checkpoint,
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

    resolved, batches = resume_ball_detection_checkpoint(
        source, detector, detector_input, on_progress
    )
    failed_frame_count, fallback_frame_count = batch_degradation_counts(batches)
    settings = get_settings()
    checkpoint_interval = max(1, int(settings.ball_detection_checkpoint_interval))
    resolved_failure_policy = str(
        failure_policy or settings.ball_detection_failure_policy
    )
    if resolved_failure_policy not in {"raise", "fallback"}:
        raise ReconstructionError(
            "BALL_DETECTION_FAILURE_POLICY must be raise or fallback"
        )
    primary_circuit_reason: str | None = None
    circuit_retry_interval = max(
        1, int(settings.ball_detection_circuit_retry_interval)
    )
    circuit_cooldown_remaining = 0
    circuit_transitions: list[dict] = []
    circuit_opened_count = 0

    def record_circuit_transition(frame_index: int, event: str, reason: str | None):
        if len(circuit_transitions) < 40:
            circuit_transitions.append(
                {"frameIndex": frame_index, "event": event, "reason": reason}
            )
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
    resumed_degraded = bool(batches) and any(frame_degradation(batches[-1]))
    adaptive_seeds = (
        deepcopy(resolved[-1][0]) if resolved and not resumed_degraded else []
    )
    adaptive_image_size = initial_adaptive_image_size(resolved, batches)
    force_global_reason: str | None = (
        "previous-frame-fallback" if resumed_degraded else None
    )
    previous_dense_time = float(resolved[-1][1]) if resolved else None

    def flush_frame_bookkeeping(frame_index: int, evidence) -> None:
        # A degraded frame keeps its explicit markers inside the checkpoint,
        # so an interrupted run resumes instead of recomputing the prefix.
        if (
            source.dense_cache_key
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

    # Batched sequence transport: one multipart request per contiguous run of
    # frames. Only active when the queued detector input opted in, the
    # detector supports it and no adaptive ROI is involved. A failed request
    # deactivates batching and the remaining frames flow through the ordinary
    # per-frame fallback/circuit path below.
    batched_requested = (
        adaptive_roi is None
        and isinstance(detector_input, dict)
        and str(detector_input.get("wasbTransport") or "") == "batched-sequence"
        and callable(getattr(detector, "detect_sequence", None))
    )
    if batched_requested:
        chunk_size = max(3, int(detector_input.get("wasbBatchSize") or 9))
        batched_request_count = 0
        batched_fallback_reason: str | None = None
        while len(resolved) < len(source_frames):
            start_index = len(resolved)
            chunk = source_frames[start_index : start_index + chunk_size]
            try:
                sequence = detector.detect_sequence(
                    [
                        (path, start_index + offset, float(timestamp))
                        for offset, (path, timestamp) in enumerate(chunk)
                    ]
                )
            except Exception as exc:
                batched_fallback_reason = f"{type(exc).__name__}: {exc}"
                break
            if len(sequence) != len(chunk):
                batched_fallback_reason = (
                    "batched transport returned an incomplete sequence"
                )
                break
            batched_request_count += 1
            for offset, (batch, (path, timestamp)) in enumerate(
                zip(sequence, chunk, strict=True)
            ):
                frame_index = start_index + offset
                attempt = BallDetectionAttempt(batch, None, None)
                evidence = materialize_ball_frame_evidence(
                    attempt,
                    frame_index=frame_index,
                    generic_fallback_items=generic_fallback_by_frame.get(
                        frame_index, []
                    ),
                    generic_fallback_distance=(
                        generic_fallback_distance_by_frame.get(frame_index)
                    ),
                    generic_fallback_tolerance=generic_fallback_tolerance,
                )
                resolved.append((evidence.detections, float(timestamp)))
                batches.append(
                    {
                        "frameIndex": frame_index,
                        "t": round(float(timestamp), 4),
                        "backend": evidence.backend,
                        "candidateCount": len(evidence.detections),
                        "imageSize": (
                            list(evidence.image_size)
                            if evidence.image_size
                            else None
                        ),
                        "fallbackReason": None,
                        "scanMode": evidence.metadata.get("scanMode"),
                        "metadata": evidence.metadata,
                    }
                )
                flush_frame_bookkeeping(frame_index, evidence)
                previous_dense_time = float(timestamp)
        source_metadata["batchedTransport"] = {
            "requested": True,
            "requestCount": batched_request_count,
            "framesPerRequest": chunk_size,
            **(
                {"fallbackReason": batched_fallback_reason}
                if batched_fallback_reason
                else {}
            ),
        }

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

        # Half-open circuit: after the cooldown the primary detector is probed
        # again instead of degrading every remaining frame to the fallback.
        probe_primary = (
            primary_circuit_reason is None or circuit_cooldown_remaining <= 0
        )
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
            circuit_reason=None if probe_primary else primary_circuit_reason,
        )
        if attempt.circuit_reason is None:
            if primary_circuit_reason is not None:
                record_circuit_transition(frame_index, "closed", None)
            circuit_cooldown_remaining = 0
        elif probe_primary:
            circuit_opened_count += 1
            record_circuit_transition(
                frame_index,
                "opened" if primary_circuit_reason is None else "probe-failed",
                attempt.circuit_reason,
            )
            circuit_cooldown_remaining = circuit_retry_interval
        else:
            circuit_cooldown_remaining -= 1
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
            force_global_reason = "previous-frame-fallback"
            adaptive_seeds = []
        else:
            force_global_reason = None
            adaptive_seeds = deepcopy(evidence.detections)
            if evidence.image_size is not None:
                adaptive_image_size = tuple(map(int, evidence.image_size))
        flush_frame_bookkeeping(frame_index, evidence)
        previous_dense_time = float(timestamp)

    warnings.extend(
        ball_degradation_warnings(
            failed_frame_count,
            fallback_frame_count,
            len(source_frames),
        )
    )
    source_metadata.update(
        {
            "failedFrameCount": failed_frame_count,
            "fallbackFrameCount": fallback_frame_count,
            "circuitBreaker": {
                "opened": primary_circuit_reason is not None,
                "reason": primary_circuit_reason,
                "retryIntervalFrames": circuit_retry_interval,
                "openedCount": circuit_opened_count,
                "transitions": circuit_transitions,
            },
            "backendCounts": {
                backend: sum(item["backend"] == backend for item in batches)
                for backend in sorted({str(item["backend"]) for item in batches})
            },
            "adaptiveRoi": adaptive_ball_roi_diagnostics(batches, adaptive_roi),
        }
    )
    publish_ball_detection_cache(
        source,
        detector,
        detector_input,
        resolved,
        batches,
    )
    return resolved, source_metadata, batches, warnings
