from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .ball_detection_contract import BallDetector
from .reconstruction_ball_detection_contract import (
    BallDetectionAttempt,
    BallFrameEvidence,
)
from .reconstruction_ball_roi import AdaptiveBallRoiConfig, adaptive_ball_detect
from .reconstruction_errors import ReconstructionError


def attempt_ball_detection(
    detector: BallDetector,
    fallback_detector: BallDetector | None,
    path: Path,
    *,
    frame_index: int,
    frame_count: int,
    timestamp: float,
    context_paths: tuple[Path, Path],
    adaptive_roi: AdaptiveBallRoiConfig | None,
    adaptive_seeds: list[dict],
    adaptive_image_size: tuple[int, int] | None,
    force_global_reason: str | None,
    failure_policy: str,
    circuit_reason: str | None,
) -> BallDetectionAttempt:
    """Execute one timestamp across the explicit primary/fallback boundary."""

    batch = None
    failure_detail: str | None = None
    next_circuit_reason = circuit_reason
    if circuit_reason is not None and fallback_detector is not None:
        failure_detail = f"circuit-open after {circuit_reason}"
        try:
            batch = fallback_detector.detect(
                path,
                frame_index=frame_index,
                timestamp=timestamp,
                context_frames=context_paths,
            )
        except Exception as fallback_exc:
            failure_detail += f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
    else:
        try:
            if adaptive_roi is not None:
                batch = adaptive_ball_detect(
                    detector,
                    path,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    timestamp=timestamp,
                    context_paths=context_paths,
                    seeds=adaptive_seeds,
                    image_size=adaptive_image_size,
                    config=adaptive_roi,
                    force_global_reason=force_global_reason,
                )
            else:
                batch = detector.detect(
                    path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    context_frames=context_paths,
                )
        except Exception as exc:
            failure_detail = f"{type(exc).__name__}: {exc}"
            if failure_policy == "raise":
                raise ReconstructionError(
                    f"Ball detector {detector.backend_name} failed on dense frame "
                    f"{frame_index + 1}/{frame_count}: {failure_detail}"
                ) from exc
            if fallback_detector is not None:
                try:
                    batch = fallback_detector.detect(
                        path,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        context_frames=context_paths,
                    )
                except Exception as fallback_exc:
                    failure_detail += (
                        f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                    )
                next_circuit_reason = failure_detail
    if batch is not None:
        adapter_fallback_reason = batch.metadata.get("fallbackReason")
        if adapter_fallback_reason and failure_detail is None:
            failure_detail = str(adapter_fallback_reason)
    return BallDetectionAttempt(batch, failure_detail, next_circuit_reason)


def materialize_ball_frame_evidence(
    attempt: BallDetectionAttempt,
    *,
    frame_index: int,
    generic_fallback_items: list[dict],
    generic_fallback_distance: float | None,
    generic_fallback_tolerance: float,
) -> BallFrameEvidence:
    """Normalize model or generic fallback output into reconstruction evidence."""

    if attempt.batch is None:
        detections = [
            {
                **deepcopy(item),
                "detectorBackend": "generic-coco-fallback",
                "candidateId": f"ball-f{frame_index:05d}-generic-{rank:02d}",
                "provenance": {
                    "backend": "generic-coco-fallback",
                    "failureReason": attempt.failure_detail,
                },
            }
            for rank, item in enumerate(generic_fallback_items, start=1)
        ]
        return BallFrameEvidence(
            detections=detections,
            backend="generic-coco-fallback",
            metadata={
                "fallbackReason": attempt.failure_detail,
                "genericFallbackMatchDistanceSeconds": (
                    round(generic_fallback_distance, 5)
                    if generic_fallback_distance is not None
                    else None
                ),
                "genericFallbackMatchToleranceSeconds": round(
                    generic_fallback_tolerance, 5
                ),
                "genericFallbackCandidateAccepted": bool(generic_fallback_items),
            },
            image_size=None,
            detector_failed=True,
        )

    batch = attempt.batch
    detections = batch.as_reconstruction_detections()
    metadata = dict(batch.metadata)
    for rank, item in enumerate(detections, start=1):
        candidate_id = f"ball-f{frame_index:05d}-c{rank:02d}"
        item["candidateId"] = candidate_id
        item["sourceFrameIndex"] = frame_index
        item["imageWidth"] = int(batch.image_size[0])
        item["imageHeight"] = int(batch.image_size[1])
        item["provenance"] = {
            "backend": item.get("detectorBackend") or batch.backend,
            "candidateId": candidate_id,
            "detectorMetadata": deepcopy(item.get("detectorMetadata") or {}),
            "batchMetadata": deepcopy(metadata),
        }
    return BallFrameEvidence(
        detections=detections,
        backend=batch.backend,
        metadata=metadata,
        image_size=batch.image_size,
        detector_failed=False,
    )
