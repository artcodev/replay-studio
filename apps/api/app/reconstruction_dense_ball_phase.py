from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .ball_detection_contract import BallDetector
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_ball_detection import detect_ball_frames
from .reconstruction_ball_candidate_projection import apply_dense_ball_projection
from .reconstruction_dense_ball_projection_context import dense_ball_projection_context
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_temporal_calibration_phase import TemporalCalibrationResult


@dataclass(frozen=True)
class DenseBallDetectionResult:
    frames: list[tuple[list[dict], float]]
    frame_metadata: dict
    batches: list[dict]
    warnings: list[str]
    counts: list[int]
    metric_sample_count: int


def _queued_detector_policy(detector_input: Mapping) -> tuple[float, str]:
    analysis_frame_rate = float(detector_input.get("analysisFrameRate") or 0.0)
    if not np.isfinite(analysis_frame_rate) or analysis_frame_rate <= 0:
        raise ValueError("Queued ball detector input has no valid analysisFrameRate")
    failure_policy = str(detector_input.get("failurePolicy") or "")
    if failure_policy not in {"raise", "fallback"}:
        raise ValueError("Queued ball detector input has no valid failurePolicy")
    return analysis_frame_rate, failure_policy


def detect_dense_ball_phase(
    scene: dict,
    *,
    detector: BallDetector,
    fallback_detector: BallDetector | None,
    sampled_frames: list[tuple[Path, float]],
    generic_fallback_ball_frames: list[tuple[list[dict], float]],
    detector_input: Mapping,
    backend: str,
    frame_sizes: Mapping[int, tuple[int, int]],
    temporal_calibration: TemporalCalibrationResult,
    frame_evidence: list[dict],
    camera_transforms: Mapping[int, np.ndarray],
    progress: ReconstructionProgress,
) -> DenseBallDetectionResult:
    analysis_frame_rate, failure_policy = _queued_detector_policy(detector_input)
    progress.update(
        "detection",
        3,
        "Preparing dense ball analysis",
        (
            f"Decoding up to {analysis_frame_rate:g} FPS "
            f"for {backend}; player/calibration samples stay unchanged."
        ),
        62,
        84,
        completed=0,
        total=max(1, round(float(scene["duration"]) * analysis_frame_rate)),
        eta_padding=5.0,
    )

    def ball_progress(completed: int, total: int, detail: str) -> None:
        progress.update(
            "detection",
            3,
            "Detecting and scoring ball hypotheses",
            f"Dense ball frame {completed}/{total} · {detail}.",
            62,
            84,
            completed=completed,
            total=total,
            eta_padding=3.0,
        )

    (
        ball_frames,
        frame_metadata,
        detection_batches,
        detection_warnings,
    ) = detect_ball_frames(
        scene,
        detector,
        fallback_detector,
        sampled_frames,
        generic_fallback_ball_frames,
        ball_progress,
        failure_policy=failure_policy,
        detector_input=detector_input,
    )
    ball_counts = [len(detections) for detections, _ in ball_frames]

    metric_ball_sample_count = 0
    sampled_times = [float(time) for _, time in sampled_frames]
    for ball_frame_index, (balls, ball_time) in enumerate(ball_frames):
        if not sampled_frames:
            continue
        projection_context = dense_ball_projection_context(
            float(ball_time),
            sampled_times,
            frame_sizes,
            temporal_calibration.resolved_by_sample,
            temporal_calibration.anchor_by_sample,
            temporal_calibration.uncertainty_by_sample,
            frame_evidence,
            camera_transforms,
        )
        metric_ball_sample_count += apply_dense_ball_projection(
            balls,
            projection_context,
            scene["payload"]["pitch"],
            ball_frame_index,
        )
    return DenseBallDetectionResult(
        frames=ball_frames,
        frame_metadata=frame_metadata,
        batches=detection_batches,
        warnings=detection_warnings,
        counts=ball_counts,
        metric_sample_count=metric_ball_sample_count,
    )
