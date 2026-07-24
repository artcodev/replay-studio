from __future__ import annotations

"""Detection phase that consumes, but can never produce, calibration."""

from dataclasses import replace
from typing import Mapping

from .reconstruction_calibration_application import (
    apply_snapshot_to_people,
    calibration_impact,
)
from .reconstruction_calibration_snapshot import (
    load_persisted_calibration_snapshot,
)
from .reconstruction_dense_ball_phase import (
    detect_dense_ball_phase,
    skipped_dense_ball_result,
)
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_detection_result_projection import (
    project_frame_analysis_result,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_reid_phase import extract_reid_evidence
from .reconstruction_sampled_detection_preparation import (
    load_sampled_frames,
    prepare_sampled_detectors,
)
from .reconstruction_sampled_frame_detection import (
    analyze_sampled_detections,
)


def detect_with_persisted_calibration_phase(
    scene: dict,
    *,
    model_name: str,
    ball_backend: str,
    ball_detection_input: Mapping,
    ball_detection_profile: str = "automatic",
    contact_point_profile: str = "bbox-bottom",
    progress: ReconstructionProgress,
) -> tuple[FrameAnalysisResult, CalibrationPhaseResult, dict]:
    frames = load_sampled_frames(scene, progress)
    snapshot = load_persisted_calibration_snapshot(scene, frames)
    runtime = prepare_sampled_detectors(
        scene,
        frames,
        model_name=model_name,
        ball_backend=ball_backend,
        ball_detection_input=ball_detection_input,
        progress=progress,
    )
    sampled = analyze_sampled_detections(
        scene,
        frames,
        runtime,
        progress,
    )
    if sampled.frame_sizes != snapshot.frame_sizes:
        raise ReconstructionError(
            "Decoded frame sizes do not match the completed calibration artifact"
        )

    person_application = apply_snapshot_to_people(
        scene,
        sampled.person_frames,
        snapshot,
        contact_point_profile=contact_point_profile,
        progress=progress,
    )

    identity_diagnostics, identity_warnings = extract_reid_evidence(
        frames,
        sampled.person_frames,
        progress,
    )
    if ball_detection_profile == "skip-manual-authoritative":
        dense_ball = skipped_dense_ball_result(ball_detection_profile)
        progress.update(
            "detection",
            3,
            "Skipping dense ball detection",
            "The manual ball trajectory is authoritative for this run.",
            38,
            56,
            completed=1,
            total=1,
        )
    else:
        dense_ball = detect_dense_ball_phase(
            scene,
            detector=runtime.ball_detector,
            fallback_detector=runtime.ball_fallback_detector,
            sampled_frames=frames,
            generic_fallback_ball_frames=sampled.generic_ball_frames,
            detector_input=ball_detection_input,
            backend=ball_backend,
            frame_sizes=snapshot.frame_sizes,
            temporal_calibration=snapshot.temporal,
            frame_evidence=snapshot.result.frame_evidence,
            camera_transforms=snapshot.camera_transforms,
            progress=progress,
        )

    calibration_result = replace(
        snapshot.result,
        metric_person_sample_count=person_application.metric_count,
        metric_ball_sample_count=dense_ball.metric_sample_count,
        contact_point_diagnostics=person_application.contact_point_diagnostics,
    )
    frame_result = project_frame_analysis_result(
        frames,
        sampled,
        runtime,
        dense_ball,
        identity_diagnostics,
        identity_warnings,
    )
    impact = calibration_impact(
        snapshot,
        person_application,
        dense_ball,
        contact_point_profile=contact_point_profile,
    )
    return frame_result, calibration_result, impact


__all__ = ("detect_with_persisted_calibration_phase",)
