from __future__ import annotations

from typing import Mapping

from .reconstruction_calibration_selection import select_representative_calibration
from .reconstruction_dense_ball_phase import detect_dense_ball_phase
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_detection_result_projection import (
    project_calibration_phase_result,
    project_frame_analysis_result,
)
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_reid_phase import extract_reid_evidence
from .reconstruction_sampled_calibration import prepare_sampled_calibrations
from .reconstruction_sampled_detection_preparation import (
    load_sampled_frames,
    prepare_sampled_detectors,
)
from .reconstruction_sampled_frame_detection import analyze_sampled_frames
from .reconstruction_temporal_calibration_phase import (
    solve_temporal_calibration_phase,
)


def detect_and_calibrate_phase(
    scene: dict,
    *,
    model_name: str,
    reconstruction_request: Mapping,
    ball_backend: str,
    ball_detection_input: Mapping,
    progress: ReconstructionProgress,
) -> tuple[FrameAnalysisResult, CalibrationPhaseResult]:
    frames = load_sampled_frames(scene, progress)
    calibration_inputs = prepare_sampled_calibrations(
        frames,
        reconstruction_request,
        progress,
    )
    runtime = prepare_sampled_detectors(
        scene,
        frames,
        model_name=model_name,
        ball_backend=ball_backend,
        ball_detection_input=ball_detection_input,
        progress=progress,
    )
    sampled = analyze_sampled_frames(
        scene,
        frames,
        runtime,
        calibration_inputs,
        progress,
    )
    identity_diagnostics, identity_warnings = extract_reid_evidence(
        frames,
        sampled.person_frames,
        progress,
    )
    temporal = solve_temporal_calibration_phase(
        scene,
        frames,
        sampled.calibration.frame_sizes,
        sampled.calibration.accepted_automatic_direct_by_sample,
        sampled.calibration.accepted_manual_direct_by_sample,
        sampled.calibration.camera_motion_edges,
        sampled.calibration.camera_transforms,
        sampled.calibration.frame_evidence,
        sampled.person_frames,
        bool(calibration_inputs.manual_stabilized_by_sample),
        progress,
    )
    dense_ball = detect_dense_ball_phase(
        scene,
        detector=runtime.ball_detector,
        fallback_detector=runtime.ball_fallback_detector,
        sampled_frames=frames,
        generic_fallback_ball_frames=sampled.generic_ball_frames,
        detector_input=ball_detection_input,
        backend=ball_backend,
        frame_sizes=sampled.calibration.frame_sizes,
        temporal_calibration=temporal,
        frame_evidence=sampled.calibration.frame_evidence,
        camera_transforms=sampled.calibration.camera_transforms,
        progress=progress,
    )
    selection = select_representative_calibration(
        frames=frames,
        frame_size=sampled.calibration.frame_size,
        frame_evidence=sampled.calibration.frame_evidence,
        accepted_frame_calibrations=sampled.calibration.accepted_frame_calibrations,
        accepted_manual_direct_by_sample=(
            sampled.calibration.accepted_manual_direct_by_sample
        ),
        camera_transforms=sampled.calibration.camera_transforms,
        manual_stabilized_by_sample=calibration_inputs.manual_stabilized_by_sample,
        manual_reference=calibration_inputs.manual_reference,
        rejected_frame_count=sampled.calibration.rejected_frame_count,
        temporal_recovered_frame_count=temporal.recovered_frame_count,
        warnings=calibration_inputs.calibration_warnings,
    )
    return (
        project_frame_analysis_result(
            frames,
            sampled,
            runtime,
            dense_ball,
            identity_diagnostics,
            identity_warnings,
        ),
        project_calibration_phase_result(
            sampled,
            calibration_inputs,
            temporal,
            dense_ball,
            selection,
        ),
    )
