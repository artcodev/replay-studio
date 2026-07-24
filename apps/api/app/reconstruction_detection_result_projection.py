from __future__ import annotations

from pathlib import Path

from .reconstruction_calibration_selection import CalibrationSelectionResult
from .reconstruction_dense_ball_phase import DenseBallDetectionResult
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_sampled_detection_preparation import SampledDetectionRuntime
from .reconstruction_sampled_frame_contract import (
    SampledCalibrationInputs,
    SampledCalibrationAnalysis,
    SampledDetectionAnalysis,
)
from .temporal_calibration_contract import TemporalCalibrationResult


def project_frame_analysis_result(
    frames: list[tuple[Path, float]],
    sampled: SampledDetectionAnalysis,
    runtime: SampledDetectionRuntime,
    dense_ball: DenseBallDetectionResult,
    identity_diagnostics: dict,
    identity_warnings: list[str],
) -> FrameAnalysisResult:
    return FrameAnalysisResult(
        frames=frames,
        person_frames=sampled.person_frames,
        ball_frames=dense_ball.frames,
        frame_size=sampled.frame_sizes[max(sampled.frame_sizes, default=0)],
        person_counts=sampled.person_counts,
        ball_counts=dense_ball.counts,
        person_detection_cache_diagnostics=runtime.person_cache_diagnostics,
        ball_detection_batches=dense_ball.batches,
        ball_detection_warnings=dense_ball.warnings,
        ball_dense_frame_metadata=dense_ball.frame_metadata,
        identity_worker_diagnostics=identity_diagnostics,
        identity_warnings=identity_warnings,
    )


def project_calibration_phase_result(
    calibration: SampledCalibrationAnalysis,
    inputs: SampledCalibrationInputs,
    temporal: TemporalCalibrationResult,
    dense_ball: DenseBallDetectionResult,
    selection: CalibrationSelectionResult,
) -> CalibrationPhaseResult:
    return CalibrationPhaseResult(
        calibration=selection.calibration,
        quality=selection.quality,
        coordinate_mode=selection.coordinate_mode,
        metric_calibration=selection.metric,
        frame_evidence=calibration.frame_evidence,
        accepted_frame_calibrations=calibration.accepted_frame_calibrations,
        accepted_automatic_direct_by_sample=(
            calibration.accepted_automatic_direct_by_sample
        ),
        accepted_manual_direct_by_sample=calibration.accepted_manual_direct_by_sample,
        resolved_calibrations_by_sample=temporal.resolved_by_sample,
        manual_override_by_sample=inputs.manual_override_by_sample,
        representative_manual_sample=selection.representative_manual_sample,
        rejected_frame_count=calibration.rejected_frame_count,
        temporal_recovered_frame_count=temporal.recovered_frame_count,
        metric_person_sample_count=temporal.metric_person_sample_count,
        metric_ball_sample_count=dense_ball.metric_sample_count,
        warnings=selection.warnings,
        contact_point_diagnostics=temporal.contact_point_diagnostics,
    )
