from __future__ import annotations

"""Calibration-only reconstruction phase.

The gated first stage runs ONLY pitch calibration — line/keypoint homographies
and the temporal camera-graph solve. It deliberately performs no person
detection, no crop extraction, no ReID and no ball inference: those belong to
the full reconstruction that follows the calibration gate. Frames are decoded
directly (no detector model is loaded), and empty people are fed through so the
person-support gates never engage (they are conditional add-ons that only fire
with >= 4 detected people).
"""

from typing import Mapping

import cv2

from .config import get_settings
from .direct_calibration_sampling import (
    resolve_direct_calibration_max_gap_seconds,
)
from .reconstruction_calibration_detection import select_calibration_anchor_frames
from .reconstruction_calibration_selection import select_representative_calibration
from .reconstruction_dense_ball_phase import skipped_dense_ball_result
from .reconstruction_detection_contract import CalibrationPhaseResult
from .reconstruction_detection_result_projection import (
    project_calibration_phase_result,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import source_frame_index
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_pnlcalib_demotion_retry import retry_demoted_pnlcalib_anchors
from .reconstruction_pnlcalib_retry import (
    PnlCalibBatchRequest,
    resolve_pnlcalib_batch_attempts,
)
from .reconstruction_sampled_calibration import (
    SampledCalibrationAccumulator,
    prepare_sampled_calibrations,
)
from .reconstruction_sampled_detection_preparation import load_sampled_frames
from .reconstruction_temporal_calibration_phase import solve_temporal_calibration_phase
from .reconstruction_calibration_incremental_phase import (
    finalize_staged_calibration_phase,
)


def calibrate_only_phase(
    scene: dict,
    *,
    reconstruction_request: Mapping,
    progress: ReconstructionProgress,
) -> CalibrationPhaseResult:
    frames = load_sampled_frames(scene, progress)
    if (
        str(reconstruction_request.get("calibrationTrigger") or "")
        == "manual-draft-finalize"
    ):
        return finalize_staged_calibration_phase(
            scene,
            frames,
            reconstruction_request=reconstruction_request,
            progress=progress,
        )
    calibration_inputs = prepare_sampled_calibrations(
        frames,
        reconstruction_request,
        progress,
    )
    accumulator = SampledCalibrationAccumulator(scene, calibration_inputs)
    settings = get_settings()
    retry_count = max(
        0,
        min(2, int(getattr(settings, "calibration_pnlcalib_retry_count", 2))),
    )
    direct_calibration_max_gap_seconds = (
        resolve_direct_calibration_max_gap_seconds(
            reconstruction_request.get("directCalibrationMaxGapSeconds")
        )
    )
    anchor_sources = {
        source_frame_index(path)
        for path, _ in select_calibration_anchor_frames(
            frames,
            direct_calibration_max_gap_seconds,
        )
    }
    forced_attempt_count = 0
    retry_requests = [
        PnlCalibBatchRequest(
            sample_index=sample_index,
            source_frame_index=source_frame_index(path),
            scene_time=scene_time,
            frame_path=path,
            initial_calibration=calibration_inputs.frame_calibrations.get(
                source_frame_index(path)
            ),
        )
        for sample_index, (path, scene_time) in enumerate(frames)
        if source_frame_index(path) in anchor_sources
        and sample_index not in calibration_inputs.manual_stabilized_by_sample
    ]
    maximum_forced_attempts = max(1, len(anchor_sources) * retry_count)
    retry_progress_fraction = 0.0

    def retry_worker_progress(
        retry_stage: str,
        retry_index: int,
        total_retries: int,
        completed: int,
        total: int,
        valid: int,
        request_seconds: float,
        diagnostics: dict,
    ) -> None:
        nonlocal retry_progress_fraction
        fraction = (
            (retry_index - 1) + completed / max(1, total)
        ) / max(1, total_retries)
        retry_progress_fraction = max(retry_progress_fraction, fraction)
        per_frame = request_seconds / max(
            1,
            int(diagnostics.get("requestedFrameCount") or min(completed, total)),
        )
        progress.update(
            "calibration",
            2,
            (
                "Recheck shot-wide p95 outliers"
                if retry_stage == "shot-p95-demotion"
                else "Retry frame-local rejects"
            ),
            f"{'Shot-wide residual p95 QA' if retry_stage == 'shot-p95-demotion' else 'Frame-local direct QA'} · fresh pass {retry_index}/{total_retries} · "
            f"batch progress {completed}/{total} · {valid} homographies · "
            f"{per_frame:.1f}s/frame.",
            56,
            62,
            completed=completed,
            total=total,
            fraction=retry_progress_fraction,
            eta_padding=max(1.0, per_frame),
        )
        if progress.run_log is not None:
            progress.run_log.event(
                "pnlcalib-worker-batch-finished",
                retryStage=retry_stage,
                retryRound=retry_index,
                maximumRetryRounds=total_retries,
                completed=completed,
                total=total,
                validHomographies=valid,
                requestSeconds=round(request_seconds, 3),
                effectiveSecondsPerFrame=round(per_frame, 3),
                workerDiagnostics=diagnostics,
            )

    def retry_batch_progress(
        retry_stage: str,
        retry_index: int,
        total_retries: int,
        frame_count: int,
        elapsed_seconds: float,
        statuses: dict[str, int],
    ) -> None:
        nonlocal forced_attempt_count, retry_progress_fraction
        forced_attempt_count += frame_count
        retry_progress_fraction = max(
            retry_progress_fraction,
            retry_index / max(1, total_retries),
        )
        outcome = ", ".join(
            f"{status} {count}" for status, count in sorted(statuses.items())
        )
        progress.update(
            "calibration",
            2,
            (
                "Recheck shot-wide p95 outliers"
                if retry_stage == "shot-p95-demotion"
                else "Retry frame-local rejects"
            ),
            f"{'Shot-wide residual p95 QA' if retry_stage == 'shot-p95-demotion' else 'Frame-local direct QA'} · fresh batch pass {retry_index}/{total_retries} · "
            f"{frame_count} frame(s) · {outcome} · "
            f"{forced_attempt_count} forced inference(s).",
            56,
            62,
            completed=forced_attempt_count,
            total=maximum_forced_attempts,
            fraction=retry_progress_fraction,
            eta_padding=3.0,
        )
        if progress.run_log is not None:
            progress.run_log.event(
                "pnlcalib-retry-batch-finished",
                retryStage=retry_stage,
                retryRound=retry_index,
                maximumRetryRounds=total_retries,
                frameCount=frame_count,
                durationSeconds=round(elapsed_seconds, 3),
                outcomes=statuses,
            )

    automatic_observations = resolve_pnlcalib_batch_attempts(
        scene,
        retry_requests,
        additional_attempts=retry_count,
        on_retry_batch=lambda *values: retry_batch_progress(
            "local-direct-qa", *values
        ),
        on_retry_progress=lambda *values: retry_worker_progress(
            "local-direct-qa", *values
        ),
    )
    retried_frame_count = sum(
        len(observation.attempts) > 1
        for observation in automatic_observations.values()
    )
    retry_accepted_count = sum(
        observation.accepted_attempt is not None
        and observation.accepted_attempt > 1
        for observation in automatic_observations.values()
    )

    for sample_index, (path, scene_time) in enumerate(frames):
        image = cv2.imread(str(path))
        if image is None:
            raise ReconstructionError(
                f"Could not decode sampled frame {path.name} for calibration"
            )
        source_index = source_frame_index(path)
        automatic_observation = automatic_observations.get(sample_index)
        accumulator.add_frame(
            sample_index=sample_index,
            source_index=source_index,
            scene_time=scene_time,
            image=image,
            people=[],
            automatic_observation=automatic_observation,
        )
    sampled_calibration = accumulator.result()
    demotion_retry_frame_count = 0
    demotion_retry_recovered_count = 0
    if settings.calibration_anchor_p95_demotion_enabled:
        max_gap_seconds = (
            max(2.0, float(scene["duration"]))
            if calibration_inputs.manual_stabilized_by_sample
            else 2.0
        )
        demotion_retry = retry_demoted_pnlcalib_anchors(
            scene,
            frames,
            sampled_calibration,
            additional_attempts=retry_count,
            residual_floor_pixels=float(
                settings.calibration_anchor_p95_demotion_floor
            ),
            best_quartile_ratio=float(
                settings.calibration_anchor_p95_demotion_ratio
            ),
            max_gap_seconds=max_gap_seconds,
            on_retry_batch=lambda *values: retry_batch_progress(
                "shot-p95-demotion", *values
            ),
            on_retry_progress=lambda *values: retry_worker_progress(
                "shot-p95-demotion", *values
            ),
        )
        sampled_calibration = demotion_retry.analysis
        demotion_retry_frame_count = demotion_retry.retried_frame_count
        demotion_retry_recovered_count = demotion_retry.recovered_frame_count
    progress.update(
        "calibration",
        2,
        "Direct calibration QA complete",
        f"PnLCalib direct QA · {forced_attempt_count} fresh retry inference(s) · "
        f"{retry_accepted_count} local and {demotion_retry_recovered_count} "
        "p95-outlier frame(s) recovered by retry.",
        56,
        62,
        completed=max(1, forced_attempt_count),
        total=max(1, forced_attempt_count),
        fraction=1.0,
        eta_padding=2.0,
    )
    if retried_frame_count or demotion_retry_frame_count:
        calibration_inputs.calibration_warnings.append(
            f"PnLCalib forced {forced_attempt_count} fresh inference(s) in batch rounds: "
            f"{retried_frame_count} frame(s) failed local direct QA, "
            f"{demotion_retry_frame_count} frame(s) were shot-level p95 outliers; "
            f"{retry_accepted_count + demotion_retry_recovered_count} frame(s) "
            "passed the applicable retry QA gate. Final direct/temporal status "
            "is recorded per frame."
        )
    empty_people = [([], scene_time) for _, scene_time in frames]
    temporal = solve_temporal_calibration_phase(
        scene,
        frames,
        sampled_calibration.frame_sizes,
        sampled_calibration.accepted_automatic_direct_by_sample,
        sampled_calibration.accepted_manual_direct_by_sample,
        sampled_calibration.camera_motion_edges,
        sampled_calibration.camera_transforms,
        sampled_calibration.frame_evidence,
        empty_people,
        bool(calibration_inputs.manual_stabilized_by_sample),
        progress,
    )
    if temporal.demoted_anchors:
        calibration_inputs.calibration_warnings.append(
            f"Demoted {len(temporal.demoted_anchors)} direct calibration "
            "anchor(s) whose line-residual tail was an outlier; their frames "
            "were re-solved temporally from healthier anchors."
        )
    selection = select_representative_calibration(
        frames=frames,
        frame_size=sampled_calibration.frame_size,
        frame_evidence=sampled_calibration.frame_evidence,
        accepted_frame_calibrations=sampled_calibration.accepted_frame_calibrations,
        accepted_manual_direct_by_sample=(
            sampled_calibration.accepted_manual_direct_by_sample
        ),
        camera_transforms=sampled_calibration.camera_transforms,
        manual_stabilized_by_sample=calibration_inputs.manual_stabilized_by_sample,
        manual_reference=calibration_inputs.manual_reference,
        rejected_frame_count=sampled_calibration.rejected_frame_count,
        temporal_recovered_frame_count=temporal.recovered_frame_count,
        warnings=calibration_inputs.calibration_warnings,
    )
    dense_ball = skipped_dense_ball_result("calibrate-only")
    return project_calibration_phase_result(
        sampled_calibration,
        calibration_inputs,
        temporal,
        dense_ball,
        selection,
    )


__all__ = ("calibrate_only_phase",)
