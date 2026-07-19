from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Callable, Mapping

from . import analysis_runtime as _analysis_runtime
from .ball_detection_configuration import ball_detection_input as _ball_detection_input
from .config import get_settings as _get_settings
from .artifact_store import ArtifactStore as _ArtifactStore
from .reconstruction_artifact_hydration import (
    hydrate_scene_reconstruction as _hydrate_scene_reconstruction,
)
from .reconstruction_ball_phase import (
    BallTrajectoryPhaseResult as _BallTrajectoryPhaseResult,
    resolve_ball_phase as _resolve_ball_phase,
)
from .reconstruction_detection_phase import (
    detect_and_calibrate_phase as _detect_and_calibrate_phase,
)
from .reconstruction_errors import (
    IdentityCorrectionError as _IdentityCorrectionError,
    ReconstructionCancelled as _ReconstructionCancelled,
    ReconstructionError as _ReconstructionError,
    StaleReconstructionRun as _StaleReconstructionRun,
)
from .reconstruction_identity_phase import (
    track_and_resolve_identity_phase as _track_and_resolve_identity_phase,
)
from .reconstruction_progress import (
    ProgressWriteThrottle as _ProgressWriteThrottle,
    ReconstructionProgress as _ReconstructionProgress,
)
from .reconstruction_run_log import (
    NullRunLog as _NullRunLog,
    open_reconstruction_run_log as _open_reconstruction_run_log,
)
from .reconstruction_publish_phase import (
    publish_reconstruction_phase as _publish_reconstruction_phase,
)
from .reconstruction_status import (
    set_reconstruction_status_in_memory as _set_reconstruction_status_in_memory,
)
from .reconstruction_run_repository import reconstruction_runs as _reconstruction_runs
from .scene_document import (
    reconstruction_input_fingerprint as _reconstruction_input_fingerprint,
)

__all__ = ("reconstruct_scene",)


def _publish_fenced_reconstruction_terminal(
    scene: dict,
    *,
    expected_run_id: str,
    expected_input_fingerprint: str,
    expected_lease_owner_id: str,
) -> None:
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    status = str(reconstruction.get("status") or "")
    if status not in {"ready", "failed", "cancelled"}:
        raise _ReconstructionError(
            f"Cannot publish non-terminal reconstruction state {status or 'unknown'}"
        )
    if not _reconstruction_runs.put_if_reconstruction_run(
        scene,
        expected_run_id,
        expected_input_fingerprint,
        expected_lease_owner_id,
    ):
        raise _StaleReconstructionRun(
            f"Reconstruction run {expected_run_id} lost its terminal publication lease"
        )
    _analysis_runtime.publish_reconstruction_terminal(
        scene,
        status,
        error=reconstruction.get("error"),
    )


def _reconstruct_scene_in_memory(
    scene: dict,
    progress_listener: Callable[[dict], None] | None = None,
    *,
    artifact_store: _ArtifactStore | None = None,
    match_snapshot: Mapping[str, object] | None = None,
    run_log=None,
) -> dict:
    """Compute a reconstruction without reading or mutating scheduler state."""

    if run_log is None:
        run_log = _NullRunLog()

    previous_tracks = deepcopy(scene.get("payload", {}).get("tracks") or [])
    previous_canonical_people = deepcopy(
        scene.get("payload", {}).get("canonicalPeople") or []
    )
    previous_ball = deepcopy(scene.get("payload", {}).get("ball") or {"keyframes": []})
    previous_team_colors = {
        str(team.get("id")): team.get("color")
        for team in scene.get("payload", {}).get("teams") or []
    }
    previous_processing_state = (
        scene.get("payload", {}).get("videoAsset", {}).get("processingState") or "frames-ready"
    )
    _hydrate_scene_reconstruction(scene, store=artifact_store)
    model_name = str(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("model")
        or _get_settings().reconstruction_model
    )
    reconstruction_request = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        or {}
    )
    ball_backend = str(
        reconstruction_request.get("ballBackend")
        or _get_settings().ball_detection_backend
    )
    queued_ball_detection_input = reconstruction_request.get("ballDetectionInput")
    ball_detection_input = (
        deepcopy(queued_ball_detection_input)
        if isinstance(queued_ball_detection_input, dict)
        else _ball_detection_input(ball_backend)
    )
    ball_detection_profile = str(
        reconstruction_request.get("ballDetectionProfile") or "automatic"
    )
    jersey_ocr_profile = str(
        reconstruction_request.get("jerseyOcrProfile") or "automatic"
    )
    progress = _ReconstructionProgress(scene, progress_listener, run_log=run_log)
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    reconstruction.update(
        {
            "status": "processing",
            "processingStatus": "processing",
            "qualityVerdict": "pending",
            "model": model_name,
            "ballBackend": ball_backend,
            "ballDetectionInput": ball_detection_input,
            "ballDetectionProfile": ball_detection_profile,
            "jerseyOcrProfile": jersey_ocr_profile,
            "startedAt": reconstruction.get("startedAt")
            or datetime.now(UTC).isoformat(),
            "error": None,
        }
    )
    video["reconstruction"] = reconstruction
    run_log.event(
        "run-inputs",
        sceneDuration=scene.get("duration"),
        model=model_name,
        ballBackend=ball_backend,
        ballDetectionProfile=ball_detection_profile,
        jerseyOcrProfile=jersey_ocr_profile,
    )
    try:
        frame_result, calibration_result = _detect_and_calibrate_phase(
            scene,
            model_name=model_name,
            reconstruction_request=reconstruction_request,
            ball_backend=ball_backend,
            ball_detection_input=ball_detection_input,
            ball_detection_profile=ball_detection_profile,
            progress=progress,
        )
        run_log.event(
            "phase-finished",
            phase="detection-calibration",
            frameCount=len(frame_result.frames),
            coordinateMode=calibration_result.coordinate_mode,
            calibrationSummary=calibration_result.quality.get("summary"),
            calibrationWarnings=list(calibration_result.warnings),
            personDetectionCache=frame_result.person_detection_cache_diagnostics,
            denseBallFrameMetadata=frame_result.ball_dense_frame_metadata,
            ballDetectionWarnings=list(frame_result.ball_detection_warnings),
        )
        identity_result = _track_and_resolve_identity_phase(
            scene,
            frame_result.frames,
            frame_result.person_frames,
            frame_result.frame_size,
            calibration_result.coordinate_mode,
            calibration_result.resolved_calibrations_by_sample,
            calibration_result.calibration,
            progress,
            match_snapshot,
            frame_result.identity_worker_diagnostics,
            frame_result.identity_warnings,
            jersey_ocr_profile=jersey_ocr_profile,
        )
        run_log.event(
            "phase-finished",
            phase="identity",
            rawTrackCount=identity_result.raw_track_count,
            stableTrackCount=identity_result.stable_track_count,
            canonicalPersonCount=len(identity_result.canonical_people),
            renderableTrackCount=len(identity_result.tracks),
            jerseyOcrStatus=identity_result.jersey_ocr_diagnostics.get("status"),
            jerseySwitchSuspects=identity_result.jersey_ocr_diagnostics.get(
                "numberSwitchSuspects"
            ),
            reidStatus=frame_result.identity_worker_diagnostics.get("status"),
            warnings=list(identity_result.warnings),
        )
        if ball_detection_profile == "skip-manual-authoritative":
            ball_result = _BallTrajectoryPhaseResult(
                keyframes=[],
                diagnostics={
                    "trajectoryMode": "skipped",
                    "skippedByProfile": True,
                    "profile": ball_detection_profile,
                },
            )
        else:
            ball_result = _resolve_ball_phase(
                scene,
                frame_result.ball_frames,
                frame_result.frame_size,
                calibration_result.coordinate_mode,
                len(identity_result.tracks),
                progress,
            )
        run_log.event(
            "phase-finished",
            phase="ball",
            keyframeCount=len(ball_result.keyframes),
            diagnostics=ball_result.diagnostics,
        )

        published = _publish_reconstruction_phase(
            scene,
            frame_result=frame_result,
            calibration_result=calibration_result,
            identity_result=identity_result,
            ball_result=ball_result,
            ball_backend=ball_backend,
            ball_detection_input=ball_detection_input,
            ball_detection_profile=ball_detection_profile,
            progress=progress,
            artifact_store=artifact_store,
        )
        published_reconstruction = (
            published["payload"]["videoAsset"].get("reconstruction") or {}
        )
        run_log.event(
            "phase-finished",
            phase="publish",
            qualityVerdict=published_reconstruction.get("qualityVerdict"),
            trackCount=published_reconstruction.get("trackCount"),
            ballSamples=published_reconstruction.get("ballSamples"),
            warnings=published_reconstruction.get("warnings"),
        )
        return published
    except _analysis_runtime.AnalysisCancellationRequested as exc:
        scene["payload"]["tracks"] = previous_tracks
        scene["payload"]["canonicalPeople"] = previous_canonical_people
        scene["payload"]["ball"] = previous_ball
        for team in scene.get("payload", {}).get("teams") or []:
            if str(team.get("id")) in previous_team_colors:
                team["color"] = previous_team_colors[str(team.get("id"))]
        scene["payload"]["videoAsset"]["processingState"] = previous_processing_state
        # Cancellation already committed Scene, job, lease and AnalysisRun in
        # one transaction. The fenced worker only abandons its in-memory result.
        raise _ReconstructionCancelled(str(exc)) from exc
    except _StaleReconstructionRun:
        # The current database document belongs to a newer run or newer manual
        # inputs. Never restore snapshots or mark that newer state as failed.
        raise
    except Exception as exc:
        scene["payload"]["tracks"] = previous_tracks
        scene["payload"]["canonicalPeople"] = previous_canonical_people
        scene["payload"]["ball"] = previous_ball
        for team in scene.get("payload", {}).get("teams") or []:
            if str(team.get("id")) in previous_team_colors:
                team["color"] = previous_team_colors[str(team.get("id"))]
        scene["payload"]["videoAsset"]["processingState"] = previous_processing_state
        identity_correction_diagnostics = (
            [deepcopy(exc.diagnostic)]
            if isinstance(exc, _IdentityCorrectionError)
            else []
        )
        failure_values: dict = {}
        failed_progress = progress.failed(str(exc))
        if identity_correction_diagnostics:
            reconstruction = scene["payload"]["videoAsset"].get("reconstruction") or {}
            diagnostics = {
                **(reconstruction.get("diagnostics") or {}),
                "identityCorrections": identity_correction_diagnostics,
            }
            failed_progress["identityCorrections"] = identity_correction_diagnostics
            failure_values = {
                "identityCorrectionDiagnostics": identity_correction_diagnostics,
                "diagnostics": diagnostics,
            }
        _set_reconstruction_status_in_memory(
            scene,
            "failed",
            processingStatus="failed",
            qualityVerdict="reject",
            error=str(exc),
            completedAt=datetime.now(UTC).isoformat(),
            progress=failed_progress,
            **failure_values,
        )
        if isinstance(exc, _ReconstructionError):
            raise
        raise _ReconstructionError(str(exc)) from exc


def reconstruct_scene(
    scene: dict,
    *,
    expected_run_id: str,
    expected_input_fingerprint: str,
    expected_lease_owner_id: str,
    progress_listener: Callable[[dict], None] | None = None,
    artifact_store: _ArtifactStore | None = None,
    match_snapshot: Mapping[str, object] | None = None,
) -> dict:
    """Execute and publish only an explicitly claimed reconstruction run."""

    run_id = str(expected_run_id or "")
    input_fingerprint = str(expected_input_fingerprint or "")
    owner_id = str(expected_lease_owner_id or "")
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    if not run_id or not input_fingerprint or not owner_id:
        raise _ReconstructionError(
            "Reconstruction execution requires run, input and lease-owner fences"
        )
    if (
        reconstruction.get("status") != "processing"
        or str(reconstruction.get("runId") or "") != run_id
        or str(reconstruction.get("inputFingerprint") or "")
        != input_fingerprint
        or _reconstruction_input_fingerprint(scene) != input_fingerprint
    ):
        raise _StaleReconstructionRun(
            f"Reconstruction run {run_id} does not match the claimed scene input"
        )

    throttle = _ProgressWriteThrottle(
        _get_settings().reconstruction_progress_write_interval_seconds
    )

    def publish_progress(payload: dict) -> None:
        if throttle.should_write(payload):
            if not _analysis_runtime.publish_reconstruction_progress(
                scene,
                payload,
                expected_run_id=run_id,
                expected_input_fingerprint=input_fingerprint,
                expected_lease_owner_id=owner_id,
                run_repository=_reconstruction_runs,
            ):
                raise _StaleReconstructionRun(
                    f"Reconstruction run {run_id} lost its progress lease"
                )
        if progress_listener is not None:
            progress_listener(deepcopy(payload))

    settings = _get_settings()
    run_log = _open_reconstruction_run_log(
        scene_id=str(scene.get("id") or ""),
        run_id=run_id,
        directory=settings.analysis_run_log_directory,
        enabled=bool(settings.analysis_run_log_enabled),
    )
    try:
        result = _reconstruct_scene_in_memory(
            scene,
            progress_listener=publish_progress,
            artifact_store=artifact_store,
            match_snapshot=match_snapshot,
            run_log=run_log,
        )
    except _ReconstructionCancelled as exc:
        run_log.close("cancelled", detail=str(exc))
        raise
    except _StaleReconstructionRun as exc:
        run_log.close("stale", detail=str(exc))
        raise
    except Exception as exc:
        run_log.close("failed", detail=str(exc))
        _publish_fenced_reconstruction_terminal(
            scene,
            expected_run_id=run_id,
            expected_input_fingerprint=input_fingerprint,
            expected_lease_owner_id=owner_id,
        )
        raise
    _publish_fenced_reconstruction_terminal(
        result,
        expected_run_id=run_id,
        expected_input_fingerprint=input_fingerprint,
        expected_lease_owner_id=owner_id,
    )
    run_log.close("ready")
    return result
