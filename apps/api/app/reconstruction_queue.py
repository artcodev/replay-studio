from __future__ import annotations

"""Publish immutable artifacts and one lease-fenced reconstruction job."""

from copy import deepcopy
from uuid import uuid4

from .config import get_settings
from .project_match import match_snapshot_reference
from .project_match_persistence_contract import MatchSnapshotDocument
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_artifact_publication import (
    publish_dense_reconstruction_artifacts,
)
from .reconstruction_artifact_manifest import artifact_references
from .ball_detection_configuration import ball_detection_input
from .reconstruction_coordinate_policy import (
    METRIC_REQUIRED,
    resolve_full_run_coordinate_authorization,
)
from .direct_calibration_sampling import (
    resolve_direct_calibration_max_gap_seconds,
)
from .reconstruction_calibration_fingerprint import calibration_input_fingerprint
from .reconstruction_calibration_snapshot import (
    calibration_artifact_input as resolve_calibration_artifact_input,
)
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_inputs import (
    frame_paths,
    require_model_weights_available,
    resolve_sampling_frame_rate,
)
from .project_resource_repository import ProjectResourceConflict
from .reconstruction_queue_draft import (
    ReconstructionQueueInputs,
    prepare_reconstruction_queue_draft,
    validate_reconstruction_queue_scene,
)
from .reconstruction_run_repository import reconstruction_runs
from .scene_document import SceneRevisionConflict, reconstruction_input_fingerprint


def _manual_ball_is_authoritative(scene: dict) -> bool:
    ball = scene.get("payload", {}).get("ball")
    return isinstance(ball, dict) and ball.get("mode") == "manual"


def queue_reconstruction(
    scene: dict,
    model_name: str | None = None,
    *,
    ball_backend: str | None = None,
    ball_detection_profile: str | None = None,
    jersey_ocr_profile: str | None = None,
    contact_point_profile: str | None = None,
    mode: str | None = None,
    sampling_frame_rate: float | None = None,
    direct_calibration_max_gap_seconds: float | None = None,
    calibration_trigger: str | None = None,
    match_snapshot: MatchSnapshotDocument | None,
    expected_scene_fingerprint: str | None = None,
) -> dict:
    validate_reconstruction_queue_scene(scene)
    video = scene["payload"]["videoAsset"]
    previous = video.get("reconstruction") or {}
    # Mode is an explicit per-click directive, never inherited: the two buttons
    # ("Calibrate", "Reconstruct") each send their own mode.
    selected_mode = str(mode or "full")
    if selected_mode not in {"calibrate", "full"}:
        raise ReconstructionError("Unknown reconstruction mode")
    selected_calibration_trigger = (
        str(calibration_trigger or "full-request")
        if selected_mode == "calibrate"
        else None
    )
    previous_sampling_frame_rate = previous.get("samplingFrameRate")
    if (
        selected_mode == "full"
        and sampling_frame_rate is not None
        and previous_sampling_frame_rate is not None
        and abs(float(sampling_frame_rate) - float(previous_sampling_frame_rate))
        > 1e-3
    ):
        raise ReconstructionError(
            "Reconstruction FPS cannot differ from the calibrated FPS; "
            "run calibration again with the new cadence"
        )
    selected_sampling_frame_rate = resolve_sampling_frame_rate(
        scene,
        sampling_frame_rate
        if sampling_frame_rate is not None
        else (
            float(previous_sampling_frame_rate)
            if previous_sampling_frame_rate is not None
            else None
        ),
    )
    previous_direct_gap = previous.get("directCalibrationMaxGapSeconds")
    if (
        selected_mode == "full"
        and direct_calibration_max_gap_seconds is not None
        and previous_direct_gap is not None
        and abs(
            float(direct_calibration_max_gap_seconds)
            - float(previous_direct_gap)
        )
        > 1e-6
    ):
        raise ReconstructionError(
            "Reconstruction direct-calibration sampling cannot differ from "
            "the calibrated policy; run calibration again"
        )
    selected_direct_gap = resolve_direct_calibration_max_gap_seconds(
        direct_calibration_max_gap_seconds
        if direct_calibration_max_gap_seconds is not None
        else (
            previous_direct_gap
            if selected_mode == "full"
            or selected_calibration_trigger == "manual-draft-finalize"
            else None
        )
    )
    selected_model = (
        model_name
        or previous.get("model")
        or get_settings().reconstruction_model
    )
    selected_ball_backend = str(
        ball_backend
        or previous.get("ballBackend")
        or get_settings().ball_detection_backend
    )
    # Only full reconstruction owns person/ball model inputs. Calibration is
    # an independent process and must not be blocked by reconstruction weights.
    if selected_mode == "full":
        require_model_weights_available(str(selected_model))
    if ball_detection_profile == "skip-manual-authoritative" and (
        not _manual_ball_is_authoritative(scene)
    ):
        raise ReconstructionError(
            "Ball detection can be skipped only while the manual ball "
            "trajectory is the authoritative channel"
        )
    selected_profile = str(
        ball_detection_profile
        or previous.get("ballDetectionProfile")
        or "automatic"
    )
    # An inherited skip profile silently outliving its precondition would hide
    # a needed dense-ball pass; degrade it to an explicit automatic run.
    if selected_profile == "skip-manual-authoritative" and (
        not _manual_ball_is_authoritative(scene)
    ):
        selected_profile = "automatic"
    selected_jersey_profile = str(
        jersey_ocr_profile
        or previous.get("jerseyOcrProfile")
        or "automatic"
    )
    selected_contact_profile = str(
        contact_point_profile
        or previous.get("contactPointProfile")
        or "bbox-bottom"
    )
    ball_input = (
        ball_detection_input(selected_ball_backend)
        if selected_mode == "full"
        else {
            "schemaVersion": 1,
            "backend": selected_ball_backend,
            "usage": "not-used-by-calibration",
        }
    )
    expected_input_fingerprint = (
        expected_scene_fingerprint or reconstruction_input_fingerprint(scene)
    )
    selected_match_snapshot_ref = match_snapshot_reference(match_snapshot)
    selected_calibration_input_fingerprint = calibration_input_fingerprint(
        scene,
        sampling_frame_rate=selected_sampling_frame_rate,
        direct_calibration_max_gap_seconds=selected_direct_gap,
    )
    tracking_coordinate_policy = METRIC_REQUIRED
    calibration_fallback_consent = None
    if selected_mode == "full":
        (
            tracking_coordinate_policy,
            calibration_fallback_consent,
        ) = resolve_full_run_coordinate_authorization(
            previous,
            calibration_input_fingerprint=selected_calibration_input_fingerprint,
        )
    input_frames = frame_paths(
        scene,
        sampling_frame_rate=selected_sampling_frame_rate,
    )

    # Artifact hydration/publication is intentionally part of this command,
    # not the pure queue draft. Content-addressed files may be published before
    # the database CAS; a losing CAS leaves only an unreferenced immutable blob.
    artifact_ready_scene = deepcopy(scene)
    hydrate_scene_reconstruction(artifact_ready_scene)
    selected_calibration_artifact_input = (
        resolve_calibration_artifact_input(
            artifact_ready_scene["payload"]["videoAsset"]["reconstruction"]
        )
        if selected_mode == "full"
        else None
    )
    publish_dense_reconstruction_artifacts(artifact_ready_scene)
    if selected_calibration_artifact_input is not None:
        # Pin the exact compact artifact reference that the queued document
        # carries. Its content was built from the materialized, fingerprint-
        # validated calibration above.
        compact_reconstruction = artifact_ready_scene["payload"]["videoAsset"][
            "reconstruction"
        ]
        current_reference = artifact_references(compact_reconstruction).get(
            "calibrationFrames"
        )
        if current_reference is None:
            raise ReconstructionError(
                "Completed calibration artifact disappeared during queue publication"
            )
        selected_calibration_artifact_input["artifact"] = deepcopy(
            current_reference
        )
    queued_scene = prepare_reconstruction_queue_draft(
        artifact_ready_scene,
        ReconstructionQueueInputs(
            model=str(selected_model),
            ball_backend=selected_ball_backend,
            ball_detection_input=ball_input,
            ball_detection_profile=selected_profile,
            jersey_ocr_profile=selected_jersey_profile,
            contact_point_profile=selected_contact_profile,
            sampling_frame_rate=selected_sampling_frame_rate,
            direct_calibration_max_gap_seconds=selected_direct_gap,
            mode=selected_mode,
            tracking_coordinate_policy=tracking_coordinate_policy,
            calibration_fallback_consent=calibration_fallback_consent,
            calibration_input_fingerprint=selected_calibration_input_fingerprint,
            calibration_artifact_input=selected_calibration_artifact_input,
            calibration_trigger=selected_calibration_trigger,
            frame_count=len(input_frames),
            run_id=uuid4().hex,
            match_snapshot_ref=selected_match_snapshot_ref,
        ),
    )
    try:
        return reconstruction_runs.enqueue_reconstruction(
            queued_scene,
            expected_input_fingerprint=expected_input_fingerprint,
        )
    except (ProjectResourceConflict, SceneRevisionConflict) as exc:
        # Keep the concrete refused fence: "active reconstruction lease"
        # asks the user to wait, a revision conflict asks them to refresh.
        raise StaleReconstructionRun(str(exc)) from exc
