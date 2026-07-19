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
from .ball_detection_configuration import ball_detection_input
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_inputs import frame_paths
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
    match_snapshot: MatchSnapshotDocument | None,
    expected_scene_fingerprint: str | None = None,
) -> dict:
    validate_reconstruction_queue_scene(scene)
    video = scene["payload"]["videoAsset"]
    previous = video.get("reconstruction") or {}
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
    ball_input = ball_detection_input(selected_ball_backend)
    base_input_fingerprint = (
        expected_scene_fingerprint or reconstruction_input_fingerprint(scene)
    )
    input_frames = frame_paths(scene)

    # Artifact hydration/publication is intentionally part of this command,
    # not the pure queue draft. Content-addressed files may be published before
    # the database CAS; a losing CAS leaves only an unreferenced immutable blob.
    artifact_ready_scene = deepcopy(scene)
    hydrate_scene_reconstruction(artifact_ready_scene)
    publish_dense_reconstruction_artifacts(artifact_ready_scene)
    queued_scene = prepare_reconstruction_queue_draft(
        artifact_ready_scene,
        ReconstructionQueueInputs(
            model=str(selected_model),
            ball_backend=selected_ball_backend,
            ball_detection_input=ball_input,
            ball_detection_profile=selected_profile,
            jersey_ocr_profile=selected_jersey_profile,
            frame_count=len(input_frames),
            run_id=uuid4().hex,
            match_snapshot_ref=match_snapshot_reference(match_snapshot),
        ),
    )
    try:
        return reconstruction_runs.enqueue_reconstruction(
            queued_scene,
            expected_input_fingerprint=base_input_fingerprint,
        )
    except (ProjectResourceConflict, SceneRevisionConflict) as exc:
        raise StaleReconstructionRun(
            "The scene revision or reconstruction inputs changed while queuing"
        ) from exc
