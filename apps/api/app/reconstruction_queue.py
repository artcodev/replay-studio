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
from .reconstruction_errors import StaleReconstructionRun
from .reconstruction_inputs import frame_paths
from .project_resource_repository import ProjectResourceConflict
from .reconstruction_queue_draft import (
    ReconstructionQueueInputs,
    prepare_reconstruction_queue_draft,
    validate_reconstruction_queue_scene,
)
from .reconstruction_run_repository import reconstruction_runs
from .scene_document import SceneRevisionConflict, reconstruction_input_fingerprint


def queue_reconstruction(
    scene: dict,
    model_name: str | None = None,
    *,
    ball_backend: str | None = None,
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
