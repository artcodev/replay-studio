from __future__ import annotations

"""Pure SceneDocument transitions for reconstruction control state."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .reconstruction_run_contract import (
    ReconstructionRunFence,
    reconstruction_state,
    scene_matches_fence,
)
from .scene_document import next_scene_payload, scene_revision


@dataclass(frozen=True, slots=True)
class TransitionedScene:
    payload: dict[str, Any]
    reconstruction: dict[str, Any]
    revision: int


def transition_scene_to_processing(
    scene: dict[str, Any],
    fence: ReconstructionRunFence,
    *,
    current_time: float,
) -> TransitionedScene | None:
    if not scene_matches_fence(
        scene,
        fence,
        statuses={"queued", "processing"},
    ):
        return None

    claimed = deepcopy(scene)
    video = claimed["payload"]["videoAsset"]
    reconstruction = video["reconstruction"]
    reconstruction.update(
        {
            "status": "processing",
            "processingStatus": "processing",
            "startedAt": (
                reconstruction.get("startedAt")
                or datetime.fromtimestamp(current_time, UTC).isoformat()
            ),
            "error": None,
        }
    )
    video["processingState"] = "reconstructing"
    revision = scene_revision(scene) + 1
    payload = next_scene_payload(claimed, revision)
    return TransitionedScene(
        payload=payload,
        reconstruction=reconstruction_state(payload),
        revision=revision,
    )


def transition_matching_scene_to_failed(
    scene: dict[str, Any],
    fence: ReconstructionRunFence,
    *,
    current_time: float,
    error: str,
) -> TransitionedScene | None:
    """Release the dense read model for an invalid current scheduler job.

    A Scene input edit can invalidate the computed fingerprint while leaving
    the same queued/processing run tokens in the document.  The scheduler job
    must become terminal, but so must that exact dense run or the editor stays
    locked and a retry returns 409.  A Scene that already belongs to another
    run is deliberately left untouched.
    """

    reconstruction = reconstruction_state(scene)
    if (
        str(scene.get("id") or "") != fence.scene_id
        or reconstruction.get("status") not in {"queued", "processing"}
        or str(reconstruction.get("runId") or "") != fence.run_id
    ):
        return None

    failed = deepcopy(scene)
    video = failed["payload"]["videoAsset"]
    failed_reconstruction = video["reconstruction"]
    completed_at = datetime.fromtimestamp(current_time, UTC).isoformat()
    current_progress = failed_reconstruction.get("progress")
    progress = (
        deepcopy(current_progress)
        if isinstance(current_progress, dict)
        else {}
    )
    progress.update(
        {
            "phase": "failed",
            "label": "Analysis failed",
            "detail": error,
            "etaSeconds": 0.0,
            "updatedAt": completed_at,
        }
    )
    failed_reconstruction.update(
        {
            "status": "failed",
            "processingStatus": "failed",
            "qualityVerdict": "reject",
            "error": error,
            "completedAt": completed_at,
            "progress": progress,
        }
    )
    video["processingState"] = "frames-ready"
    revision = scene_revision(scene) + 1
    payload = next_scene_payload(failed, revision)
    return TransitionedScene(
        payload=payload,
        reconstruction=reconstruction_state(payload),
        revision=revision,
    )
