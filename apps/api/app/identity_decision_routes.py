"""Guarded mutation routes for human identity-review decisions."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .identity_decisions import (
    IdentityDecisionError,
    clear_roster_candidate_rejection,
    reject_roster_candidate,
)
from .reconstruction import (
    ReconstructionError,
    StaleReconstructionRun,
    queue_reconstruction,
    reconstruct_scene_by_id,
)
from .store import reconstruction_input_fingerprint, scene_store


router = APIRouter()


class RosterCandidateDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    external_player_id: str = Field(min_length=1, max_length=120)


def _mutate_and_queue(
    scene_id: str,
    background_tasks: BackgroundTasks,
    mutation,
) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before reviewing identity",
        )
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        mutation(scene)
        if video.get("selectedSegmentId"):
            saved = queue_reconstruction(
                scene,
                expected_scene_fingerprint=expected_fingerprint,
            )
            queued = saved["payload"]["videoAsset"]["reconstruction"]
            background_tasks.add_task(
                reconstruct_scene_by_id,
                scene_id,
                queued["runId"],
                queued["inputFingerprint"],
            )
            return saved
        return scene_store.put(scene)
    except IdentityDecisionError as exc:
        status = 404 if "not found" in str(exc).lower() or "no longer exists" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while the identity decision was being saved; retry.",
        ) from exc
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/api/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-rejections",
    status_code=202,
)
def save_roster_candidate_rejection(
    scene_id: str,
    canonical_person_id: str,
    request: RosterCandidateDecisionRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    return _mutate_and_queue(
        scene_id,
        background_tasks,
        lambda scene: reject_roster_candidate(
            scene,
            canonical_person_id,
            request.external_player_id,
        ),
    )


@router.delete(
    "/api/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-rejections/{external_player_id}",
    status_code=202,
)
def delete_roster_candidate_rejection(
    scene_id: str,
    canonical_person_id: str,
    external_player_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    return _mutate_and_queue(
        scene_id,
        background_tasks,
        lambda scene: clear_roster_candidate_rejection(
            scene,
            canonical_person_id,
            external_player_id,
        ),
    )


__all__ = ["router"]
