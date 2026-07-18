"""Guarded mutation routes for human identity-review decisions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .identity_decisions import (
    IdentityDecisionError,
    clear_roster_candidate_rejection,
    reject_roster_candidate,
)
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_queue import queue_reconstruction
from .project_match_repository import project_matches
from .project_resource_repository import ProjectResourceConflict, project_resources
from .scene_document import reconstruction_input_fingerprint
from .scene_repository import scenes


router = APIRouter()


class RosterCandidateDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    external_player_id: str = Field(min_length=1, max_length=120)


def _mutate_and_queue(
    project_id: str,
    scene_id: str,
    mutation,
) -> dict:
    try:
        owner = project_resources.scene_owner(scene_id)
    except ProjectResourceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    scene = scenes.get(scene_id) if owner == project_id else None
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found in project")
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before reviewing identity",
        )
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    match_snapshot = project_matches.current_snapshot(project_id)
    try:
        mutation(
            scene,
            match_snapshot.payload if match_snapshot is not None else None,
        )
        if video.get("selectedSegmentId"):
            saved = queue_reconstruction(
                scene,
                match_snapshot=match_snapshot,
                expected_scene_fingerprint=expected_fingerprint,
            )
            return saved
        return scenes.put(scene)
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
    "/api/projects/{project_id}/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-rejections",
    status_code=202,
)
def save_roster_candidate_rejection(
    project_id: str,
    scene_id: str,
    canonical_person_id: str,
    request: RosterCandidateDecisionRequest,
) -> dict:
    return _mutate_and_queue(
        project_id,
        scene_id,
        lambda scene, match_snapshot: reject_roster_candidate(
            scene,
            canonical_person_id,
            request.external_player_id,
            match_snapshot=match_snapshot,
        ),
    )


@router.delete(
    "/api/projects/{project_id}/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-rejections/{external_player_id}",
    status_code=202,
)
def delete_roster_candidate_rejection(
    project_id: str,
    scene_id: str,
    canonical_person_id: str,
    external_player_id: str,
) -> dict:
    return _mutate_and_queue(
        project_id,
        scene_id,
        lambda scene, _match_snapshot: clear_roster_candidate_rejection(
            scene,
            canonical_person_id,
            external_player_id,
        ),
    )


__all__ = ["router"]
