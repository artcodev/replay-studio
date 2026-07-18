from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .artifact_store import ReconstructionArtifactError
from .reconstruction_series import (
    ReconstructionSeriesWindowError,
    reconstruction_series_window,
)
from . import project_resource_access
from .scene_contracts import SceneDocument
from .scene_document import reconstruction_input_fingerprint
from .scene_repository import scenes


router = APIRouter(prefix="/api/projects/{project_id}/scenes", tags=["scenes"])


@router.get("/{scene_id}", response_model=SceneDocument)
def get_scene(project_id: str, scene_id: str):
    return project_resource_access.project_scene_or_404(project_id, scene_id)


@router.get("/{scene_id}/reconstruction-series")
def get_scene_reconstruction_series(
    project_id: str,
    scene_id: str,
    start: float = Query(ge=0.0),
    end: float = Query(ge=0.0),
    frame_start: int | None = Query(default=None, ge=0),
    frame_end: int | None = Query(default=None, ge=0),
    track_id: str | None = Query(default=None, min_length=1, max_length=240),
    canonical_person_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=240,
    ),
):
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return reconstruction_series_window(
            scene,
            start=start,
            end=end,
            frame_start=frame_start,
            frame_end=frame_end,
            track_id=track_id,
            canonical_person_id=canonical_person_id,
        )
    except ReconstructionSeriesWindowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReconstructionArtifactError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Reconstruction artifact is unavailable or corrupt: {exc}",
        ) from exc


@router.put("/{scene_id}", response_model=SceneDocument)
def update_scene(project_id: str, scene_id: str, scene: SceneDocument):
    if scene_id != scene.id:
        raise HTTPException(status_code=400, detail="Scene id does not match URL")
    current = project_resource_access.project_scene_or_404(project_id, scene_id)
    incoming = scene.model_dump()
    current_reconstruction = (
        current.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    incoming_reconstruction = (
        incoming.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if current_reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before saving the scene",
        )
    runtime_fields = ("runId", "runRevision", "inputFingerprint", "status")
    if any(
        current_reconstruction.get(field) != incoming_reconstruction.get(field)
        for field in runtime_fields
    ):
        raise HTTPException(
            status_code=409,
            detail="The scene reconstruction changed; reload before saving",
        )
    if reconstruction_input_fingerprint(current) != reconstruction_input_fingerprint(
        incoming
    ):
        raise HTTPException(
            status_code=409,
            detail="Reconstruction inputs must be edited through their dedicated endpoints",
        )
    return scenes.put(incoming)
