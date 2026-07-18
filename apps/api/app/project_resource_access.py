from __future__ import annotations

from fastapi import HTTPException

from .project_match import match_snapshot_reference
from .project_match_repository import project_matches
from .project_resource_repository import ProjectResourceConflict, project_resources
from .scene_document import annotate_reconstruction_input_state
from .scene_repository import scenes
from .video_store import video_store


def project_scene_or_404(project_id: str, scene_id: str) -> dict:
    try:
        owner = project_resources.scene_owner(scene_id)
    except ProjectResourceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if owner != project_id:
        raise HTTPException(status_code=404, detail="Scene not found in project")
    scene = scenes.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found in project")
    return annotate_reconstruction_input_state(
        scene,
        match_snapshot_reference(
            project_matches.current_summary(project_id)
        ),
    )


def project_video_or_404(project_id: str, asset_id: str) -> dict:
    try:
        owner = project_resources.video_asset_owner(asset_id)
    except ProjectResourceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if owner != project_id:
        raise HTTPException(status_code=404, detail="Video asset not found in project")
    asset = video_store.get(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Video asset not found in project")
    return asset
