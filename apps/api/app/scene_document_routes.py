from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .artifact_store import ReconstructionArtifactError
from .reconstruction_series import (
    ReconstructionSeriesWindowError,
    reconstruction_series_window,
)
from . import project_resource_access
from .reconstruction_errors import ReconstructionError
from .scene_contracts import (
    SceneDocument,
    SceneEventBindingsRequest,
    SceneTitleRequest,
    SegmentLayoutRequest,
    TrackMetadataRequest,
    TrackTrajectoryRequest,
)
from .scene_metadata_command import (
    set_scene_event_bindings,
    set_scene_title,
    set_track_metadata,
)
from .segment_layout_command import set_scene_segment_layout
from .track_trajectory_corrections import set_track_trajectory


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


# The generic whole-document PUT was retired: every editor write is a
# dedicated command below, so no client ever round-trips a scene document
# and the revision fence no longer arbitrates between the UI and the
# reconstruction runner.


@router.put("/{scene_id}/title", response_model=SceneDocument)
def update_scene_title(project_id: str, scene_id: str, request: SceneTitleRequest):
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return set_scene_title(scene, request.title)
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{scene_id}/event-bindings", response_model=SceneDocument)
def update_scene_event_bindings(
    project_id: str,
    scene_id: str,
    request: SceneEventBindingsRequest,
):
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return set_scene_event_bindings(
            scene,
            [binding.model_dump() for binding in request.bindings],
        )
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{scene_id}/tracks/{track_id}/metadata", response_model=SceneDocument)
def update_scene_track_metadata(
    project_id: str,
    scene_id: str,
    track_id: str,
    request: TrackMetadataRequest,
):
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return set_track_metadata(
            scene,
            track_id,
            label=request.label,
            number=request.number,
        )
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{scene_id}/segment-layout", response_model=SceneDocument)
def update_scene_segment_layout(
    project_id: str,
    scene_id: str,
    request: SegmentLayoutRequest,
):
    """Publish event grouping onto the current scene.

    The client sends only the layout, never a whole scene document, so an
    editor holding an older revision still saves: grouping is idempotent and
    outside the reconstruction fingerprint.
    """
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return set_scene_segment_layout(
            scene,
            [entry.model_dump(exclude_none=True) for entry in request.segments],
            request.status,
        )
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{scene_id}/tracks/{track_id}/trajectory", response_model=SceneDocument)
def update_scene_track_trajectory(
    project_id: str,
    scene_id: str,
    track_id: str,
    request: TrackTrajectoryRequest,
):
    """Publish a durable manual trajectory correction for one player.

    The correction is anchored to the track's canonical identity so it
    survives a rebuild; a track without one is refused rather than accepting
    an edit the next run would drop (TD-TRACK-01).
    """
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    try:
        return set_track_trajectory(
            scene,
            track_id,
            [keyframe.model_dump() for keyframe in request.keyframes],
        )
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
