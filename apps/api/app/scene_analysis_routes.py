from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from .model_comparison_contract import BASELINE_MODEL, CANDIDATE_MODEL
from .model_comparison_pipeline_service import ModelComparisonPipelineService
from .pipeline_domain import PipelineJobConflict
from .player_action_commands import (
    delete_player_action,
    upsert_player_action,
)
from .player_action_planning import PlayerActionError
from .project_match_repository import project_matches
from .reconstruction_ball_trajectory_command import set_scene_ball_trajectory
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_queue import queue_reconstruction
from . import project_resource_access
from .ball_contracts import BallTrajectoryRequest
from .player_action_contracts import PlayerActionUpsertRequest
from .reconstruction_contracts import ReconstructionRequest
from .scene_contracts import SceneDocument


router = APIRouter(prefix="/api/projects/{project_id}/scenes", tags=["analysis"])
model_comparison_pipeline = ModelComparisonPipelineService()


def _video(scene: dict) -> dict:
    return scene.get("payload", {}).get("videoAsset") or {}


def _ensure_idle(scene: dict, operation: str) -> None:
    if (_video(scene).get("reconstruction") or {}).get("status") in {
        "queued",
        "processing",
    }:
        raise HTTPException(
            status_code=409,
            detail=f"Wait for reconstruction to finish before {operation}",
        )


@router.post("/{scene_id}/reconstruct", response_model=SceneDocument, status_code=202)
def reconstruct_video_scene(
    project_id: str,
    scene_id: str,
    request: ReconstructionRequest | None = None,
):
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _video(scene)
    if video.get("multiPass"):
        raise HTTPException(
            status_code=409,
            detail=(
                "This is a multi-pass composite; rerun multi-angle analysis "
                "instead of single-pass reconstruction"
            ),
        )
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail="Choose a continuous shot scene before reconstruction",
        )
    if (video.get("reconstruction") or {}).get("status") in {
        "queued",
        "processing",
    }:
        raise HTTPException(status_code=409, detail="Reconstruction is already running")
    try:
        return queue_reconstruction(
            scene,
            request.model if request else None,
            ball_backend=request.ball_backend if request else None,
            match_snapshot=project_matches.current_snapshot(project_id),
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while reconstruction was being queued; "
                "retry with the latest scene."
            ),
        ) from exc


@router.put("/{scene_id}/ball-trajectory", response_model=SceneDocument)
def update_scene_ball_trajectory(
    project_id: str,
    scene_id: str,
    request: BallTrajectoryRequest,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    _ensure_idle(scene, "editing the ball trajectory")
    keyframes = (
        None
        if request.keyframes is None
        else [item.model_dump(exclude_none=True) for item in request.keyframes]
    )
    try:
        return set_scene_ball_trajectory(scene, request.mode, keyframes)
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{scene_id}/player-actions", response_model=SceneDocument)
def save_scene_player_action(
    project_id: str,
    scene_id: str,
    request: PlayerActionUpsertRequest,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    _ensure_idle(scene, "editing player actions")
    try:
        upsert_player_action(scene, request.model_dump())
    except PlayerActionError as exc:
        status_code = (
            404 if str(exc) == "The canonical person no longer exists" else 422
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return scene


@router.delete(
    "/{scene_id}/player-actions/{action_id}",
    response_model=SceneDocument,
)
def remove_scene_player_action(
    project_id: str,
    scene_id: str,
    action_id: str,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    _ensure_idle(scene, "editing player actions")
    try:
        delete_player_action(scene, action_id)
    except PlayerActionError as exc:
        status_code = 404 if str(exc) == "Player action not found" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return scene


@router.post("/{scene_id}/compare-models", status_code=202)
def compare_video_scene_models(project_id: str, scene_id: str) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _video(scene)
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail="Choose a continuous shot scene before comparing models",
        )
    _ensure_idle(scene, "comparing models")
    try:
        job = model_comparison_pipeline.enqueue(
            job_id=f"model-comparison-{uuid4().hex}",
            project_id=project_id,
            scene_id=scene_id,
            baseline_model=BASELINE_MODEL,
            candidate_model=CANDIDATE_MODEL,
        )
    except PipelineJobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "runId": job.id,
        "sceneId": scene_id,
        "kind": job.kind,
        "status": job.status,
    }
