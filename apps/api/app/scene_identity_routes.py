from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .project_match_repository import project_matches
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_frame_analysis import analyze_scene_frame
from .reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete,
    draft_frame_person_annotation_upsert,
)
from .reconstruction_identity_roster_draft import (
    draft_canonical_roster_binding,
    draft_clear_canonical_roster_binding,
)
from .reconstruction_queue import queue_reconstruction
from . import project_resource_access
from .frame_annotation_contracts import (
    CanonicalRosterBindingRequest,
    FrameAnalysisRequest,
    FramePersonAnnotationRequest,
)
from .scene_contracts import SceneDocument
from .scene_document import reconstruction_input_fingerprint


router = APIRouter(prefix="/api/projects/{project_id}/scenes", tags=["identity"])


def _video(scene: dict) -> dict:
    return scene.get("payload", {}).get("videoAsset") or {}


def _require_segment(scene: dict, operation: str) -> dict:
    video = _video(scene)
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail=f"Choose a continuous shot scene before {operation}",
        )
    return video


def _require_idle(video: dict, operation: str) -> None:
    if (video.get("reconstruction") or {}).get("status") in {
        "queued",
        "processing",
    }:
        raise HTTPException(
            status_code=409,
            detail=f"Wait for reconstruction to finish before {operation}",
        )


def _require_scene_time(scene: dict, scene_time: float) -> None:
    if scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")


def _analysis_with_queued_reconstruction(analysis: dict, queued: dict) -> dict:
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    analysis["reconstruction"] = {
        key: reconstruction.get(key)
        for key in ("status", "model", "runId", "runRevision", "inputFingerprint")
    }
    return analysis


@router.post("/{scene_id}/analyze-frame")
def analyze_video_scene_frame(
    project_id: str,
    scene_id: str,
    request: FrameAnalysisRequest,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    _require_segment(scene, "analyzing a frame")
    _require_scene_time(scene, request.scene_time)
    try:
        return analyze_scene_frame(scene, request.scene_time)
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{scene_id}/frame-annotations")
def save_video_scene_frame_annotation(
    project_id: str,
    scene_id: str,
    request: FramePersonAnnotationRequest,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _require_segment(scene, "labeling a frame")
    _require_idle(video, "labeling a frame")
    _require_scene_time(scene, request.scene_time)
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        annotation = draft_frame_person_annotation_upsert(
            scene, request.model_dump()
        )
        analysis = analyze_scene_frame(scene, float(annotation["sceneTime"]))
        queued = queue_reconstruction(
            scene,
            match_snapshot=project_matches.current_snapshot(project_id),
            expected_scene_fingerprint=expected_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while the correction was being saved; "
                "retry on the latest frame."
            ),
        ) from exc
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _analysis_with_queued_reconstruction(analysis, queued)


@router.delete("/{scene_id}/frame-annotations/{annotation_id}")
def remove_video_scene_frame_annotation(
    project_id: str,
    scene_id: str,
    annotation_id: str,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _video(scene)
    _require_idle(video, "editing labels")
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        annotation = draft_frame_person_annotation_delete(scene, annotation_id)
        analysis = analyze_scene_frame(scene, float(annotation["sceneTime"]))
        queued = queue_reconstruction(
            scene,
            match_snapshot=project_matches.current_snapshot(project_id),
            expected_scene_fingerprint=expected_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while the correction was being removed; "
                "retry on the latest frame."
            ),
        ) from exc
    except ReconstructionError as exc:
        status_code = 404 if str(exc) == "Frame annotation was not found" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return _analysis_with_queued_reconstruction(analysis, queued)


@router.put(
    "/{scene_id}/canonical-people/{canonical_person_id}/roster-binding",
    response_model=SceneDocument,
    status_code=202,
)
def update_canonical_roster_binding(
    project_id: str,
    scene_id: str,
    canonical_person_id: str,
    request: CanonicalRosterBindingRequest,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _require_segment(scene, "binding its roster")
    _require_idle(video, "editing roster identities")
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    match_snapshot = project_matches.current_snapshot(project_id)
    try:
        draft_canonical_roster_binding(
            scene,
            canonical_person_id,
            request.external_player_id,
            match_snapshot=(
                match_snapshot.payload if match_snapshot is not None else None
            ),
        )
        return queue_reconstruction(
            scene,
            match_snapshot=match_snapshot,
            expected_scene_fingerprint=expected_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while the roster binding was being saved; retry."
            ),
        ) from exc
    except ReconstructionError as exc:
        status_code = (
            404 if str(exc) == "The canonical person no longer exists" else 422
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.delete(
    "/{scene_id}/canonical-people/{canonical_person_id}/roster-binding",
    response_model=SceneDocument,
    status_code=202,
)
def clear_canonical_roster_binding_decision(
    project_id: str,
    scene_id: str,
    canonical_person_id: str,
) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = _require_segment(scene, "clearing its roster decision")
    _require_idle(video, "editing roster identities")
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        draft_clear_canonical_roster_binding(
            scene,
            canonical_person_id,
        )
        return queue_reconstruction(
            scene,
            match_snapshot=project_matches.current_snapshot(project_id),
            expected_scene_fingerprint=expected_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while the roster decision was being cleared; retry."
            ),
        ) from exc
    except ReconstructionError as exc:
        status_code = (
            404
            if str(exc)
            in {
                "The canonical person no longer exists",
                "This canonical person has no roster decision to clear",
            }
            else 422
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
