from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .calibration_worker import CalibrationWorkerError
from .project_match_repository import project_matches
from .reconstruction_calibration_borrow import borrow_scene_pitch_calibration
from .reconstruction_calibration_edit_command import (
    save_scene_pitch_calibration_draft,
)
from .reconstruction_calibration_finalize_command import (
    finalize_scene_pitch_calibration_drafts,
)
from .reconstruction_calibration_manual_preview import (
    preview_scene_pitch_calibration,
)
from .reconstruction_calibration_proposal import propose_scene_pitch_calibration
from .scene_calibration_review_command import confirm_calibration_review
from .scene_calibration_reset_command import reset_scene_calibration
from .reconstruction_errors import ReconstructionError, StaleReconstructionRun
from .reconstruction_pitch_side_command import set_scene_pitch_side
from . import project_resource_access
from .calibration_contracts import (
    PitchCalibrationDraftRequest,
    PitchCalibrationBorrowRequest,
    PitchCalibrationPreviewRequest,
    PitchCalibrationSaveDraftRequest,
    PitchSideRequest,
)
from .scene_contracts import SceneDocument


router = APIRouter(prefix="/api/projects/{project_id}/scenes", tags=["calibration"])


def _calibration_scene(project_id: str, scene_id: str) -> dict:
    scene = project_resource_access.project_scene_or_404(project_id, scene_id)
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail="Choose a continuous shot scene before calibration",
        )
    return scene


def _require_scene_time(scene: dict, scene_time: float) -> None:
    if scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")


@router.post("/{scene_id}/pitch-calibration/auto")
def auto_pitch_calibration(
    project_id: str,
    scene_id: str,
    request: PitchCalibrationDraftRequest,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    _require_scene_time(scene, request.scene_time)
    try:
        return propose_scene_pitch_calibration(scene, request.scene_time, request.preset)
    except CalibrationWorkerError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "PnLCalib is required for automatic calibration and is unavailable: "
                f"{exc}"
            ),
        ) from exc
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{scene_id}/pitch-calibration/preview")
def preview_pitch_calibration(
    project_id: str,
    scene_id: str,
    request: PitchCalibrationPreviewRequest,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    _require_scene_time(scene, request.scene_time)
    try:
        return preview_scene_pitch_calibration(
            scene,
            request.scene_time,
            request.preset,
            [anchor.model_dump() for anchor in request.anchors],
        )
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/{scene_id}/pitch-calibration/borrow",
)
def borrow_pitch_calibration(
    project_id: str,
    scene_id: str,
    request: PitchCalibrationBorrowRequest,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    _require_scene_time(scene, request.scene_time)
    try:
        return borrow_scene_pitch_calibration(
            scene,
            request.scene_time,
            request.source,
            request.preset,
        )
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/{scene_id}/pitch-calibration/drafts",
    response_model=SceneDocument,
)
def save_pitch_calibration_draft(
    project_id: str,
    scene_id: str,
    request: PitchCalibrationSaveDraftRequest,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    _require_scene_time(scene, request.scene_time)
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Reconstruction is already running")
    try:
        return save_scene_pitch_calibration_draft(
            scene,
            request.scene_time,
            request.preset,
            [anchor.model_dump() for anchor in request.anchors],
            draft_source=request.source,
            accept_quality_warning=request.accept_quality_warning,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while calibration was being applied; "
                "reopen the latest frame and retry."
            ),
        ) from exc
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/{scene_id}/pitch-calibration/finalize",
    response_model=SceneDocument,
    status_code=202,
)
def finalize_pitch_calibration_drafts(
    project_id: str,
    scene_id: str,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    try:
        return finalize_scene_pitch_calibration_drafts(
            scene,
            match_snapshot=project_matches.current_snapshot(project_id),
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The scene changed while calibration finalization was queued; "
                "reload the timeline and retry."
            ),
        ) from exc
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{scene_id}/pitch-side", response_model=SceneDocument)
def change_scene_pitch_side(
    project_id: str,
    scene_id: str,
    request: PitchSideRequest,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    try:
        return set_scene_pitch_side(scene, request.side)
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{scene_id}/pitch-calibration/confirm-review",
    response_model=SceneDocument,
)
def confirm_pitch_calibration_review(
    project_id: str,
    scene_id: str,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    try:
        return confirm_calibration_review(scene)
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{scene_id}/pitch-calibration/reset",
    response_model=SceneDocument,
)
def reset_pitch_calibration(
    project_id: str,
    scene_id: str,
) -> dict:
    scene = _calibration_scene(project_id, scene_id)
    try:
        return reset_scene_calibration(scene)
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
