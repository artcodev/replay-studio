"""HTTP boundary for the read-only identity review workbench."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from .identity_review_contract import IdentityReviewResponse
from .identity_review_crop_service import identity_observation_crop
from .identity_review_errors import (
    IdentityReviewArtifactUnavailableError,
    IdentityReviewError,
)
from .identity_review_http_presenter import present_identity_review
from .identity_review_projection import build_identity_review_projection
from .identity_worker_client import identity_worker_readiness
from .jersey_ocr_worker_client import jersey_ocr_worker_readiness
from .project_match_repository import project_matches
from .project_resource_repository import ProjectResourceConflict, project_resources
from .scene_repository import scenes


router = APIRouter()


def _owned_scene(project_id: str, scene_id: str) -> dict:
    try:
        owner = project_resources.scene_owner(scene_id)
    except ProjectResourceConflict as exc:
        raise HTTPException(status_code=500, detail="Scene ownership is inconsistent") from exc
    scene = scenes.get(scene_id) if owner == project_id else None
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found in project")
    return scene


@router.get(
    "/api/projects/{project_id}/scenes/{scene_id}/identity-review",
    response_model=IdentityReviewResponse,
    response_model_exclude_unset=True,
)
def get_identity_review(project_id: str, scene_id: str) -> IdentityReviewResponse:
    scene = _owned_scene(project_id, scene_id)
    try:
        review = build_identity_review_projection(
            scene,
            match_snapshot=project_matches.current_snapshot(project_id),
            worker_health={
                "reid": identity_worker_readiness(timeout=0.75),
                "jerseyOcr": jersey_ocr_worker_readiness(timeout=0.75),
            },
        )
        return IdentityReviewResponse.model_validate(
            present_identity_review(
                review,
                project_id=project_id,
                scene_id=scene_id,
            )
        )
    except IdentityReviewArtifactUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IdentityReviewError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/api/projects/{project_id}/scenes/{scene_id}/identity-observations/{observation_id}/crop"
)
def get_identity_observation_crop(
    project_id: str,
    scene_id: str,
    observation_id: str,
) -> Response:
    scene = _owned_scene(project_id, scene_id)
    try:
        content = identity_observation_crop(scene, observation_id)
    except IdentityReviewError as exc:
        status = 404 if str(exc) == "Identity observation was not found" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = ["router"]
