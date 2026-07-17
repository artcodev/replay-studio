"""HTTP boundary for the read-only identity review workbench."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from .identity_review import (
    IdentityReviewError,
    build_identity_review,
    identity_observation_crop,
)
from .identity_worker import identity_worker_readiness
from .jersey_ocr_worker import jersey_ocr_worker_readiness
from .store import scene_store


router = APIRouter()


@router.get("/api/scenes/{scene_id}/identity-review")
def get_identity_review(scene_id: str) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return build_identity_review(
        scene,
        worker_health={
            "reid": identity_worker_readiness(timeout=0.75),
            "jerseyOcr": jersey_ocr_worker_readiness(timeout=0.75),
        },
    )


@router.get("/api/scenes/{scene_id}/identity-observations/{observation_id}/crop")
def get_identity_observation_crop(scene_id: str, observation_id: str) -> Response:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
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
