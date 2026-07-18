"""Aggregate the project HTTP capabilities without owning endpoint behavior."""

from fastapi import APIRouter

from .project_analysis_routes import router as analysis_router
from .project_core_routes import router as core_router
from .project_identity_routes import router as identity_router
from .project_match_routes import router as match_router
from .project_media_routes import router as media_router


router = APIRouter()
for capability_router in (
    core_router,
    match_router,
    identity_router,
    media_router,
    analysis_router,
):
    router.include_router(capability_router)
