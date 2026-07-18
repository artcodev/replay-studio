from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .database import init_database
from .health_routes import router as health_router
from .identity_decision_routes import router as identity_decision_router
from .identity_review_routes import router as identity_review_router
from .match_import_routes import router as match_import_router
from .project_routes import router as project_router
from .scene_analysis_routes import router as scene_analysis_router
from .scene_calibration_routes import router as scene_calibration_router
from .scene_document_routes import router as scene_document_router
from .scene_identity_routes import router as scene_identity_router
from .scene_document import SceneRevisionConflict
from .video_routes import router as video_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


async def scene_revision_conflict_handler(
    _request: Request,
    _error: SceneRevisionConflict,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "The scene changed while it was being saved; reload and retry."
        },
    )


def create_app() -> FastAPI:
    """Build the HTTP composition root; route modules own endpoint behavior."""

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(
        SceneRevisionConflict,
        scene_revision_conflict_handler,
    )
    for router in (
        health_router,
        project_router,
        match_import_router,
        video_router,
        scene_document_router,
        scene_analysis_router,
        scene_identity_router,
        scene_calibration_router,
        identity_review_router,
        identity_decision_router,
    ):
        application.include_router(router)
    return application


app = create_app()
