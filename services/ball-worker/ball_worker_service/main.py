from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .detection_service import BallDetectionService, BallInferenceError
from .provider_contract import BallDetectionProvider
from .provider_factory import provider_from_environment
from .request_contract import SERVICE_NAME, BallRequestError
from .settings import BallWorkerSettings


def create_app(
    provider: BallDetectionProvider | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or provider_from_environment()
    settings = BallWorkerSettings.from_environment()
    service = BallDetectionService(configured_provider, settings)
    should_preload = (
        os.environ.get("WASB_PRELOAD", "1") not in {"0", "false", "False"}
        if preload is None
        else preload
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.provider_error = None
        if should_preload:
            try:
                await run_in_threadpool(configured_provider.load)
            except Exception as exc:
                application.state.provider_error = str(exc)
        yield

    application = FastAPI(
        title="Replay Studio WASB Ball Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
    application.state.detection_service = service
    application.state.provider_error = None

    async def ensure_loaded() -> None:
        if configured_provider.loaded:
            application.state.provider_error = None
            return
        try:
            await run_in_threadpool(configured_provider.load)
            application.state.provider_error = None
        except Exception as exc:
            application.state.provider_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not configured_provider.loaded:
            raise HTTPException(
                status_code=503, detail="WASB provider did not become ready"
            )

    async def execute(call, *args):
        await ensure_loaded()
        try:
            return await run_in_threadpool(call, *args)
        except BallRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except BallInferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/health/live")
    async def health_live() -> dict[str, Any]:
        return {"status": "ok", "service": SERVICE_NAME}

    @application.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        await ensure_loaded()
        return {
            "status": "ready",
            "service": SERVICE_NAME,
            **configured_provider.info().to_wire(),
        }

    @application.post("/v1/detections")
    async def batch_detections(
        frames: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict[str, Any]:
        if not frames:
            raise HTTPException(status_code=422, detail="At least one frame is required")
        if len(frames) > settings.max_batch_frames:
            raise HTTPException(
                status_code=413,
                detail=f"At most {settings.max_batch_frames} files are allowed",
            )
        frame_bytes = [
            await frame.read(settings.max_frame_bytes + 1) for frame in frames
        ]
        return await execute(service.batch_detections, frame_bytes, manifest)

    return application


app = create_app()
