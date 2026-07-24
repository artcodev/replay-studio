from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .detection_contract import DetectionRequestError
from .detection_service import PersonDetectionService
from .ultralytics_engine import (
    DetectionEngineUnavailable,
    UltralyticsDetectionEngine,
)


def create_app(
    engine: UltralyticsDetectionEngine | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    configured_engine = engine or UltralyticsDetectionEngine()
    service = PersonDetectionService(configured_engine)
    should_preload = (
        os.environ.get("PERSON_DETECTION_PRELOAD", "1")
        not in {"0", "false", "False"}
        if preload is None
        else preload
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.engine_error = None
        if should_preload:
            try:
                await run_in_threadpool(configured_engine.load)
            except Exception as exc:
                application.state.engine_error = str(exc)
        yield

    application = FastAPI(
        title="Replay Studio Person Detection Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.engine = configured_engine
    application.state.service = service
    application.state.engine_error = None

    async def ensure_loaded() -> None:
        if configured_engine.loaded:
            application.state.engine_error = None
            return
        try:
            await run_in_threadpool(configured_engine.load)
            application.state.engine_error = None
        except Exception as exc:
            application.state.engine_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/health/live")
    async def health_live() -> dict:
        return {
            "status": "ok",
            "service": "replay-studio-person-detection-worker",
        }

    @application.get("/health/ready")
    async def health_ready() -> dict:
        await ensure_loaded()
        return {"status": "ready", **configured_engine.info()}

    @application.post("/v1/detections")
    async def detections(
        frame: UploadFile = File(...),
        manifest: str = Form(...),
    ) -> dict:
        await ensure_loaded()
        frame_bytes = await frame.read()
        if not frame_bytes:
            raise HTTPException(status_code=422, detail="Frame is empty")
        try:
            return await run_in_threadpool(
                service.process,
                frame_bytes,
                manifest,
            )
        except DetectionRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DetectionEngineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
