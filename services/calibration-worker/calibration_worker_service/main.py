from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .calibration_service import (
    CalibrationInferenceError,
    CalibrationRequestError,
    CalibrationService,
)
from .engine_factory import create_pnlcalib_engine
from .runtime import CalibrationEngineRuntime


def create_app(
    service: CalibrationService | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    configured_service = service or CalibrationService(
        CalibrationEngineRuntime(create_pnlcalib_engine)
    )
    should_preload = (
        os.environ.get("PNLCALIB_PRELOAD", "1").lower() not in {"0", "false", "no"}
        if preload is None
        else preload
    )

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        if should_preload:
            try:
                await run_in_threadpool(configured_service.readiness)
            except Exception:
                # Liveness remains inspectable while readiness exposes the model error.
                pass
        yield

    application = FastAPI(
        title="Replay Studio PnLCalib Worker",
        version="1.1.0",
        lifespan=lifespan,
    )
    application.state.calibration_service = configured_service

    @application.get("/health/live")
    def liveness() -> dict:
        return {"status": "ok", "service": "pnlcalib-worker"}

    @application.get("/health/ready")
    async def readiness() -> dict:
        try:
            details = await run_in_threadpool(configured_service.readiness)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ready", **details}

    @application.get("/health")
    async def health() -> dict:
        return await readiness()

    @application.post("/v1/calibrate")
    async def calibrate(
        frames: list[UploadFile] = File(...),
        frame_indices: str = Form(...),
    ) -> dict:
        payloads = [await upload.read() for upload in frames]
        try:
            return await run_in_threadpool(
                configured_service.calibrate,
                frame_indices,
                payloads,
            )
        except CalibrationRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CalibrationInferenceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return application


app = create_app()
