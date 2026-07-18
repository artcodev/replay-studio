from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .analysis_policy import QualityPolicy
from .analysis_service import JerseyInferenceError, JerseyOcrService, provider_info
from .provider_contract import JerseyOcrProvider, ProviderUnavailable
from .provider_factory import provider_from_environment
from .request_contract import (
    CONTRACT_VERSION,
    SERVICE_NAME,
    JerseyRequestError,
    capabilities,
)
from .result_cache import OcrResultCache


def create_app(
    provider: JerseyOcrProvider | None = None,
    *,
    quality_policy: QualityPolicy | None = None,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or provider_from_environment()
    policy_error: str | None = None
    try:
        policy = quality_policy or QualityPolicy.from_environment()
        policy.validate()
    except (TypeError, ValueError, ProviderUnavailable) as exc:
        policy = QualityPolicy()
        policy_error = f"Invalid jersey OCR policy: {exc}"
    should_preload = (
        os.environ.get("JERSEY_OCR_PRELOAD", "1") not in {"0", "false", "False"}
        if preload is None
        else preload
    )
    try:
        result_cache = OcrResultCache(
            int(os.environ.get("JERSEY_OCR_CACHE_MAX_ENTRIES", "4096")),
            float(os.environ.get("JERSEY_OCR_CACHE_TTL_SECONDS", "86400")),
        )
    except (TypeError, ValueError) as exc:
        result_cache = OcrResultCache(0, 0)
        policy_error = policy_error or f"Invalid jersey OCR cache policy: {exc}"
    service = JerseyOcrService(configured_provider, policy, result_cache)

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
        title="Replay Studio Jersey OCR Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
    application.state.ocr_service = service
    application.state.provider_error = None

    async def ensure_loaded() -> dict[str, Any]:
        if policy_error is not None:
            raise HTTPException(status_code=503, detail=policy_error)
        if not configured_provider.loaded:
            try:
                await run_in_threadpool(configured_provider.load)
                application.state.provider_error = None
            except Exception as exc:
                application.state.provider_error = str(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            return provider_info(configured_provider)
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/health/live")
    async def health_live() -> dict[str, Any]:
        return {"status": "ok", "service": SERVICE_NAME}

    @application.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        info = await ensure_loaded()
        return {
            "status": "ready",
            "service": SERVICE_NAME,
            "contractVersion": CONTRACT_VERSION,
            "capabilities": capabilities(),
            **info,
        }

    @application.post("/v1/analyze")
    async def analyze(
        crops: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict[str, Any]:
        info = await ensure_loaded()
        if not crops:
            raise HTTPException(status_code=422, detail="At least one crop is required")
        if len(crops) > policy.max_batch_size:
            raise HTTPException(status_code=413, detail="OCR batch exceeds the file limit")
        crop_bytes = [await crop.read() for crop in crops]
        try:
            return await run_in_threadpool(service.analyze, crop_bytes, manifest, info)
        except JerseyRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except JerseyInferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
