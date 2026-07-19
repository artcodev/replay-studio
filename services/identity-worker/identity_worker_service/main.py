from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .cache import IdentityEmbeddingCache
from .embedding_service import IdentityEmbeddingService, IdentityInferenceError
from .evidence import EVIDENCE_FINGERPRINT_VERSION
from .provider_contract import (
    EMBEDDING_DIMENSION,
    IdentityEmbeddingProvider,
)
from .prtreid_provider import PRTReIDProvider
from .request_contract import IdentityRequestError


def create_app(
    provider: IdentityEmbeddingProvider | None = None,
    *,
    embedding_cache: IdentityEmbeddingCache | None = None,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or PRTReIDProvider()
    cache = embedding_cache or IdentityEmbeddingCache.from_environment(
        dimension=EMBEDDING_DIMENSION,
        environment=os.environ,
    )
    service = IdentityEmbeddingService(configured_provider, cache)
    should_preload = (
        os.environ.get("REID_PRELOAD", "1") not in {"0", "false", "False"}
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
        title="Replay Studio Identity Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
    application.state.identity_service = service
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

    @application.get("/health/live")
    async def health_live() -> dict:
        return {
            "status": "ok",
            "service": "replay-studio-identity-worker",
        }

    @application.get("/health/ready")
    async def health_ready() -> dict:
        await ensure_loaded()
        if not configured_provider.loaded:
            raise HTTPException(
                status_code=503, detail="Identity provider is not loaded"
            )
        return {
            "status": "ready",
            **configured_provider.info(),
            "evidenceFingerprintVersion": EVIDENCE_FINGERPRINT_VERSION,
            "cache": cache.stats(),
        }

    @application.post("/v1/embeddings")
    async def embeddings(
        crops: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict:
        await ensure_loaded()
        if not crops:
            raise HTTPException(status_code=422, detail="At least one crop is required")
        crop_bytes = [await crop.read() for crop in crops]
        try:
            return await run_in_threadpool(service.process, crop_bytes, manifest)
        except IdentityRequestError as exc:
            raise HTTPException(status_code=422, detail=exc.detail) from exc
        except IdentityInferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
