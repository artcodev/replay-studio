from __future__ import annotations

import os
from time import perf_counter

from .calibration_cache import CalibrationResultCache
from .pnlcalib_engine import PnLCalibEngine
from .pnlcalib_inference import PnLCalibInference
from .pnlcalib_runtime import load_pnlcalib_models


def create_pnlcalib_engine(
    *,
    cache_max_entries: int | None = None,
) -> PnLCalibEngine:
    """Compose the pinned models, inference adapter, cache and engine."""

    started = perf_counter()
    models = load_pnlcalib_models()
    resolved_cache_max_entries = (
        max(0, cache_max_entries)
        if cache_max_entries is not None
        else max(0, int(os.environ.get("PNLCALIB_CACHE_MAX_ENTRIES", "512")))
    )
    cache_ttl_seconds = max(
        0.0,
        float(os.environ.get("PNLCALIB_CACHE_TTL_SECONDS", "3600")),
    )
    return PnLCalibEngine(
        inference=PnLCalibInference(models),
        model_version=models.model_version,
        device=str(models.device),
        # The pinned points+lines CPU runtime is only process-safe for a
        # single-frame tensor. The service may accept a multi-frame HTTP
        # request, but evaluates those frames sequentially here.
        batch_size=max(1, int(os.environ.get("PNLCALIB_BATCH_SIZE", "1"))),
        cache=CalibrationResultCache(
            max_entries=resolved_cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        ),
        model_load_seconds=perf_counter() - started,
    )
