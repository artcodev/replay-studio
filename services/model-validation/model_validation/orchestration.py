"""Application service coordinating real worker validation."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

from .identity_evaluator import evaluate_identity
from .identity_worker_client import fetch_identity_predictions
from .jersey_evaluator import evaluate_jersey_ocr
from .jersey_worker_client import fetch_jersey_predictions
from .manifest_contract import ValidationManifest
from .reports import build_report
from .worker_transport import WorkerUnavailable


def _benchmark(
    *,
    started: float,
    crop_count: int,
    batch_size: int,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    elapsed = perf_counter() - started
    return {
        "scope": "ready-check+http+crop-qa+provider-inference",
        "wallSeconds": round(elapsed, 6),
        "cropCount": crop_count,
        "cropsPerSecond": round(crop_count / max(elapsed, 1e-9), 6),
        "requestBatchSize": batch_size,
        "requestCount": len(diagnostics),
        "batchDiagnostics": diagnostics,
    }


def run_http_validation(
    manifest: ValidationManifest,
    *,
    workers: Sequence[str],
    identity_url: str = "http://127.0.0.1:8091",
    jersey_ocr_url: str = "http://127.0.0.1:8093",
    batch_size: int = 16,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    selected = tuple(dict.fromkeys(workers))
    if not selected or any(worker not in {"identity", "jersey-ocr"} for worker in selected):
        raise ValueError("workers must contain identity and/or jersey-ocr")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        import httpx
    except ImportError as exc:
        raise WorkerUnavailable("httpx is required to call the configured workers") from exc

    identity_result = None
    jersey_result = None
    with httpx.Client(timeout=timeout_seconds) as client:
        if "identity" in selected:
            started = perf_counter()
            provider, predictions, diagnostics = fetch_identity_predictions(
                client,
                identity_url,
                manifest,
                batch_size,
            )
            identity_result = evaluate_identity(manifest, provider, predictions)
            identity_result["benchmark"] = _benchmark(
                started=started,
                crop_count=len(manifest.crops),
                batch_size=batch_size,
                diagnostics=diagnostics,
            )
        if "jersey-ocr" in selected:
            started = perf_counter()
            provider, predictions, diagnostics = fetch_jersey_predictions(
                client,
                jersey_ocr_url,
                manifest,
                batch_size,
            )
            jersey_result = evaluate_jersey_ocr(manifest, provider, predictions)
            jersey_result["benchmark"] = _benchmark(
                started=started,
                crop_count=len(manifest.crops),
                batch_size=batch_size,
                diagnostics=diagnostics,
            )
    return build_report(
        manifest,
        identity=identity_result,
        jersey_ocr=jersey_result,
    )
