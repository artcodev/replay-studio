from __future__ import annotations

"""HTTP transport for the jersey OCR worker; no batching or domain policy."""

from typing import Any

import httpx

from .jersey_ocr_worker_contract import JerseyOcrWorkerError


class JerseyOcrTransportError(JerseyOcrWorkerError):
    pass


def fetch_readiness(
    worker_url: str,
    *,
    timeout: float,
) -> object:
    try:
        response = httpx.get(
            f"{worker_url.rstrip('/')}/health/ready",
            timeout=max(0.1, float(timeout)),
        )
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise JerseyOcrTransportError(str(exc)) from exc


def post_analysis_batch(
    worker_url: str,
    *,
    files: list[tuple[str, tuple[str, bytes, str]]],
    manifest: str,
    timeout: float,
) -> object:
    try:
        response = httpx.post(
            f"{worker_url.rstrip('/')}/v1/analyze",
            files=files,
            data={"manifest": manifest},
            timeout=float(timeout),
        )
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise JerseyOcrTransportError(str(exc)) from exc
