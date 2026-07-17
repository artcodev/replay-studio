"""Readiness reporting for the optional WASB challenger service."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import get_settings


def ball_worker_readiness(*, timeout: float = 2.0) -> dict:
    settings = get_settings()
    configured = settings.ball_wasb_worker_url
    if not configured:
        return {"configured": False, "status": "disabled", "backend": None}
    parsed = urlsplit(configured)
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, "/health/ready", "", ""))
    try:
        response = httpx.get(endpoint, timeout=max(0.1, float(timeout)))
        response.raise_for_status()
        payload = response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        return {
            "configured": True,
            "selected": settings.ball_detection_backend == "wasb-service",
            "status": "unavailable",
            "backend": "wasb-sbdt-soccer",
            "detail": str(exc),
        }
    if (
        payload.get("status") != "ready"
        or payload.get("backend") != "wasb-sbdt-soccer"
        or not payload.get("modelVersion")
    ):
        return {
            "configured": True,
            "selected": settings.ball_detection_backend == "wasb-service",
            "status": "invalid-response",
            "backend": payload.get("backend"),
        }
    return {
        "configured": True,
        "selected": settings.ball_detection_backend == "wasb-service",
        "status": "ready",
        "backend": payload["backend"],
        "modelVersion": payload["modelVersion"],
        "device": payload.get("device"),
        "framesIn": payload.get("framesIn"),
        "framesOut": payload.get("framesOut"),
        "inputSize": payload.get("inputSize"),
        "scoreThreshold": payload.get("scoreThreshold"),
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
    }
