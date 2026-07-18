from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx

from .config import get_settings
from .field_keypoints import calibration_from_worker_result
from .pitch_calibration_contract import PitchCalibration


class CalibrationWorkerError(RuntimeError):
    pass


def calibration_worker_readiness(*, timeout: float = 2.0) -> dict:
    """Return an operational status without making the API itself unhealthy.

    The worker readiness endpoint loads and validates both PnLCalib models.  A
    short timeout keeps the general API health check useful when the optional
    service is stopped or still warming up.
    """

    settings = get_settings()
    if not settings.calibration_worker_url:
        return {
            "configured": False,
            "status": "disabled",
            "backend": None,
        }
    endpoint = f"{settings.calibration_worker_url.rstrip('/')}/health/ready"
    try:
        response = httpx.get(endpoint, timeout=max(0.1, float(timeout)))
        response.raise_for_status()
        payload = response.json()
    except (OSError, ValueError, httpx.HTTPError) as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": "pnlcalib-points-lines",
            "detail": str(exc),
        }
    backend = payload.get("backend")
    if payload.get("status") not in {"ok", "ready"} or backend not in {
        "pnlcalib",
        "pnlcalib-points-lines",
    }:
        return {
            "configured": True,
            "status": "invalid-response",
            "backend": backend,
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": backend,
        "device": payload.get("device"),
        "batchSize": payload.get("batchSize"),
        "modelVersion": payload.get("modelVersion"),
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
        "cacheEntryCount": payload.get("cacheEntryCount"),
    }


def calibrate_frames_with_worker(
    frames: list[tuple[int, Path]],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    timeout: float | None = None,
) -> dict[int, PitchCalibration]:
    settings = get_settings()
    if not settings.calibration_worker_url or not frames:
        return {}
    result: dict[int, PitchCalibration] = {}
    batch_size = max(1, int(settings.calibration_worker_batch_size))
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        files = [
            ("frames", (path.name, path.read_bytes(), "image/jpeg"))
            for _, path in batch
        ]
        try:
            response = httpx.post(
                f"{settings.calibration_worker_url.rstrip('/')}/v1/calibrate",
                data={"frame_indices": json.dumps([index for index, _ in batch])},
                files=files,
                timeout=(
                    min(float(timeout), float(settings.calibration_worker_timeout))
                    if timeout is not None
                    else settings.calibration_worker_timeout
                ),
            )
            response.raise_for_status()
            payload = response.json()
        except (OSError, ValueError, httpx.HTTPError) as exc:
            raise CalibrationWorkerError(f"Calibration worker failed: {exc}") from exc
        if payload.get("backend") not in {"pnlcalib", "pnlcalib-points-lines"}:
            raise CalibrationWorkerError("Calibration worker returned an unsupported backend")
        backend_diagnostics = payload.get("diagnostics")
        for item in payload.get("frames") or []:
            enriched = {
                **item,
                "backendDiagnostics": backend_diagnostics,
            }
            calibration = calibration_from_worker_result(enriched)
            if calibration is not None:
                result[int(item.get("frameIndex") or 0)] = calibration
        if on_progress is not None:
            on_progress(min(len(frames), start + len(batch)), len(frames), len(result))
    if not result:
        raise CalibrationWorkerError("PnLCalib did not return a valid frame calibration")
    return result
