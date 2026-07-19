from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx

from .calibration_anchor_cache import (
    CalibrationAnchorCacheError,
    lookup_calibration_anchor_cache,
    store_calibration_anchor_cache,
)
from .config import get_settings
from .field_keypoints import calibration_from_worker_result
from .person_detection_cache import frame_content_sha256
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


def _anchor_cache_worker_contract() -> dict | None:
    """Resolve the worker's model identity; no identity means no caching."""

    readiness = calibration_worker_readiness(timeout=2.0)
    if readiness.get("status") != "ready":
        return None
    model_version = readiness.get("modelVersion")
    if not isinstance(model_version, str) or not model_version:
        return None
    return {
        "backend": str(readiness.get("backend")),
        "modelVersion": model_version,
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
    cache_directory = (
        Path(settings.media_root) / "calibration-anchors"
        if settings.calibration_anchor_cache_enabled
        else None
    )
    worker_contract = (
        _anchor_cache_worker_contract() if cache_directory is not None else None
    )
    frame_digests: dict[int, str] = {}
    pending: list[tuple[int, Path]] = []
    for index, path in frames:
        if worker_contract is None:
            pending.append((index, path))
            continue
        try:
            digest = frame_content_sha256(path)
            lookup = lookup_calibration_anchor_cache(
                cache_directory,
                frame_sha256=digest,
                worker_contract=worker_contract,
            )
        except (CalibrationAnchorCacheError, OSError):
            pending.append((index, path))
            continue
        frame_digests[index] = digest
        if lookup.status != "hit":
            pending.append((index, path))
            continue
        cached_item = lookup.entry.detached_item()
        if cached_item is not None:
            calibration = calibration_from_worker_result(cached_item)
            if calibration is not None:
                result[index] = calibration
        # A cached "no-solution" stays authoritative for this exact frame and
        # model; the worker is not asked again.
    if on_progress is not None and len(pending) < len(frames):
        on_progress(len(frames) - len(pending), len(frames), len(result))

    batch_size = max(1, int(settings.calibration_worker_batch_size))
    completed = len(frames) - len(pending)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
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
        solved_items: dict[int, dict] = {}
        for item in payload.get("frames") or []:
            enriched = {
                **item,
                "backendDiagnostics": backend_diagnostics,
            }
            frame_index = int(item.get("frameIndex") or 0)
            solved_items[frame_index] = enriched
            calibration = calibration_from_worker_result(enriched)
            if calibration is not None:
                result[frame_index] = calibration
        if worker_contract is not None:
            for index, path in batch:
                digest = frame_digests.get(index)
                try:
                    store_calibration_anchor_cache(
                        cache_directory,
                        frame_sha256=digest or frame_content_sha256(path),
                        worker_contract=worker_contract,
                        worker_item=solved_items.get(index),
                    )
                except (CalibrationAnchorCacheError, OSError):
                    # Cache IO must never invalidate a healthy worker result.
                    continue
        completed += len(batch)
        if on_progress is not None:
            on_progress(min(len(frames), completed), len(frames), len(result))
    if not result:
        raise CalibrationWorkerError("PnLCalib did not return a valid frame calibration")
    return result
