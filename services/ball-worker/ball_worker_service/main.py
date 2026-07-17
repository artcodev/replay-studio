from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
import io
import json
import os
from typing import Any, Mapping, Sequence

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .providers import (
    BallCandidate,
    BallDetectionProvider,
    ProviderUnavailable,
    WasbSoccerProvider,
)


SERVICE_NAME = "replay-studio-ball-worker"
CONTRACT_VERSION = 1


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HTTPException(status_code=422, detail=f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise HTTPException(status_code=422, detail=f"{label} must not exceed {maximum}")
    return value


def _decode_image(data: bytes, *, label: str, max_bytes: int, max_pixels: int) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=422, detail=f"{label} is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds WASB_MAX_FRAME_BYTES={max_bytes}",
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise HTTPException(
                    status_code=413,
                    detail=f"{label} exceeds WASB_MAX_FRAME_PIXELS={max_pixels}",
                )
            image = source.convert("RGB")
            return np.asarray(image).copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label} is not a readable image") from exc


def _decode_base64(value: Any, *, label: str, max_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=422, detail=f"{label}.dataBase64 is required")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label}.dataBase64 is invalid") from exc
    if len(decoded) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds WASB_MAX_FRAME_BYTES={max_bytes}",
        )
    return decoded


def _decode_contract_frame(
    value: Any,
    *,
    index: int,
    max_bytes: int,
    max_pixels: int,
) -> np.ndarray:
    label = f"frames[{index}]"
    if not isinstance(value, Mapping):
        raise HTTPException(status_code=422, detail=f"{label} must be an object")
    encoding = value.get("encoding")
    raw = _decode_base64(value.get("dataBase64"), label=label, max_bytes=max_bytes)
    if encoding == "file-base64":
        return _decode_image(raw, label=label, max_bytes=max_bytes, max_pixels=max_pixels)
    if encoding != "numpy-base64":
        raise HTTPException(status_code=422, detail=f"{label}.encoding is unsupported")

    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) not in (2, 3)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise HTTPException(status_code=422, detail=f"{label}.shape is invalid")
    if int(np.prod(shape, dtype=np.int64)) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{label}.shape exceeds the byte limit")
    if shape[0] * shape[1] > max_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds WASB_MAX_FRAME_PIXELS={max_pixels}",
        )
    if value.get("dtype") != "uint8":
        raise HTTPException(status_code=422, detail=f"{label}.dtype must be uint8")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if len(raw) != expected_bytes:
        raise HTTPException(status_code=422, detail=f"{label} byte count does not match shape")
    array = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).copy()
    if array.shape[2] not in (1, 3, 4):
        raise HTTPException(status_code=422, detail=f"{label} must have 1, 3, or 4 channels")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    color_space = str(value.get("colorSpace") or "RGB").upper()
    if color_space == "BGR":
        array = array[:, :, ::-1]
    elif color_space not in ("RGB", "GRAY"):
        raise HTTPException(status_code=422, detail=f"{label}.colorSpace is unsupported")
    return np.ascontiguousarray(array)


def _parse_manifest(raw: str, frame_count: int, max_candidates_default: int) -> tuple[list[dict[str, Any]], int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="manifest must be an object")
    if value.get("contractVersion", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise HTTPException(status_code=422, detail="Unsupported manifest contractVersion")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise HTTPException(status_code=422, detail="manifest.frames must be a non-empty array")
    max_candidates = _positive_int(
        value.get("maxCandidates", max_candidates_default),
        "manifest.maxCandidates",
        maximum=100,
    )
    parsed: list[dict[str, Any]] = []
    for offset, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise HTTPException(status_code=422, detail=f"manifest.frames[{offset}] must be an object")
        file_index = frame.get("fileIndex")
        if (
            isinstance(file_index, bool)
            or not isinstance(file_index, int)
            or not 0 <= file_index < frame_count
        ):
            raise HTTPException(status_code=422, detail=f"manifest.frames[{offset}].fileIndex is invalid")
        frame_index = frame.get("frameIndex", offset)
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise HTTPException(status_code=422, detail=f"manifest.frames[{offset}].frameIndex is invalid")
        timestamp = frame.get("timestamp")
        timestamp_ms = frame.get("timestampMs")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            raise HTTPException(status_code=422, detail=f"manifest.frames[{offset}].timestamp is invalid")
        if timestamp_ms is not None and (
            isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, float))
        ):
            raise HTTPException(status_code=422, detail=f"manifest.frames[{offset}].timestampMs is invalid")
        parsed.append(
            {
                "fileIndex": file_index,
                "frameIndex": frame_index,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "timestampMs": float(timestamp_ms) if timestamp_ms is not None else None,
            }
        )
    return parsed, max_candidates


def _candidate_json(
    candidate: BallCandidate,
    *,
    provider: BallDetectionProvider,
    model_version: str,
    source_frame_index: int,
    radius: float,
) -> dict[str, Any]:
    confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
    return {
        "position": [round(candidate.x, 4), round(candidate.y, 4)],
        "x": round(candidate.x, 4),
        "y": round(candidate.y, 4),
        "radius": radius,
        "confidence": round(confidence, 6),
        "backend": provider.backend,
        "modelVersion": model_version,
        "observed": True,
        "sourceFrameIndex": source_frame_index,
        "heatmapPeak": round(candidate.heatmap_peak, 6),
        "componentScore": round(candidate.component_score, 6),
        "componentArea": candidate.component_area,
        **candidate.metadata,
    }


def _provider_identity(provider: BallDetectionProvider) -> tuple[dict[str, Any], str]:
    info = provider.info()
    model_version = info.get("modelVersion")
    if not isinstance(model_version, str) or not model_version:
        raise ProviderUnavailable("Ball provider did not report modelVersion")
    return info, model_version


def _detect_sequence(
    provider: BallDetectionProvider,
    images: Sequence[np.ndarray],
    manifest_frames: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    radius: float,
) -> tuple[list[dict[str, Any]], int]:
    if provider.frames_in != provider.frames_out:
        raise ProviderUnavailable("This worker requires equal WASB framesIn and framesOut")
    window_size = provider.frames_in
    info, model_version = _provider_identity(provider)
    del info
    responses: list[dict[str, Any]] = []
    window_count = 0
    for start in range(0, len(images), window_size):
        source_indices = [min(start + offset, len(images) - 1) for offset in range(window_size)]
        window = [images[index] for index in source_indices]
        detections = provider.detect_window(window, max_candidates=max_candidates)
        if len(detections) != provider.frames_out:
            raise ProviderUnavailable("Ball provider returned an incomplete temporal window")
        window_count += 1
        for offset in range(min(window_size, len(images) - start)):
            sequence_index = start + offset
            frame = manifest_frames[sequence_index]
            image = images[sequence_index]
            frame_index = int(frame["frameIndex"])
            responses.append(
                {
                    **frame,
                    "imageSize": [int(image.shape[1]), int(image.shape[0])],
                    "temporalPadding": len(set(source_indices)) != len(source_indices),
                    "candidates": [
                        _candidate_json(
                            candidate,
                            provider=provider,
                            model_version=model_version,
                            source_frame_index=frame_index,
                            radius=radius,
                        )
                        for candidate in detections[offset]
                    ],
                }
            )
    return responses, window_count


def _window_for_target(
    images: Sequence[np.ndarray],
    target_index: int,
    window_size: int,
) -> tuple[list[np.ndarray], int, list[int]]:
    if window_size != 3:
        raise ProviderUnavailable("The current WASB soccer contract requires a 3-frame model")
    if len(images) >= window_size:
        start = min(max(target_index - (window_size - 1), 0), len(images) - window_size)
        indices = list(range(start, start + window_size))
        return [images[index] for index in indices], target_index - start, indices
    if len(images) == 2:
        indices = [0, 1, 1] if target_index == 0 else [0, 0, 1]
        output_offset = 0 if target_index == 0 else 2
        return [images[index] for index in indices], output_offset, indices
    return [images[0], images[0], images[0]], 1, [0, 0, 0]


def create_app(
    provider: BallDetectionProvider | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    configured_provider = provider or WasbSoccerProvider()
    should_preload = (
        os.environ.get("WASB_PRELOAD", "1") not in {"0", "false", "False"}
        if preload is None
        else preload
    )
    max_frame_bytes = max(1, int(os.environ.get("WASB_MAX_FRAME_BYTES", str(32 * 1024 * 1024))))
    max_frame_pixels = max(1, int(os.environ.get("WASB_MAX_FRAME_PIXELS", "16000000")))
    max_batch_frames = max(1, int(os.environ.get("WASB_MAX_BATCH_FRAMES", "96")))
    max_candidates_default = max(1, int(os.environ.get("WASB_MAX_CANDIDATES", "12")))
    candidate_radius = max(0.1, float(os.environ.get("WASB_CANDIDATE_RADIUS", "4")))

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
        title="Replay Studio WASB Ball Worker",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.provider = configured_provider
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
        if not configured_provider.loaded:
            raise HTTPException(status_code=503, detail="WASB provider did not become ready")

    async def run_provider(call, *args):
        await ensure_loaded()
        try:
            return await run_in_threadpool(call, *args)
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"WASB inference failed: {type(exc).__name__}: {exc}",
            ) from exc

    @application.get("/health/live")
    async def health_live() -> dict[str, Any]:
        return {"status": "ok", "service": SERVICE_NAME}

    @application.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        await ensure_loaded()
        return {"status": "ready", "service": SERVICE_NAME, **configured_provider.info()}

    @application.post("/v1/detections")
    async def batch_detections(
        frames: list[UploadFile] = File(...),
        manifest: str = Form(...),
    ) -> dict[str, Any]:
        if not frames:
            raise HTTPException(status_code=422, detail="At least one frame is required")
        if len(frames) > max_batch_frames:
            raise HTTPException(status_code=413, detail=f"At most {max_batch_frames} files are allowed")
        manifest_frames, max_candidates = _parse_manifest(
            manifest,
            len(frames),
            max_candidates_default,
        )
        if len(manifest_frames) > max_batch_frames:
            raise HTTPException(status_code=413, detail=f"At most {max_batch_frames} frames are allowed")
        file_bytes = [await frame.read(max_frame_bytes + 1) for frame in frames]
        decoded_by_file_index = [
            _decode_image(
                data,
                label=f"frames[{file_index}]",
                max_bytes=max_frame_bytes,
                max_pixels=max_frame_pixels,
            )
            for file_index, data in enumerate(file_bytes)
        ]
        images = [decoded_by_file_index[item["fileIndex"]] for item in manifest_frames]

        result_frames, window_count = await run_provider(
            lambda: _detect_sequence(
                configured_provider,
                images,
                manifest_frames,
                max_candidates=max_candidates,
                radius=candidate_radius,
            )
        )
        info, model_version = _provider_identity(configured_provider)
        return {
            "contractVersion": CONTRACT_VERSION,
            "backend": configured_provider.backend,
            "modelVersion": model_version,
            "frames": result_frames,
            "metadata": {
                **info,
                "windowCount": window_count,
                "requestedFrameCount": len(manifest_frames),
            },
        }

    @application.post("/detect")
    async def compatible_detection(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if payload.get("contractVersion", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise HTTPException(status_code=422, detail="Unsupported contractVersion")
        encoded_frames = payload.get("frames")
        if not isinstance(encoded_frames, list) or not encoded_frames:
            raise HTTPException(status_code=422, detail="frames must be a non-empty array")
        if len(encoded_frames) > max_batch_frames:
            raise HTTPException(status_code=413, detail=f"At most {max_batch_frames} frames are allowed")
        target_index = payload.get("targetIndex")
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < len(encoded_frames)
        ):
            raise HTTPException(status_code=422, detail="targetIndex is out of range")
        max_candidates = _positive_int(
            payload.get("maxCandidates", max_candidates_default),
            "maxCandidates",
            maximum=100,
        )
        images = [
            _decode_contract_frame(
                value,
                index=index,
                max_bytes=max_frame_bytes,
                max_pixels=max_frame_pixels,
            )
            for index, value in enumerate(encoded_frames)
        ]
        window, output_offset, source_indices = _window_for_target(
            images,
            target_index,
            configured_provider.frames_in,
        )
        detections = await run_provider(
            lambda: configured_provider.detect_window(window, max_candidates=max_candidates)
        )
        if len(detections) != configured_provider.frames_out:
            raise HTTPException(status_code=503, detail="Ball provider returned an incomplete window")
        info, model_version = _provider_identity(configured_provider)
        target = images[target_index]
        source_frame_index = payload.get("frameIndex", target_index)
        if isinstance(source_frame_index, bool) or not isinstance(source_frame_index, int):
            raise HTTPException(status_code=422, detail="frameIndex must be an integer")
        return {
            "contractVersion": CONTRACT_VERSION,
            "imageSize": [int(target.shape[1]), int(target.shape[0])],
            "candidates": [
                _candidate_json(
                    candidate,
                    provider=configured_provider,
                    model_version=model_version,
                    source_frame_index=source_frame_index,
                    radius=candidate_radius,
                )
                for candidate in detections[output_offset]
            ],
            "metadata": {
                **info,
                "targetIndex": target_index,
                "sourceWindowIndices": source_indices,
                "temporalPadding": len(set(source_indices)) != len(source_indices),
            },
        }

    return application


app = create_app()

