"""Shared transport rules for configured model workers."""

from __future__ import annotations

from io import BytesIO
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Sequence

from .manifest_contract import CropLabel, ManifestError


class WorkerUnavailable(RuntimeError):
    """A real worker or its configured assets are not ready."""


class WorkerProtocolError(RuntimeError):
    """A ready worker violated its published response contract."""


def batches(
    values: Sequence[CropLabel],
    size: int,
) -> Iterable[Sequence[CropLabel]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _worker_error(response: Any) -> str:
    try:
        value = response.json()
    except Exception:
        return str(response.text)[:500]
    if isinstance(value, dict) and value.get("detail"):
        return str(value["detail"])[:500]
    return str(value)[:500]


def response_object(response: Any, worker_name: str) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception as exc:
        raise WorkerProtocolError(f"{worker_name} did not return JSON") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"{worker_name} JSON response is not an object")
    return value


def require_ready(client: Any, base_url: str, worker_name: str) -> dict[str, Any]:
    try:
        response = client.get(f"{base_url.rstrip('/')}/health/ready")
    except Exception as exc:
        raise WorkerUnavailable(f"{worker_name} readiness request failed: {exc}") from exc
    if response.status_code != 200:
        raise WorkerUnavailable(
            f"{worker_name} is not ready (HTTP {response.status_code}): {_worker_error(response)}"
        )
    value = response_object(response, worker_name)
    if value.get("status") != "ready":
        raise WorkerProtocolError(f"{worker_name} readiness response is invalid")
    return value


def require_success(response: Any, worker_name: str) -> dict[str, Any]:
    if response.status_code == 503:
        raise WorkerUnavailable(
            f"{worker_name} inference unavailable: {_worker_error(response)}"
        )
    if response.status_code != 200:
        raise WorkerProtocolError(
            f"{worker_name} inference failed (HTTP {response.status_code}): {_worker_error(response)}"
        )
    return response_object(response, worker_name)


def image_size(data: bytes, crop_id: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:
        raise ManifestError(f"Crop {crop_id!r} is not a readable image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ManifestError(f"Crop {crop_id!r} has an empty image size")
    return int(width), int(height)
