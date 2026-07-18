from __future__ import annotations

from .config import get_settings
from .reconstruction_errors import ReconstructionError


def set_reconstruction_status_in_memory(
    scene: dict,
    status: str,
    **values,
) -> dict:
    """Update the working document without any persistence side effect."""

    video = scene.get("payload", {}).get("videoAsset")
    if video is None:
        raise ReconstructionError("Scene has no source video")
    current = video.get("reconstruction") or {}
    model_name = values.pop("model", None) or current.get("model") or get_settings().reconstruction_model
    video["reconstruction"] = {
        **current,
        "status": status,
        "model": model_name,
        **values,
    }
    return scene

