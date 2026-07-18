from __future__ import annotations

import re
from pathlib import Path

from .config import get_settings
from .video_processing_contract import VideoProcessingError


def asset_directory(asset_id: str) -> Path:
    return Path(get_settings().media_root).resolve() / asset_id


def video_generation_directory(
    asset_id: str,
    generation_key: str,
    *,
    media_root: Path | None = None,
) -> Path:
    """Resolve one immutable derived-media generation without path traversal."""

    if (
        not generation_key
        or generation_key in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", generation_key) is None
    ):
        raise VideoProcessingError("Video generation key is invalid")
    base = (
        Path(media_root).resolve() / asset_id
        if media_root is not None
        else asset_directory(asset_id)
    )
    return base / ".pipeline-runs" / generation_key


def published_video_directory(asset: dict) -> Path:
    generation_key = str(asset.get("generation_key") or "")
    if not generation_key:
        raise VideoProcessingError("Video asset has no published generation")
    return video_generation_directory(str(asset["id"]), generation_key)
