from __future__ import annotations

"""Read one exact immutable analysis JPEG belonging to a Scene generation."""

from pathlib import Path

from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import native_frame_paths, source_frame_index


def exact_scene_analysis_frame(
    scene: dict,
    generation_key: str,
    requested_source_frame_index: int,
) -> Path:
    video = scene.get("payload", {}).get("videoAsset") or {}
    current_generation_key = str(video.get("generationKey") or "")
    if not current_generation_key:
        raise ReconstructionError("Scene has no published video generation")
    if generation_key != current_generation_key:
        raise ReconstructionError(
            "The requested analysis frame belongs to an older video generation"
        )
    if requested_source_frame_index <= 0:
        raise ReconstructionError("Source frame index must be positive")
    selected = next(
        (
            path
            for path, _scene_time in native_frame_paths(scene)
            if source_frame_index(path) == requested_source_frame_index
        ),
        None,
    )
    if selected is None:
        raise ReconstructionError(
            "The requested analysis frame is outside this scene segment"
        )
    return selected


__all__ = ("exact_scene_analysis_frame",)
