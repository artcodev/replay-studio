from __future__ import annotations

"""Resolve sampled inputs and construct detector/cache runtime dependencies."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import app.person_base_detection_cache as person_base_detection_cache

from .ball_detection_contract import BallDetector
from .config import get_settings
from .person_detector_provenance import person_detection_input
from .reconstruction_ball_detector_selection import configured_ball_detectors
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import frame_paths, load_model
from .reconstruction_progress import ReconstructionProgress


@dataclass(frozen=True)
class SampledDetectionRuntime:
    model: Any
    person_detector_input: dict
    person_cache_diagnostics: dict
    person_cache_directory: Path
    ball_detector: BallDetector
    ball_fallback_detector: BallDetector | None


def load_sampled_frames(
    scene: dict,
    progress: ReconstructionProgress,
) -> list[tuple[Path, float]]:
    progress.update(
        "preparing",
        1,
        "Preparing sampled frames",
        "Reading the scene range and checking extracted images.",
        0,
        4,
        completed=0,
        total=1,
    )
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    progress.update(
        "preparing",
        1,
        "Inputs ready",
        f"Found {len(frames)} sampled frames for analysis.",
        0,
        4,
        completed=1,
        total=1,
        eta_padding=max(8.0, len(frames) * 0.35),
    )
    return frames


def prepare_sampled_detectors(
    scene: dict,
    frames: list[tuple[Path, float]],
    *,
    model_name: str,
    ball_backend: str,
    ball_detection_input: Mapping,
    progress: ReconstructionProgress,
) -> SampledDetectionRuntime:
    progress.update(
        "detection",
        3,
        "Loading object detectors",
        f"Preparing {model_name} for people and {ball_backend} for dense ball inference.",
        62,
        84,
        completed=0,
        total=len(frames),
    )
    model = load_model(model_name)
    detector_input = person_detection_input(model_name, model)
    cache_diagnostics = person_base_detection_cache.base_detection_cache_diagnostics(
        len(frames),
        detector_input,
    )
    cache_directory = (
        Path(get_settings().media_root).resolve()
        / str(scene["payload"]["videoAsset"]["id"])
    )
    ball_detector, ball_fallback = configured_ball_detectors(
        model,
        ball_backend,
        ball_detection_input,
    )
    return SampledDetectionRuntime(
        model=model,
        person_detector_input=detector_input,
        person_cache_diagnostics=cache_diagnostics,
        person_cache_directory=cache_directory,
        ball_detector=ball_detector,
        ball_fallback_detector=ball_fallback,
    )


__all__ = [
    "SampledDetectionRuntime",
    "load_sampled_frames",
    "prepare_sampled_detectors",
]
