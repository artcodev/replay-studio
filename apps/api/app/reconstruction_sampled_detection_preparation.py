from __future__ import annotations

"""Resolve sampled inputs and construct detector/cache runtime dependencies."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import app.person_base_detection_cache as person_base_detection_cache

from .ball_detection_contract import BallDetector
from .config import get_settings
from .person_detection_provider_contract import PersonDetectionProvider
from .person_detection_provider_factory import build_person_detection_provider
from .person_detector_provenance import person_detection_input
from .reconstruction_ball_detector_selection import configured_ball_detectors
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import frame_paths, load_model
from .reconstruction_progress import ReconstructionProgress


@dataclass(frozen=True)
class SampledDetectionRuntime:
    model: Any
    person_provider: PersonDetectionProvider
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
        8,
        completed=0,
        total=1,
    )
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    progress.update(
        "preparing",
        1,
        "Inputs ready",
        f"Found {len(frames)} frames at "
        f"{float(reconstruction.get('samplingFrameRate') or video.get('fps') or 0):g} "
        f"FPS (source {float(video.get('fps') or 0):g} FPS; materialized "
        f"{float(video.get('analysisFps') or 0):g} FPS).",
        0,
        8,
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
        8,
        38,
        completed=0,
        total=len(frames),
    )
    model = load_model(model_name)
    person_provider = build_person_detection_provider(model_name, model)
    provider_info = person_provider.info()
    detector_input = person_detection_input(
        model_name,
        model,
        provider_info=provider_info,
    )
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
        person_provider=person_provider,
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
