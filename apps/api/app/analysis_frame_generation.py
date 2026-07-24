from __future__ import annotations

"""Prepare immutable source-resolution analysis frames for an existing asset."""

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Callable

from .reconstruction_inputs import resolve_analysis_frame_rate
from .video_ffmpeg import probe_video, require_ffmpeg, sample_detector_frames
from .video_media_paths import (
    asset_directory,
    published_video_directory,
    video_generation_directory,
)
from .video_processing_contract import VideoProcessingCancelled, VideoProcessingError
from .video_store import video_store


@dataclass(frozen=True, slots=True)
class PreparedAnalysisFrameGeneration:
    asset_id: str
    generation_key: str
    generation_directory: Path
    frame_count: int
    source_fps: float
    analysis_fps: float
    analysis_frame_input: dict

    def validate(self) -> None:
        if abs(self.source_fps - self.analysis_fps) > 1e-3:
            raise VideoProcessingError(
                "Analysis-frame generation must retain the native source cadence"
            )
        required = (
            self.generation_directory / "proxy.mp4",
            self.generation_directory / "poster.jpg",
            self.generation_directory / "manifest.json",
        )
        if any(not path.is_file() for path in required):
            raise VideoProcessingError(
                "Source-resolution analysis generation is incomplete"
            )
        frames = tuple(
            sorted((self.generation_directory / "frames").glob("frame_*.jpg"))
        )
        if len(frames) != self.frame_count:
            raise VideoProcessingError(
                "Source-resolution analysis frame count is incomplete"
            )


def prepare_analysis_frame_generation(
    asset_id: str,
    *,
    staging_key: str,
    claim_check: Callable[[], bool],
    progress_writer: Callable[[dict], bool],
) -> PreparedAnalysisFrameGeneration:
    """Create a complete generation while the published one stays readable."""

    def checkpoint() -> None:
        if not claim_check():
            raise VideoProcessingCancelled("Analysis-frame generation was fenced")

    def progress(percent: int, label: str, detail: str) -> None:
        checkpoint()
        if not progress_writer(
            {
                "phase": "analysis-frame-generation",
                "label": label,
                "detail": detail,
                "completed": percent,
                "total": 100,
                "phasePercent": percent,
                "overallPercent": percent,
                "etaSeconds": None,
            }
        ):
            raise VideoProcessingCancelled("Analysis-frame generation was fenced")

    checkpoint()
    asset = video_store.get(asset_id)
    if asset is None:
        raise VideoProcessingError("Video asset was not found")
    source = asset_directory(asset_id) / str(asset["filename"])
    if not source.is_file():
        raise VideoProcessingError("The immutable uploaded source video is missing")
    current_generation = published_video_directory(asset)
    generation = video_generation_directory(asset_id, staging_key)
    try:
        generation.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise VideoProcessingError("Analysis-frame staging key was already used") from exc

    try:
        require_ffmpeg()
        progress(5, "Reading source video", "Verifying the uploaded source pixel grid.")
        metadata = probe_video(source)
        if metadata["duration"] <= 0 or metadata["width"] <= 0 or metadata["height"] <= 0:
            raise VideoProcessingError("Could not resolve source video dimensions")
        analysis_fps = resolve_analysis_frame_rate(float(metadata["fps"]))

        # Presentation media is immutable too. Reuse its bytes inside the new
        # complete generation; only the analysis-frame capability is rebuilt.
        progress(10, "Preparing immutable generation", "Keeping the browser proxy and poster unchanged.")
        for name in ("proxy.mp4", "poster.jpg"):
            published = current_generation / name
            if not published.is_file():
                raise VideoProcessingError(f"Published {name} is missing")
            shutil.copy2(published, generation / name)

        frames = generation / "frames"
        frames.mkdir()
        progress(
            15,
            "Extracting source-resolution frames",
            f"Sampling {analysis_fps:g} FPS at {metadata['width']}×{metadata['height']} without resize.",
        )
        checkpoint()
        sample_detector_frames(
            source,
            frames / "frame_%05d.jpg",
            fps=analysis_fps,
        )
        checkpoint()
        frame_count = len(tuple(frames.glob("frame_*.jpg")))
        if frame_count <= 0:
            raise VideoProcessingError("FFmpeg produced no analysis frames")
        analysis_frame_input = {
            "schemaVersion": 1,
            "source": "uploaded-video",
            "coordinateSpace": "source-video-pixels",
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "resize": "none",
            "format": "jpeg",
            "jpegQscale": 1,
            "chromaSampling": "4:4:4",
            "sourceFps": float(metadata["fps"]),
            "averageFps": float(metadata.get("averageFps") or metadata["fps"]),
            "sourceFrameCount": int(metadata.get("sourceFrameCount") or 0),
        }
        (generation / "manifest.json").write_text(
            json.dumps(
                {
                    "assetId": asset_id,
                    "generationKey": staging_key,
                    "frameCount": frame_count,
                    "sourceFps": float(metadata["fps"]),
                    "analysisFps": analysis_fps,
                    "analysisFrameInput": analysis_frame_input,
                    "generationPurpose": "source-resolution-analysis-frames",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        progress(
            95,
            "Publishing source-resolution frames",
            f"Prepared {frame_count} immutable analysis frames.",
        )
        return PreparedAnalysisFrameGeneration(
            asset_id=asset_id,
            generation_key=staging_key,
            generation_directory=generation,
            frame_count=frame_count,
            source_fps=float(metadata["fps"]),
            analysis_fps=analysis_fps,
            analysis_frame_input=analysis_frame_input,
        )
    except VideoProcessingCancelled:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    except (OSError, ValueError, VideoProcessingError) as exc:
        shutil.rmtree(generation, ignore_errors=True)
        raise VideoProcessingError(str(exc)) from exc


__all__ = (
    "PreparedAnalysisFrameGeneration",
    "prepare_analysis_frame_generation",
)
