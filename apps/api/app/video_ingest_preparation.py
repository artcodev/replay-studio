from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from .config import get_settings
from .project_identifiers import stable_identifier
from .sample import make_video_scene
from .segment_layout import propose_segment_layout
from .video_ffmpeg import (
    create_browser_proxy,
    create_poster,
    detect_shots,
    probe_video,
    require_ffmpeg,
    sample_detector_frames,
)
from .video_media_paths import asset_directory, video_generation_directory
from .video_processing_contract import (
    PreparedVideoGeneration,
    VideoProcessingCancelled,
    VideoProcessingError,
)
from .video_segment_planning import build_recommended_segment_scenes
from .video_store import video_store


def prepare_video_generation(
    asset_id: str,
    title: str | None = None,
    *,
    claim_check: Callable[[], bool],
    progress_writer: Callable[[dict], bool],
    staging_key: str,
) -> PreparedVideoGeneration | None:
    """Prepare one immutable generation without publishing terminal state."""

    if not callable(claim_check):
        raise VideoProcessingError("Video processing requires a claim check")
    if not callable(progress_writer):
        raise VideoProcessingError("Video processing requires a progress writer")
    if not isinstance(staging_key, str) or staging_key == "direct":
        raise VideoProcessingError("Video processing requires a unique staging key")

    def checkpoint() -> None:
        if not claim_check():
            raise VideoProcessingCancelled("Video processing claim was fenced")

    def update(**values) -> dict:
        checkpoint()
        if not progress_writer(dict(values)):
            raise VideoProcessingCancelled("Video processing claim was fenced")
        asset.update(values)
        asset["status"] = "processing"
        return dict(asset)

    checkpoint()
    directory = asset_directory(asset_id)
    loaded_asset = video_store.get(asset_id)
    checkpoint()
    if loaded_asset is None:
        return None
    asset = dict(loaded_asset)
    if asset.get("status") == "ready" and asset.get("scene_id"):
        return None
    source = directory / asset["filename"]
    generation = video_generation_directory(asset_id, staging_key)
    checkpoint()
    try:
        generation.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise VideoProcessingError(
            "Video processing staging key has already been used"
        ) from exc

    try:
        checkpoint()
        require_ffmpeg()
        checkpoint()
        update(stage="Probing video", progress=8)
        checkpoint()
        metadata = probe_video(source)
        if metadata["duration"] <= 0:
            raise VideoProcessingError("Could not determine clip duration")
        if metadata["duration"] > get_settings().max_video_duration:
            raise VideoProcessingError(
                f"Clip is {metadata['duration']:.1f}s; maximum is "
                f"{get_settings().max_video_duration:.0f}s"
            )
        checkpoint()
        update(stage="Creating browser proxy", progress=22, **metadata)

        proxy = generation / "proxy.mp4"
        checkpoint()
        create_browser_proxy(source, proxy)
        checkpoint()
        update(stage="Detecting camera cuts", progress=52)
        segments = detect_shots(source, metadata["duration"])
        checkpoint()
        update(stage="Generating poster", progress=60)
        create_poster(
            source,
            generation / "poster.jpg",
            at_seconds=min(1.0, metadata["duration"] * 0.25),
        )

        checkpoint()
        update(stage="Sampling detector frames", progress=74)
        frames = generation / "frames"
        checkpoint()
        frames.mkdir(exist_ok=True)
        analysis_fps = min(get_settings().analysis_frame_rate, metadata["fps"])
        sample_detector_frames(
            source,
            frames / "frame_%05d.jpg",
            fps=analysis_fps,
        )
        checkpoint()
        frame_count = len(list(frames.glob("frame_*.jpg")))
        checkpoint()
        update(stage="Grouping replay angles", progress=86, frame_count=frame_count)
        segment_layout = propose_segment_layout(
            proxy,
            segments,
            metadata["duration"],
        )
        checkpoint()
        update(stage="Creating reconstruction scene", progress=91)

        scene = make_video_scene(
            scene_id=stable_identifier("video", asset_id, length=16),
            title=title or Path(asset["original_name"]).stem,
            duration=metadata["duration"],
            video_asset={
                "id": asset_id,
                "filename": asset["original_name"],
                "fps": metadata["fps"],
                "analysisFps": analysis_fps,
                "frameCount": frame_count,
                "generationKey": staging_key,
                "processingState": "frames-ready",
                "segments": segments,
                "segmentLayout": segment_layout,
            },
        )
        checkpoint()
        update(stage="Ranking reconstruction moments", progress=96)
        checkpoint()
        children = build_recommended_segment_scenes(scene, segments)
        recommended_label = "moment" if len(children) == 1 else "moments"
        final_stage = (
            f"Ready for reconstruction · {len(children)} recommended "
            f"{recommended_label}"
            if children
            else "Ready for reconstruction"
        )
        checkpoint()
        update(stage=final_stage, progress=99, frame_count=frame_count)
        (generation / "manifest.json").write_text(
            json.dumps(
                {
                    "assetId": asset_id,
                    "generationKey": staging_key,
                    "frameCount": frame_count,
                },
                sort_keys=True,
            )
            + "\n"
        )
        checkpoint()
        asset["frame_count"] = frame_count
        return PreparedVideoGeneration(
            asset=dict(asset),
            root_scene=scene,
            child_scenes=children,
            segments=segments,
            generation_key=staging_key,
            generation_directory=generation,
            stage=final_stage,
        )
    except VideoProcessingCancelled:
        # This generation was created by this attempt and never published:
        # a cancelled or failed attempt must not strand proxy/frames forever
        # (every retry claims a fresh staging key, so nothing reuses these).
        shutil.rmtree(generation, ignore_errors=True)
        raise
    except (VideoProcessingError, OSError, ValueError, json.JSONDecodeError) as exc:
        shutil.rmtree(generation, ignore_errors=True)
        raise VideoProcessingError(str(exc)) from exc
