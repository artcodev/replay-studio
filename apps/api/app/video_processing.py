from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from .config import get_settings
from .project_match import copy_project_match_metadata
from .reconstruction import ReconstructionError, reconstruct_scene
from .sample import make_video_scene
from .segment_layout import propose_segment_layout
from .store import scene_store
from .video_store import video_store


class VideoProcessingError(RuntimeError):
    pass


def asset_directory(asset_id: str) -> Path:
    return Path(get_settings().media_root).resolve() / asset_id


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "FFmpeg command failed"
        raise VideoProcessingError(message)
    return result


def probe_video(source: Path) -> dict:
    if not shutil.which("ffprobe"):
        raise VideoProcessingError("ffprobe is not installed")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(source),
        ]
    )
    payload = json.loads(result.stdout)
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None)
    if stream is None:
        raise VideoProcessingError("The uploaded file has no video stream")
    duration = float(payload.get("format", {}).get("duration") or stream.get("duration") or 0)
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    numerator, denominator = (rate.split("/", 1) + ["1"])[:2]
    fps = float(numerator) / max(1.0, float(denominator))
    return {
        "duration": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
    }


def detect_shots(source: Path, duration: float, threshold: float = 0.12) -> list[dict]:
    result = _run(
        [
            "ffmpeg", "-hide_banner", "-i", str(source),
            "-vf", f"select=gt(scene\\,{threshold}),showinfo",
            "-an", "-f", "null", "-",
        ]
    )
    raw_cuts = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    cuts: list[float] = []
    for value in raw_cuts:
        if value <= 0.25 or value >= duration - 0.25:
            continue
        if not cuts or value - cuts[-1] > 0.15:
            cuts.append(value)

    boundaries = [0.0, *cuts, duration]
    segments: list[dict] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        segment_duration = end - start
        if segment_duration < 2.5:
            continue
        segments.append(
            {
                "id": f"shot-{index:02d}",
                "label": f"Shot {index:02d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(segment_duration, 3),
                "score": round(min(1.0, segment_duration / 7.0), 3),
            }
        )
    return segments


def rank_reconstruction_shots(segments: list[dict], limit: int = 5) -> list[dict]:
    eligible = [segment for segment in segments if segment["duration"] >= 4.0]
    ranked = sorted(eligible, key=lambda item: (item["score"], item["duration"]), reverse=True)[:limit]
    recommended_ids = {item["id"] for item in ranked}
    for segment in segments:
        segment["recommended"] = segment["id"] in recommended_ids
    return ranked


def materialize_segment_scene(parent: dict, segment: dict) -> dict:
    existing = scene_store.find_segment_scene(parent["id"], segment["id"])
    if existing:
        segment["sceneId"] = existing["id"]
        return existing

    video = parent["payload"]["videoAsset"]
    scene_id = f"moment-{video['id'].removeprefix('asset-')}-{segment['id']}"
    child_video = {
        **video,
        "sourceStart": segment["start"],
        "sourceEnd": segment["end"],
        "parentSceneId": parent["id"],
        "selectedSegmentId": segment["id"],
        "segments": [],
    }
    child = make_video_scene(
        scene_id=scene_id,
        title=f"{parent['title']} · {segment['label']}",
        duration=segment["duration"],
        video_asset=child_video,
    )
    copy_project_match_metadata(
        child,
        parent,
        project_scene_id=parent["id"],
        inherited=True,
    )
    scene_store.put(child)
    segment["sceneId"] = child["id"]
    return child


def materialize_recommended_scenes(parent: dict, segments: list[dict]) -> list[dict]:
    ranked = rank_reconstruction_shots(segments)
    children = [materialize_segment_scene(parent, segment) for segment in ranked]
    if children:
        parent["payload"]["videoAsset"]["primarySceneId"] = children[0]["id"]
    parent["payload"]["videoAsset"]["segments"] = segments
    scene_store.put(parent)
    return children


def process_video_asset(asset_id: str, title: str | None = None) -> None:
    directory = asset_directory(asset_id)
    asset = video_store.get(asset_id)
    if asset is None:
        return
    source = directory / asset["filename"]
    try:
        if not shutil.which("ffmpeg"):
            raise VideoProcessingError("ffmpeg is not installed")
        video_store.update(asset_id, status="processing", stage="Probing video", progress=8)
        metadata = probe_video(source)
        if metadata["duration"] <= 0:
            raise VideoProcessingError("Could not determine clip duration")
        if metadata["duration"] > get_settings().max_video_duration:
            raise VideoProcessingError(
                f"Clip is {metadata['duration']:.1f}s; maximum is {get_settings().max_video_duration:.0f}s"
            )
        video_store.update(asset_id, stage="Creating browser proxy", progress=22, **metadata)

        proxy = directory / "proxy.mp4"
        _run(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale=1280:-2:force_original_aspect_ratio=decrease",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", str(proxy),
            ]
        )
        video_store.update(asset_id, stage="Detecting camera cuts", progress=52)
        segments = detect_shots(source, metadata["duration"])
        video_store.update(asset_id, stage="Generating poster", progress=60)
        poster_time = min(1.0, metadata["duration"] * 0.25)
        _run(
            [
                "ffmpeg", "-y", "-ss", f"{poster_time:.3f}", "-i", str(source),
                "-frames:v", "1", "-vf", "scale=960:-2", str(directory / "poster.jpg"),
            ]
        )

        video_store.update(asset_id, stage="Sampling detector frames", progress=74)
        frames = directory / "frames"
        frames.mkdir(exist_ok=True)
        analysis_fps = min(get_settings().analysis_frame_rate, metadata["fps"])
        _run(
            [
                # Preserve enough source detail for far-side players. The
                # detector already runs at 1280, so storing 960px frames only
                # forced an avoidable upscale before inference.
                "ffmpeg", "-y", "-i", str(source), "-vf", f"fps={analysis_fps:g},scale=1280:-2",
                "-q:v", "3", str(frames / "frame_%05d.jpg"),
            ]
        )
        frame_count = len(list(frames.glob("frame_*.jpg")))
        video_store.update(asset_id, stage="Grouping replay angles", progress=86, frame_count=frame_count)
        segment_layout = propose_segment_layout(proxy, segments, metadata["duration"])
        video_store.update(asset_id, stage="Creating reconstruction scene", progress=91)

        scene_id = f"video-{uuid4().hex[:8]}"
        scene = make_video_scene(
            scene_id=scene_id,
            title=title or Path(asset["original_name"]).stem,
            duration=metadata["duration"],
            video_asset={
                "id": asset_id,
                "filename": asset["original_name"],
                "mediaUrl": f"/api/videos/{asset_id}/media",
                "posterUrl": f"/api/videos/{asset_id}/poster",
                "fps": metadata["fps"],
                "analysisFps": analysis_fps,
                "frameCount": frame_count,
                "processingState": "frames-ready",
                "segments": segments,
                "segmentLayout": segment_layout,
            },
        )
        scene_store.put(scene)
        video_store.update(asset_id, stage="Ranking reconstruction moments", progress=96)
        children = materialize_recommended_scenes(scene, segments)
        ready_stage = "Ready for reconstruction"
        if children:
            video_store.update(asset_id, stage="Reconstructing players and ball", progress=98)
            try:
                reconstructed = reconstruct_scene(children[0])
                track_count = len(reconstructed["payload"]["tracks"])
                ready_stage = f"Ready with {track_count} automatic tracks"
            except ReconstructionError:
                ready_stage = "Ready; automatic reconstruction needs review"
        video_store.update(
            asset_id,
            status="ready",
            stage=ready_stage,
            progress=100,
            scene_id=scene_id,
            frame_count=frame_count,
        )
    except (VideoProcessingError, OSError, ValueError, json.JSONDecodeError) as exc:
        video_store.update(asset_id, status="failed", stage="Processing failed", error=str(exc), progress=100)
