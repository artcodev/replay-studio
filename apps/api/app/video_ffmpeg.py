from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .video_processing_contract import VideoProcessingError


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise VideoProcessingError("ffmpeg is not installed")


def run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (
            result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "FFmpeg command failed"
        )
        raise VideoProcessingError(message)
    return result


def probe_video(source: Path) -> dict:
    if not shutil.which("ffprobe"):
        raise VideoProcessingError("ffprobe is not installed")
    result = run_media_command(
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
    stream = next(
        (
            item
            for item in payload.get("streams", [])
            if item.get("codec_type") == "video"
        ),
        None,
    )
    if stream is None:
        raise VideoProcessingError("The uploaded file has no video stream")
    duration = float(
        payload.get("format", {}).get("duration") or stream.get("duration") or 0
    )
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    numerator, denominator = (rate.split("/", 1) + ["1"])[:2]
    fps = float(numerator) / max(1.0, float(denominator))
    return {
        "duration": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
    }


def detect_shots(
    source: Path,
    duration: float,
    threshold: float = 0.12,
) -> list[dict]:
    result = run_media_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(source),
            "-vf",
            f"select=gt(scene\\,{threshold}),showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    raw_cuts = [
        float(value)
        for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)
    ]
    cuts: list[float] = []
    for value in raw_cuts:
        if value <= 0.25 or value >= duration - 0.25:
            continue
        if not cuts or value - cuts[-1] > 0.15:
            cuts.append(value)

    boundaries = [0.0, *cuts, duration]
    segments: list[dict] = []
    for index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:]), start=1
    ):
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


def create_browser_proxy(source: Path, destination: Path) -> None:
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def create_poster(source: Path, destination: Path, *, at_seconds: float) -> None:
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:-2",
            str(destination),
        ]
    )


def sample_detector_frames(
    source: Path,
    destination_pattern: Path,
    *,
    fps: float,
) -> None:
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps={fps:g},scale=1280:-2",
            "-q:v",
            "3",
            str(destination_pattern),
        ]
    )
