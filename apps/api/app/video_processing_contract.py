from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .project_identifiers import stable_identifier


class VideoProcessingError(RuntimeError):
    pass


class VideoProcessingCancelled(VideoProcessingError):
    pass


@dataclass(frozen=True)
class PreparedVideoGeneration:
    asset: dict
    root_scene: dict
    child_scenes: list[dict]
    segments: list[dict]
    generation_key: str
    generation_directory: Path
    stage: str

    def validate(self) -> None:
        """Reject publication unless the immutable output set is complete."""

        required = (
            self.generation_directory / "proxy.mp4",
            self.generation_directory / "poster.jpg",
            self.generation_directory / "manifest.json",
        )
        if any(not path.is_file() for path in required):
            raise VideoProcessingError("Derived video generation is incomplete")
        frames = tuple(
            sorted((self.generation_directory / "frames").glob("frame_*.jpg"))
        )
        if len(frames) != int(self.asset.get("frame_count") or 0):
            raise VideoProcessingError(
                "Derived video frame generation is incomplete"
            )


def video_processing_run_id(project_id: str, asset_id: str) -> str:
    return stable_identifier(
        "analysis", project_id, asset_id, "video-processing", length=32
    )
