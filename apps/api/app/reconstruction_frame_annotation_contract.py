from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FrameAnnotationTarget:
    path: Path
    scene_time: float
    frame_index: int
    x: float
    y: float
    width: float
    height: float
