from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class BallWorkerSettings:
    max_frame_bytes: int
    max_frame_pixels: int
    max_batch_frames: int
    max_candidates_default: int
    candidate_radius: float

    @classmethod
    def from_environment(cls) -> "BallWorkerSettings":
        return cls(
            max_frame_bytes=max(
                1,
                int(
                    os.environ.get(
                        "WASB_MAX_FRAME_BYTES", str(32 * 1024 * 1024)
                    )
                ),
            ),
            max_frame_pixels=max(
                1, int(os.environ.get("WASB_MAX_FRAME_PIXELS", "16000000"))
            ),
            max_batch_frames=max(
                1, int(os.environ.get("WASB_MAX_BATCH_FRAMES", "96"))
            ),
            max_candidates_default=max(
                1, int(os.environ.get("WASB_MAX_CANDIDATES", "12"))
            ),
            candidate_radius=max(
                0.1, float(os.environ.get("WASB_CANDIDATE_RADIUS", "4"))
            ),
        )

