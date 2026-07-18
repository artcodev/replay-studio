"""Application orchestration for temporal ball trajectory resolution.

Detection, candidate normalization, numerical path solving, and editor
payload materialization have separate owners. This module only composes them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .ball_tracking_candidates import normalize_ball_frames
from .ball_tracking_contract import (
    BallFrameInput,
    BallTrackingConfig,
    BallTrajectoryResolution,
    DEFAULT_BALL_TRACKING_CONFIG,
    MotionCoordinateSelector,
    PositionProjector,
)
from .ball_tracking_solver import solve_ball_path
from .ball_trajectory_materialization import materialize_ball_trajectory


def resolve_ball_trajectory(
    ball_frames: Sequence[BallFrameInput],
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
    *,
    config: BallTrackingConfig = DEFAULT_BALL_TRACKING_CONFIG,
    coordinate_selector: MotionCoordinateSelector | None = None,
    projector: PositionProjector | None = None,
) -> BallTrajectoryResolution:
    """Resolve per-frame detections into one fail-closed temporal trajectory.

    Reconstruction supplies ``(detections, time)`` tuples; the detection cache
    supplies mappings such as ``{"t": 1.2, "candidates": [...]}``. Optional
    calibrated pitch coordinates enable metric physical constraints, while
    stabilized image coordinates support association across camera pans.
    """

    normalized = normalize_ball_frames(ball_frames, config, coordinate_selector)
    solution = solve_ball_path(normalized.frames, config)
    return materialize_ball_trajectory(
        normalized,
        solution,
        frame_size,
        pitch,
        config,
        projector,
    )


__all__ = ["resolve_ball_trajectory"]
