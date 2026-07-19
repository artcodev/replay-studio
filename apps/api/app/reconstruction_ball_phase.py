from __future__ import annotations

from dataclasses import dataclass

from .ball_tracking import resolve_ball_trajectory
from .ball_tracking_contract import BallTrackingConfig
from .config import get_settings
from .reconstruction_ball_projection_status import ball_world_projection_status
from .reconstruction_progress import ReconstructionProgress


@dataclass(frozen=True)
class BallTrajectoryPhaseResult:
    keyframes: list[dict]
    diagnostics: dict


def resolve_ball_phase(
    scene: dict,
    ball_frames: list[tuple[list[dict], float]],
    frame_size: tuple[int, int],
    coordinate_mode: str,
    player_track_count: int,
    progress: ReconstructionProgress,
) -> BallTrajectoryPhaseResult:
    if coordinate_mode == "unavailable":
        # The solver output would be discarded anyway: without metric
        # calibration no world-space trajectory is ever published. Record
        # the explicit skip instead of paying for a thrown-away solve.
        ball = []
        ball_tracking_diagnostics = {
            "trajectoryMode": "automatic",
            "solverSkipped": True,
            "solverSkippedReason": "coordinate-mode-unavailable",
            "worldProjectionStatus": ball_world_projection_status(
                coordinate_mode,
                ball,
            ),
            "detectorFrameCount": len(ball_frames),
            "detectorCandidateFrameCount": sum(
                bool(items) for items, _ in ball_frames
            ),
        }
        progress.update(
            "projection",
            5,
            "3D trajectories ready",
            (
                f"Accepted {player_track_count} player tracks; ball world "
                "trajectory requires metric calibration."
            ),
            91,
            97,
            completed=2,
            total=2,
            eta_padding=2.0,
        )
        return BallTrajectoryPhaseResult(
            keyframes=ball,
            diagnostics=ball_tracking_diagnostics,
        )

    resolver_frames = [
        (
            [
                candidate
                for candidate in candidates
                if candidate.get("pitchX") is not None
                and candidate.get("pitchZ") is not None
            ],
            time,
        )
        for candidates, time in ball_frames
    ]
    ball_resolution = resolve_ball_trajectory(
        resolver_frames,
        frame_size,
        scene["payload"]["pitch"],
        config=BallTrackingConfig(
            top_k_per_frame=min(8, get_settings().ball_detection_max_candidates),
            max_interpolation_gap_seconds=0.8,
            max_ball_speed_metres_per_second=55.0,
        ),
    )
    ball = ball_resolution.keyframes
    ball_tracking_diagnostics = {
        **ball_resolution.diagnostics,
        "worldProjectionStatus": ball_world_projection_status(
            coordinate_mode,
            ball,
        ),
        "detectorFrameCount": len(ball_frames),
        "detectorCandidateFrameCount": sum(bool(items) for items, _ in ball_frames),
    }
    progress.update(
        "projection",
        5,
        "3D trajectories ready",
        f"Accepted {player_track_count} player tracks and {len(ball)} ball samples.",
        91,
        97,
        completed=2,
        total=2,
        eta_padding=2.0,
    )
    return BallTrajectoryPhaseResult(
        keyframes=ball,
        diagnostics=ball_tracking_diagnostics,
    )
