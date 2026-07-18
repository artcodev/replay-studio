from __future__ import annotations

"""Application command for publishing an edited scene ball trajectory."""

from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_ball_artifacts import publish_ball_trajectory_artifact
from .reconstruction_ball_trajectory import (
    edit_scene_ball_trajectory,
)
from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


def set_scene_ball_trajectory(
    scene: dict,
    mode: str,
    keyframes: list[dict] | None = None,
) -> dict:
    hydrate_scene_reconstruction(scene, names=("ballTrajectory",))
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before editing the ball trajectory"
        )
    edit_scene_ball_trajectory(scene, mode, keyframes)
    publish_ball_trajectory_artifact(scene)
    return scenes.put(scene)


__all__ = ("set_scene_ball_trajectory",)
