from __future__ import annotations

"""Persist the user-selected attacking goal without changing field geometry."""

from datetime import UTC, datetime

from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


def set_scene_pitch_side(scene: dict, target_side: str) -> dict:
    if target_side not in {"left", "right"}:
        raise ReconstructionError("Pitch side must be left or right")
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before changing pitch side"
        )
    current_orientation = reconstruction.get("pitchOrientation") or {}
    reconstruction["pitchOrientation"] = {
        **current_orientation,
        "attackingGoal": target_side,
        "attackingGoalSource": "manual",
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    video["reconstruction"] = reconstruction
    return scenes.put(scene)
