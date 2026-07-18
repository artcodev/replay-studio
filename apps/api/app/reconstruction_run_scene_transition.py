from __future__ import annotations

"""Pure SceneDocument transitions for reconstruction control state."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .reconstruction_run_contract import (
    ReconstructionRunFence,
    reconstruction_state,
    scene_matches_fence,
)
from .scene_document import next_scene_payload, scene_revision


@dataclass(frozen=True, slots=True)
class ClaimedScene:
    payload: dict[str, Any]
    reconstruction: dict[str, Any]
    revision: int


def transition_scene_to_processing(
    scene: dict[str, Any],
    fence: ReconstructionRunFence,
    *,
    current_time: float,
) -> ClaimedScene | None:
    if not scene_matches_fence(
        scene,
        fence,
        statuses={"queued", "processing"},
    ):
        return None

    claimed = deepcopy(scene)
    video = claimed["payload"]["videoAsset"]
    reconstruction = video["reconstruction"]
    reconstruction.update(
        {
            "status": "processing",
            "processingStatus": "processing",
            "startedAt": (
                reconstruction.get("startedAt")
                or datetime.fromtimestamp(current_time, UTC).isoformat()
            ),
            "error": None,
        }
    )
    video["processingState"] = "reconstructing"
    revision = scene_revision(scene) + 1
    payload = next_scene_payload(claimed, revision)
    return ClaimedScene(
        payload=payload,
        reconstruction=reconstruction_state(payload),
        revision=revision,
    )
