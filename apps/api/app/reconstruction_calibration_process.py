from __future__ import annotations

"""Dedicated entry point for one claimed calibration subprocess."""

from typing import Callable, Mapping

from .artifact_store import ArtifactStore
from .reconstruction import (
    _calibrate_scene_in_memory,
    _execute_claimed_scene_process,
)


def calibrate_scene(
    scene: dict,
    *,
    expected_run_id: str,
    expected_input_fingerprint: str,
    expected_lease_owner_id: str,
    progress_listener: Callable[[dict], None] | None = None,
    artifact_store: ArtifactStore | None = None,
    match_snapshot: Mapping[str, object] | None = None,
) -> dict:
    """Execute a claimed calibration job; Reconstruction is not reachable."""

    return _execute_claimed_scene_process(
        scene,
        in_memory_runner=_calibrate_scene_in_memory,
        expected_run_id=expected_run_id,
        expected_input_fingerprint=expected_input_fingerprint,
        expected_lease_owner_id=expected_lease_owner_id,
        progress_listener=progress_listener,
        artifact_store=artifact_store,
        match_snapshot=match_snapshot,
    )


__all__ = ("calibrate_scene",)
