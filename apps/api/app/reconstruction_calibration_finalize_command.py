from __future__ import annotations

"""Queue explicit finalization of staged frame-calibration corrections."""

from typing import Mapping

from .project_match_persistence_contract import MatchSnapshotDocument
from .reconstruction_calibration_edit_session import (
    pending_calibration_edit_session,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_queue import queue_reconstruction


def finalize_scene_pitch_calibration_drafts(
    scene: dict,
    *,
    match_snapshot: MatchSnapshotDocument | None,
) -> dict:
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for the current calibration process before finalizing edits"
        )
    session = pending_calibration_edit_session(reconstruction)
    if session is None or not session.get("editedSampleIndices"):
        raise ReconstructionError("There are no staged calibration corrections")
    provenance = reconstruction.get("calibrationProvenance")
    if not isinstance(provenance, Mapping):
        raise ReconstructionError(
            "The base calibration is missing; run full calibration"
        )
    artifact = provenance.get("artifact")
    if (
        str(provenance.get("dataFingerprint") or "")
        != str(session.get("baseDataFingerprint") or "")
        or not isinstance(artifact, Mapping)
        or str(artifact.get("sha256") or "")
        != str(session.get("baseArtifactSha256") or "")
    ):
        raise ReconstructionError(
            "The published calibration changed after these drafts were created; reopen the timeline"
        )
    return queue_reconstruction(
        scene,
        mode="calibrate",
        calibration_trigger="manual-draft-finalize",
        match_snapshot=match_snapshot,
    )


__all__ = ("finalize_scene_pitch_calibration_drafts",)
