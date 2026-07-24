from __future__ import annotations

"""Persist one reversible frame exclusion as a scene analysis input."""

from datetime import UTC, datetime

from .reconstruction_calibration_edit_session import (
    clear_pending_calibration_edit_session,
)
from .reconstruction_calibration_overrides import (
    manual_pitch_calibration_overrides,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import native_frame_paths, source_frame_index
from .scene_frame_exclusions import scene_frame_exclusions
from .scene_repository import scenes


def set_scene_frame_excluded(
    scene: dict,
    requested_source_frame_index: int,
    *,
    excluded: bool,
) -> dict:
    """Exclude or restore one exact source frame for all analysis pipelines."""

    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for calibration or reconstruction to finish before excluding frames"
        )

    available = native_frame_paths(scene)
    frame_by_index = {
        source_frame_index(path): float(scene_time)
        for path, scene_time in available
    }
    if requested_source_frame_index not in frame_by_index:
        raise ReconstructionError(
            "The requested analysis frame is outside this scene segment"
        )

    current = scene_frame_exclusions(scene)
    current_by_index = {
        int(item["sourceFrameIndex"]): item
        for item in current
    }
    if excluded:
        current_by_index[requested_source_frame_index] = {
            "sourceFrameIndex": requested_source_frame_index,
            "sceneTime": frame_by_index[requested_source_frame_index],
            "excludedAt": datetime.now(UTC).isoformat(),
        }
        if len(current_by_index) >= len(available):
            raise ReconstructionError(
                "A scene segment must retain at least one analysis frame"
            )
    else:
        current_by_index.pop(requested_source_frame_index, None)

    ordered = sorted(
        current_by_index.values(),
        key=lambda item: int(item["sourceFrameIndex"]),
    )
    if ordered:
        video["frameExclusions"] = ordered
    else:
        video.pop("frameExclusions", None)

    if excluded:
        # A manual calibration for an excluded image must not be silently
        # attached to the nearest remaining sample on the next run.
        retained_overrides = [
            item
            for item in manual_pitch_calibration_overrides(reconstruction)
            if int(item.get("sourceFrameIndex") or -1)
            != requested_source_frame_index
        ]
        if retained_overrides:
            reconstruction["pitchCalibrationOverrides"] = retained_overrides
        else:
            reconstruction.pop("pitchCalibrationOverrides", None)

    # Sample indexes shift after either exclusion or restoration. Source-frame
    # keyed overrides remain valid, but an in-progress edit session does not.
    clear_pending_calibration_edit_session(reconstruction)
    reconstruction.pop("calibrationFallbackConsent", None)
    # Keep the prior artifact and timeline inspectable, but make the gate
    # unambiguously stale until an explicit full recalibration publishes a
    # replacement for the changed frame set.
    reconstruction.pop("calibrationInputFingerprint", None)
    video["reconstruction"] = reconstruction
    return scenes.put(scene)


__all__ = ("set_scene_frame_excluded",)
