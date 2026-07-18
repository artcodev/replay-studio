from __future__ import annotations

"""Persist one validated manual calibration and queue an atomic rebuild."""

from datetime import UTC, datetime

import numpy as np

from .pitch_calibration_contract import pitch_side
from .project_match_persistence_contract import MatchSnapshotDocument
from .reconstruction_calibration_frame_context import calibration_frame_context
from .reconstruction_calibration_manual_preview import (
    preview_scene_pitch_calibration,
)
from .reconstruction_calibration_overrides import (
    upsert_manual_pitch_calibration_override,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import (
    frame_paths,
    source_frame_index as parse_source_frame_index,
)
from .reconstruction_queue import queue_reconstruction
from .scene_repository import scenes


def apply_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
    *,
    match_snapshot: MatchSnapshotDocument | None = None,
) -> dict:
    draft = preview_scene_pitch_calibration(scene, scene_time, preset, anchors)
    if draft.get("quality") == "poor" or draft.get("alignmentMetrics") is None:
        raise ReconstructionError(
            "Manual calibration does not align with enough pitch markings; refine the anchors before applying."
        )
    _, _, _, camera_transform = calibration_frame_context(
        scene,
        draft["sceneTime"],
    )
    current_to_pitch = np.asarray(draft["imageToPitch"], dtype=np.float64)
    try:
        stabilized_to_pitch = current_to_pitch @ np.linalg.inv(camera_transform)
    except np.linalg.LinAlgError as exc:
        raise ReconstructionError(
            "Camera motion transform could not be inverted"
        ) from exc
    stabilized_to_pitch /= stabilized_to_pitch[2, 2]
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    resolved_anchors = draft.get("anchors") or anchors
    resolved_preset = draft.get("preset") or preset
    sampled_frames = frame_paths(scene)
    sampled_index = int(draft["frameIndex"]) - 1
    source_frame_index = (
        parse_source_frame_index(sampled_frames[sampled_index][0])
        if sampled_frames and 0 <= sampled_index < len(sampled_frames)
        else None
    )
    override = {
        "id": (
            f"manual-frame-{source_frame_index}"
            if source_frame_index is not None
            else f"manual-time-{float(draft['sceneTime']):.3f}"
        ),
        "status": "ready" if draft.get("quality") == "good" else "review",
        "validationStatus": draft.get("quality") or "poor",
        "method": "manual-pitch-anchors",
        "confidence": draft["confidence"],
        "supportedLines": len(resolved_anchors),
        "matchedCurves": 1 if resolved_preset == "center-circle" else 0,
        "meanLineScore": 0.0,
        "preset": resolved_preset,
        "pitchSide": pitch_side(resolved_preset),
        "sceneTime": draft["sceneTime"],
        "frameIndex": draft["frameIndex"],
        "sampleIndex": sampled_index,
        "alignmentError": draft["alignmentError"],
        "alignmentMetrics": draft.get("alignmentMetrics"),
        "horizon": draft.get("horizon"),
        "sourceFrameIndex": source_frame_index,
        "anchors": resolved_anchors,
        "coordinateSpace": "stabilized-reference-image",
        "imageToPitch": [
            [round(float(value), 10) for value in row]
            for row in stabilized_to_pitch
        ],
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    upsert_manual_pitch_calibration_override(reconstruction, override)
    resolved_side = pitch_side(resolved_preset)
    if resolved_side:
        current_orientation = reconstruction.get("pitchOrientation") or {}
        reconstruction["pitchOrientation"] = {
            **current_orientation,
            "visiblePitchSide": resolved_side,
            "visiblePitchSideSource": "manual-calibration",
            "attackingGoal": current_orientation.get("attackingGoal") or "unknown",
            "attackingGoalSource": current_orientation.get("attackingGoalSource")
            or "unknown",
            "updatedAt": datetime.now(UTC).isoformat(),
        }
    video["reconstruction"] = reconstruction
    persisted_scene = scenes.put(scene)
    return queue_reconstruction(persisted_scene, match_snapshot=match_snapshot)
