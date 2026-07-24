from __future__ import annotations

"""Validate and persist one manual frame correction as a staged draft."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Mapping

import numpy as np

from .pitch_calibration_contract import pitch_side
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_calibration_edit_session import (
    pending_calibration_edit_session,
    register_pending_calibration_edit,
)
from .reconstruction_calibration_fingerprint import calibration_input_fingerprint
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
from .scene_repository import scenes


def _manual_quality_warning_detail(draft: Mapping) -> str:
    metrics = draft.get("alignmentMetrics")
    if not isinstance(metrics, Mapping):
        return "automatic line-mask QA returned poor quality"
    values: list[str] = []
    for field, label in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "F1"),
    ):
        value = metrics.get(field)
        if isinstance(value, (int, float)):
            values.append(f"{label} {float(value):.3f}")
    residual = metrics.get("residualP95")
    if isinstance(residual, (int, float)):
        values.append(f"residual p95 {float(residual):.1f}px")
    return ", ".join(values) or "automatic line-mask QA returned poor quality"


def _source_evidence(
    scene: dict,
    sample_index: int,
) -> dict:
    hydrated = deepcopy(scene)
    try:
        hydrate_scene_reconstruction(
            hydrated,
            names=("calibrationFrames",),
        )
    except ReconstructionError:
        return {}
    reconstruction = (
        hydrated.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    calibration = reconstruction.get("calibration") or {}
    frames = calibration.get("frameEvidence") or []
    if not (0 <= sample_index < len(frames)) or not isinstance(
        frames[sample_index], Mapping
    ):
        return {}
    evidence = frames[sample_index]
    return {
        "backend": evidence.get("backend"),
        "sourceFrameIndex": evidence.get("sourceFrameIndex"),
        "sceneTime": evidence.get("sceneTime"),
        "solutionStatus": evidence.get("solutionStatus"),
        "projectionSource": evidence.get("projectionSource"),
        "confidence": evidence.get("confidence"),
        "imageToPitch": deepcopy(evidence.get("imageToPitch")),
        "keypoints": deepcopy(
            evidence.get("keypoints") or evidence.get("rawKeypoints") or []
        ),
        "rawLines": deepcopy(evidence.get("rawLines") or []),
        "pnlcalibAttempts": deepcopy(evidence.get("pnlcalibAttempts")),
        "cameraMotion": deepcopy(evidence.get("cameraMotion")),
    }


def build_manual_calibration_override(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
    *,
    draft_source: str = "manual",
    source_evidence: Mapping | None = None,
    camera_transform: np.ndarray | None = None,
    accept_quality_warning: bool = False,
) -> tuple[dict, list[list[float]]]:
    """Build and upsert a validated stabilized-reference manual observation."""

    draft = preview_scene_pitch_calibration(scene, scene_time, preset, anchors)
    quality_warning = draft.get("quality") == "poor"
    if quality_warning and not accept_quality_warning:
        raise ReconstructionError(
            "Automatic pitch-marking QA disagrees with this manual calibration "
            f"({_manual_quality_warning_detail(draft)}). "
            "Refine the anchors, or explicitly save it with the QA warning."
        )
    if camera_transform is None:
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
        "qualityWarningAccepted": bool(
            quality_warning and accept_quality_warning
        ),
        "qualityWarning": (
            _manual_quality_warning_detail(draft)
            if quality_warning
            else None
        ),
        "method": "manual-pitch-anchors",
        "draftSource": draft_source,
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
        "sourceEvidence": deepcopy(dict(source_evidence or {})),
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
    frame_local_image_to_pitch = [
        [round(float(value), 10) for value in row] for row in current_to_pitch
    ]
    return override, frame_local_image_to_pitch


def save_scene_pitch_calibration_draft(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
    *,
    draft_source: str,
    accept_quality_warning: bool = False,
) -> dict:
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for the current calibration process before editing frames"
        )
    provenance = reconstruction.get("calibrationProvenance")
    if not isinstance(provenance, Mapping):
        raise ReconstructionError(
            "Run full calibration once before staging individual corrections"
        )
    if pending_calibration_edit_session(reconstruction) is None and str(
        provenance.get("calibrationInputFingerprint") or ""
    ) != calibration_input_fingerprint(scene):
        raise ReconstructionError(
            "The published calibration is stale for the current scene inputs; "
            "run full calibration before editing frames"
        )
    sampled_frames = frame_paths(scene)
    if not sampled_frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    frame_index = min(
        range(len(sampled_frames)),
        key=lambda index: abs(sampled_frames[index][1] - scene_time),
    )
    evidence = _source_evidence(scene, frame_index)
    expected_source_index = parse_source_frame_index(
        sampled_frames[frame_index][0]
    )
    if int(evidence.get("sourceFrameIndex") or -1) != expected_source_index:
        raise ReconstructionError(
            "The published calibration evidence belongs to another frame generation"
        )
    camera_motion = evidence.get("cameraMotion")
    current_to_reference = (
        camera_motion.get("currentToReference")
        if isinstance(camera_motion, Mapping)
        else None
    )
    camera_transform = np.asarray(current_to_reference, dtype=np.float64)
    if (
        camera_transform.shape != (3, 3)
        or not np.isfinite(camera_transform).all()
        or abs(float(np.linalg.det(camera_transform))) < 1e-10
    ):
        raise ReconstructionError(
            "The published frame has no valid camera transform; run full calibration"
        )
    override, _ = build_manual_calibration_override(
        scene,
        scene_time,
        preset,
        anchors,
        draft_source=draft_source,
        source_evidence=evidence,
        camera_transform=camera_transform,
        accept_quality_warning=accept_quality_warning,
    )
    reconstruction = video.get("reconstruction") or {}
    register_pending_calibration_edit(
        reconstruction,
        override,
        draft_source=draft_source,
    )
    video["reconstruction"] = reconstruction
    video["processingState"] = "calibration-draft"
    return scenes.put(scene)


__all__ = (
    "build_manual_calibration_override",
    "save_scene_pitch_calibration_draft",
)
