from __future__ import annotations

"""Build and persist diagnostics for an automatic single-frame calibration proposal."""

from dataclasses import replace
from time import monotonic

import cv2
import numpy as np

from .config import get_settings
from .pitch_anchor_calibration import calibration_from_anchors
from .pitch_calibration_contract import PitchCalibration, pitch_side
from .pitch_calibration_orientation import canonicalize_penalty_side
from .pitch_geometry import ANCHOR_PRESETS
from .pitch_line_calibration import calibrate_pitch
from .reconstruction_calibration_detection import automatic_frame_calibrations
from .reconstruction_calibration_draft import calibration_draft, seed_pitch_anchors
from .reconstruction_calibration_evidence import (
    calibration_attempt_payload,
    calibration_evidence_rank,
    frame_calibration_evidence,
)
from .reconstruction_calibration_frame_context import calibration_frame_context
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import (
    frame_paths,
    source_frame_index as parse_source_frame_index,
)


def propose_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    requested_preset: str | None = None,
) -> dict:
    frame_index, frame_time, image, _ = calibration_frame_context(scene, scene_time)
    frames = frame_paths(scene)
    path = frames[frame_index][0]
    source_frame_index = parse_source_frame_index(path)
    calibration: PitchCalibration | None = None
    selected_evidence: dict | None = None
    attempts: list[dict] = []
    warnings: list[str] = []

    settings = get_settings()
    automatic, automatic_warnings = automatic_frame_calibrations(
        [(path, frame_time)],
        worker_timeout=settings.calibration_frame_worker_timeout,
    )
    warnings.extend(automatic_warnings)
    keypoint_candidate = automatic.get(source_frame_index)
    if keypoint_candidate is not None:
        keypoint_candidate = canonicalize_penalty_side(
            keypoint_candidate,
            image.shape[1],
        )
        keypoint_evidence = frame_calibration_evidence(
            scene,
            frame_index,
            frame_time,
            image,
            keypoint_candidate,
            projection_source="direct",
            source_frame_index=source_frame_index,
        )
        attempts.append(calibration_attempt_payload(keypoint_evidence))
        calibration = keypoint_candidate
        selected_evidence = keypoint_evidence

    # The bounded line/curve solver is a diagnostic fallback for an explicitly
    # requested frame. A healthy semantic-keypoint fit remains authoritative.
    if selected_evidence is None or selected_evidence.get("status") != "accepted":
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / max(1, width))
        fallback_image = (
            image
            if scale == 1.0
            else cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
        fallback_diagnostics: dict = {
            "inputWidth": fallback_image.shape[1],
            "inputHeight": fallback_image.shape[0],
        }
        fallback_started = monotonic()
        line_candidate = calibrate_pitch(
            fallback_image,
            max_quad_candidates=240,
            deadline=fallback_started + 5.0,
            diagnostics=fallback_diagnostics,
        )
        fallback_diagnostics.update(
            {
                "inputWidth": fallback_image.shape[1],
                "inputHeight": fallback_image.shape[0],
                "elapsedSeconds": round(monotonic() - fallback_started, 3),
            }
        )
        if fallback_diagnostics.get("deadlineExceeded"):
            warnings.append(
                "The bounded line/curve fallback reached its five-second deadline; its best-so-far result was retained when available."
            )
        if fallback_diagnostics.get("candidateLimitReached"):
            warnings.append(
                "The bounded line/curve fallback reached its candidate search limit before the deadline; its best-so-far result was retained when available."
            )
        if line_candidate is not None:
            if scale != 1.0:
                full_to_small = np.asarray(
                    [
                        [scale, 0.0, 0.0],
                        [0.0, scale, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                lifted = line_candidate.image_to_pitch @ full_to_small
                lifted /= lifted[2, 2]
                line_candidate = replace(line_candidate, image_to_pitch=lifted)
            line_candidate = canonicalize_penalty_side(
                line_candidate,
                image.shape[1],
            )
            line_evidence = frame_calibration_evidence(
                scene,
                frame_index,
                frame_time,
                image,
                line_candidate,
                projection_source="direct",
                source_frame_index=source_frame_index,
            )
            line_evidence["backendDiagnostics"] = fallback_diagnostics
            attempts.append(calibration_attempt_payload(line_evidence))
            if selected_evidence is None or calibration_evidence_rank(
                line_evidence
            ) > calibration_evidence_rank(selected_evidence):
                calibration = line_candidate
                selected_evidence = line_evidence
        elif keypoint_candidate is None:
            warnings.append(
                "Neither semantic keypoints nor the line/curve fallback found a camera fit."
            )

    preset = requested_preset
    if preset is None:
        calibrated_side = (
            pitch_side(calibration.rectangle) if calibration is not None else None
        )
        preset = (
            calibration.rectangle
            if calibration is not None and calibration.rectangle in ANCHOR_PRESETS
            else f"penalty-area-{calibrated_side}"
            if calibrated_side in {"left", "right"}
            else "center-circle"
        )
    if preset not in ANCHOR_PRESETS:
        raise ReconstructionError("Unsupported pitch anchor preset")
    if calibration is None:
        anchors = seed_pitch_anchors(preset, image.shape[1], image.shape[0])
        seed = calibration_from_anchors(anchors, preset, confidence=0.35)
        warnings.append("Automatic pitch fit failed; align the four anchors manually.")
        selected_evidence = frame_calibration_evidence(
            scene,
            frame_index,
            frame_time,
            image,
            None,
            projection_source="none",
            source_frame_index=source_frame_index,
        )
        draft = calibration_draft(
            scene,
            frame_index,
            frame_time,
            image,
            seed,
            preset,
            "manual-seed",
            anchors,
            warnings,
        )
    else:
        assert selected_evidence is not None
        if selected_evidence.get("status") != "accepted":
            warnings.append(
                "The best current-frame candidate failed geometric QA; inspect the reasons or refine its anchors manually."
            )
        draft = calibration_draft(
            scene,
            frame_index,
            frame_time,
            image,
            calibration,
            preset,
            "frame-evidence",
            warnings=warnings,
        )

    assert selected_evidence is not None
    selected_evidence["attempts"] = attempts
    draft.update(
        {
            "requestedSceneTime": round(float(scene_time), 3),
            "sampleIndex": frame_index,
            "sourceFrameIndex": source_frame_index,
            "sourceTime": selected_evidence.get("sourceTime"),
            "status": selected_evidence["status"],
            "solutionStatus": selected_evidence["solutionStatus"],
            "method": selected_evidence.get("backend"),
            "backend": selected_evidence.get("backend"),
            "confidenceKind": selected_evidence.get("confidenceKind"),
            "keypointCount": selected_evidence.get("keypointCount", 0),
            "detectedKeypointCount": selected_evidence.get(
                "detectedKeypointCount",
                0,
            ),
            "inlierCount": selected_evidence.get("inlierCount", 0),
            "inlierRatio": selected_evidence.get("inlierRatio"),
            "reprojectionP95": selected_evidence.get("reprojectionP95"),
            "visiblePitchSide": selected_evidence.get("visiblePitchSide"),
            "rejectionReasons": selected_evidence.get("rejectionReasons") or [],
            "qualityGates": selected_evidence.get("qualityGates") or [],
            "keypoints": selected_evidence.get("keypoints") or [],
            "detectedKeypoints": selected_evidence.get("keypoints") or [],
            "rawLines": selected_evidence.get("rawLines") or [],
            "attempts": attempts,
            "evidence": selected_evidence,
        }
    )
    # The proposal is a read-only diagnostic draft: nothing is persisted and
    # the Scene revision must not change until the user explicitly applies it.
    return draft
