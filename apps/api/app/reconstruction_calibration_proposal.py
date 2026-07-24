from __future__ import annotations

"""Build and persist diagnostics for an automatic single-frame calibration proposal."""

from .config import get_settings
from .pitch_anchor_calibration import calibration_from_anchors
from .pitch_calibration_contract import PitchCalibration, pitch_side
from .pitch_geometry import ANCHOR_PRESETS
from .reconstruction_calibration_detection import automatic_frame_calibrations
from .reconstruction_calibration_draft import calibration_draft, seed_pitch_anchors
from .reconstruction_calibration_evidence import frame_calibration_evidence
from .reconstruction_calibration_frame_context import sampled_frame_context
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import (
    frame_paths,
    source_frame_index as parse_source_frame_index,
)
from .reconstruction_pnlcalib_retry import resolve_pnlcalib_frame_attempts


def propose_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    requested_preset: str | None = None,
) -> dict:
    frame_index, frame_time, image = sampled_frame_context(scene, scene_time)
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
    resolution = resolve_pnlcalib_frame_attempts(
        scene,
        sample_index=frame_index,
        source_frame_index=source_frame_index,
        scene_time=frame_time,
        frame_path=path,
        image=image,
        initial_calibration=automatic.get(source_frame_index),
        additional_attempts=settings.calibration_pnlcalib_retry_count,
        worker_timeout=settings.calibration_frame_worker_timeout,
    )
    keypoint_candidate = resolution.calibration
    selected_evidence = resolution.evidence
    attempts = [dict(item) for item in resolution.attempts]
    calibration = keypoint_candidate
    if resolution.accepted_attempt is not None and resolution.accepted_attempt > 1:
        warnings.append(
            f"PnLCalib direct calibration passed QA on attempt "
            f"{resolution.accepted_attempt}/{1 + settings.calibration_pnlcalib_retry_count}."
        )

    if keypoint_candidate is None:
        warnings.append(
            f"PnLCalib did not find a camera fit after {len(attempts)} attempt(s); "
            "align the manual anchors or fix the PnLCalib evidence. No automatic "
            "fallback was used."
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
            "groundErrorP50Metres": selected_evidence.get(
                "groundErrorP50Metres"
            ),
            "groundErrorP95Metres": selected_evidence.get(
                "groundErrorP95Metres"
            ),
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
