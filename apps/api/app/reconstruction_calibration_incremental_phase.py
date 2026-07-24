from __future__ import annotations

"""Finalize staged manual corrections without repeating neural inference."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .direct_calibration_sampling import resolve_direct_calibration_max_gap_seconds
from .pitch_calibration_contract import PitchCalibration
from .pitch_line_mask_cache import cached_pitch_line_mask_loader
from .reconstruction_calibration_edit_session import (
    pending_calibration_edit_session,
)
from .reconstruction_calibration_evidence import (
    calibration_attempt_payload,
    frame_calibration_evidence,
)
from .reconstruction_calibration_overrides import (
    manual_pitch_calibration_overrides,
)
from .reconstruction_calibration_resolution import (
    merge_direct_calibration_anchors,
    resolve_temporal_frame_calibrations,
)
from .reconstruction_calibration_selection import (
    select_representative_calibration,
)
from .reconstruction_calibration_snapshot import (
    pitch_calibration_from_payload,
)
from .reconstruction_dense_ball_phase import skipped_dense_ball_result
from .reconstruction_detection_contract import CalibrationPhaseResult
from .reconstruction_detection_result_projection import (
    project_calibration_phase_result,
)
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import source_frame_index
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_sampled_calibration import manual_calibration_inputs
from .reconstruction_sampled_frame_contract import (
    SampledCalibrationAnalysis,
    SampledCalibrationInputs,
)
from .temporal_calibration_contract import TemporalCalibrationResult
from .config import get_settings


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix))) < 1e-10
    ):
        raise ReconstructionError(f"{label} must be a finite invertible 3x3 matrix")
    return matrix


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _motion_from_evidence(item: Mapping, sample_index: int) -> CameraMotionEstimate:
    camera = item.get("cameraMotion")
    if not isinstance(camera, Mapping):
        raise ReconstructionError(
            f"Calibration sample {sample_index} has no stored camera motion"
        )
    metrics = camera.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    reasons = camera.get("rejectionReasons")
    return CameraMotionEstimate(
        matrix=_matrix(
            camera.get("currentToPrevious"),
            f"Calibration sample {sample_index} motion edge",
        ),
        status=str(camera.get("status") or "missing"),
        confidence=float(camera.get("confidence") or 0.0),
        tracked_count=int(metrics.get("trackedCount") or 0),
        inlier_count=int(metrics.get("inlierCount") or 0),
        inlier_ratio=float(metrics.get("inlierRatio") or 0.0),
        residual_p50=_optional_float(metrics.get("residualP50Px")),
        residual_p95=_optional_float(metrics.get("residualP95Px")),
        forward_backward_p95=_optional_float(
            metrics.get("forwardBackwardP95Px")
        ),
        coverage_ratio=float(metrics.get("coverageRatio") or 0.0),
        scene_change_score=_optional_float(metrics.get("sceneChangeScore")),
        reason=(
            str(reasons[0])
            if isinstance(reasons, list) and reasons
            else None
        ),
    )


def affected_calibration_samples(
    frames: Sequence[tuple[Path, float]],
    evidence: Sequence[Mapping],
    *,
    edited_sample_indices: set[int],
    direct_sample_indices: set[int],
    max_gap_seconds: float,
) -> set[int]:
    """Return exactly the frames whose solution may depend on edited anchors."""

    affected = set(edited_sample_indices)
    edited_times = {
        sample: float(frames[sample][1])
        for sample in edited_sample_indices
        if 0 <= sample < len(frames)
    }
    for sample_index, ((_, scene_time), item) in enumerate(zip(frames, evidence)):
        if sample_index in direct_sample_indices:
            continue
        temporal = item.get("temporal")
        old_anchors = (
            temporal.get("anchorSampleIndices")
            if isinstance(temporal, Mapping)
            else []
        )
        depended_before = bool(
            set(int(value) for value in old_anchors or [])
            & edited_sample_indices
        )
        can_depend_now = any(
            abs(float(scene_time) - edited_time) <= max_gap_seconds + 1e-9
            for edited_time in edited_times.values()
        )
        if depended_before or can_depend_now:
            affected.add(sample_index)
    return affected


def _restore_direct_observation(item: dict) -> None:
    if str(item.get("projectionSource") or "") in {"direct", "manual-direct"}:
        return
    observation = item.get("observation")
    if not isinstance(observation, Mapping):
        item.update(
            {
                "status": "missing",
                "source": "none",
                "projectionSource": "none",
                "backend": None,
                "confidence": None,
                "imageToPitch": None,
                "visiblePitchSide": None,
                "rejectionReasons": ["no-automatic-calibration-candidate"],
            }
        )
        return
    item.update(
        {
            "status": observation.get("status") or "missing",
            "source": observation.get("source") or "none",
            "projectionSource": observation.get("projectionSource") or "none",
            "backend": observation.get("backend"),
            "confidence": observation.get("confidence"),
            "imageToPitch": deepcopy(observation.get("imageToPitch")),
            "visiblePitchSide": observation.get("visiblePitchSide"),
            "rejectionReasons": list(
                observation.get("rejectionReasons") or []
            ),
        }
    )


def finalize_staged_calibration_phase(
    scene: dict,
    frames: list[tuple[Path, float]],
    *,
    reconstruction_request: Mapping,
    progress: ReconstructionProgress,
) -> CalibrationPhaseResult:
    reconstruction = scene["payload"]["videoAsset"].get("reconstruction") or {}
    session = pending_calibration_edit_session(reconstruction)
    if session is None:
        raise ReconstructionError("The staged calibration edit session is missing")
    calibration = reconstruction.get("calibration")
    evidence_raw = (
        calibration.get("frameEvidence")
        if isinstance(calibration, Mapping)
        else None
    )
    if not isinstance(evidence_raw, list) or len(evidence_raw) != len(frames):
        raise ReconstructionError(
            "The base calibration artifact does not cover the current frame set"
        )
    evidence = [deepcopy(dict(item)) for item in evidence_raw]
    edited = {
        int(value)
        for value in session.get("editedSampleIndices") or []
        if 0 <= int(value) < len(frames)
    }
    if not edited:
        raise ReconstructionError("No staged calibration frames were selected")

    progress.update(
        "calibration",
        2,
        "Applying saved frame corrections",
        (
            f"Loading {len(edited)} authoritative manual correction(s) and "
            "reusing published PnLCalib evidence and camera-motion edges. "
            "No neural inference is run."
        ),
        8,
        34,
        completed=0,
        total=len(edited),
    )

    frame_sizes: dict[int, tuple[int, int]] = {}
    camera_transforms: dict[int, np.ndarray] = {}
    motion_edges: dict[int, CameraMotionEstimate] = {}
    baseline_resolved: dict[int, PitchCalibration] = {}
    baseline_anchor_by_sample: dict[int, int] = {}
    baseline_uncertainty: dict[int, float] = {}
    automatic_direct: dict[int, PitchCalibration] = {}
    existing_manual_direct: dict[int, PitchCalibration] = {}

    for sample_index, ((path, _), item) in enumerate(zip(frames, evidence)):
        expected_source = source_frame_index(path)
        if int(item.get("sourceFrameIndex", -1)) != expected_source:
            raise ReconstructionError(
                f"Base calibration sample {sample_index} refers to another frame"
            )
        width = int(item.get("frameWidth") or 0)
        height = int(item.get("frameHeight") or 0)
        if width <= 0 or height <= 0:
            raise ReconstructionError(
                f"Base calibration sample {sample_index} has no frame size"
            )
        frame_sizes[sample_index] = (width, height)
        camera = item.get("cameraMotion")
        if not isinstance(camera, Mapping):
            raise ReconstructionError(
                f"Base calibration sample {sample_index} has no camera transform"
            )
        camera_transforms[expected_source] = _matrix(
            camera.get("currentToReference"),
            f"Base calibration sample {sample_index} camera transform",
        )
        if sample_index > 0:
            motion_edges[sample_index] = _motion_from_evidence(item, sample_index)

        accepted = (
            "accepted" in str(item.get("solutionStatus") or "")
            and str(item.get("projectionSource") or "none") != "none"
            and item.get("imageToPitch") is not None
        )
        if accepted:
            resolved = pitch_calibration_from_payload(
                item,
                label=f"Base calibration sample {sample_index}",
            )
            baseline_resolved[sample_index] = resolved
            temporal = item.get("temporal")
            anchors = (
                temporal.get("anchorFrameIndices")
                if isinstance(temporal, Mapping)
                else []
            )
            baseline_anchor_by_sample[sample_index] = (
                int(anchors[0]) if anchors else expected_source
            )
            baseline_uncertainty[sample_index] = float(
                item.get("positionUncertaintyMetres") or 0.0
            )
        if str(item.get("solutionStatus") or "") == "direct-accepted":
            direct = pitch_calibration_from_payload(
                item,
                label=f"Base direct calibration sample {sample_index}",
            )
            if str(item.get("projectionSource") or "") == "manual-direct":
                existing_manual_direct[sample_index] = direct
            else:
                automatic_direct[sample_index] = direct

    overrides = manual_pitch_calibration_overrides(reconstruction)
    manual_stabilized, manual_override_by_sample = manual_calibration_inputs(
        frames,
        overrides,
    )
    manual_direct = dict(existing_manual_direct)
    for completed, sample_index in enumerate(sorted(edited), start=1):
        stabilized = manual_stabilized.get(sample_index)
        if stabilized is None:
            raise ReconstructionError(
                f"Staged calibration sample {sample_index} has no manual override"
            )
        source_index = source_frame_index(frames[sample_index][0])
        current_to_pitch = (
            stabilized.image_to_pitch @ camera_transforms[source_index]
        )
        current_to_pitch /= current_to_pitch[2, 2]
        manual = replace(
            stabilized,
            image_to_pitch=current_to_pitch,
            frame_index=source_index,
        )
        image = cv2.imread(str(frames[sample_index][0]))
        if image is None:
            raise ReconstructionError(
                f"Could not decode edited calibration frame {source_index}"
            )
        previous_evidence = evidence[sample_index]
        manual_evidence = frame_calibration_evidence(
            scene,
            sample_index,
            float(frames[sample_index][1]),
            image,
            manual,
            projection_source="manual-direct",
            people=[],
            pitch=scene["payload"]["pitch"],
            source_frame_index=source_index,
            manual=True,
        )
        manual_evidence["cameraMotion"] = deepcopy(
            previous_evidence.get("cameraMotion")
        )
        manual_evidence["manualObservation"] = {
            "kind": "manual",
            **calibration_attempt_payload(manual_evidence),
        }
        manual_evidence["automaticObservation"] = {
            "kind": "automatic",
            "status": previous_evidence.get("status"),
            "backend": previous_evidence.get("backend"),
            "confidence": previous_evidence.get("confidence"),
            "rejectionReasons": list(
                previous_evidence.get("rejectionReasons") or []
            ),
        }
        manual_evidence["observations"] = [
            manual_evidence["manualObservation"],
            manual_evidence["automaticObservation"],
        ]
        manual_evidence["observationChoice"] = {
            "selectedKind": "manual",
            "reason": "operator-staged-frame-override",
            "automaticCandidateStatus": previous_evidence.get("status"),
        }
        evidence[sample_index] = manual_evidence
        automatic_direct.pop(sample_index, None)
        manual_direct[sample_index] = manual
        progress.update(
            "calibration",
            2,
            "Applying saved frame corrections",
            (
                f"Applied correction {completed}/{len(edited)} at frame "
                f"#{source_index}. Existing neural evidence remains unchanged."
            ),
            8,
            34,
            completed=completed,
            total=len(edited),
        )

    all_direct = merge_direct_calibration_anchors(
        automatic_direct,
        manual_direct,
    )
    # Match the normal temporal solver exactly: an authoritative manual anchor
    # may recover any non-direct frame in the same continuous shot. Direct
    # PnLCalib observations remain immutable and are never recomputed here.
    max_gap_seconds = (
        max(2.0, float(scene.get("duration") or 0.0))
        if manual_stabilized
        else max(
            2.0,
            resolve_direct_calibration_max_gap_seconds(
                reconstruction_request.get("directCalibrationMaxGapSeconds")
            ),
        )
    )
    affected = affected_calibration_samples(
        frames,
        evidence,
        edited_sample_indices=edited,
        direct_sample_indices=set(all_direct),
        max_gap_seconds=max_gap_seconds,
    )
    for sample_index in affected - edited:
        _restore_direct_observation(evidence[sample_index])

    progress.update(
        "calibration",
        2,
        "Resolving affected temporal frames",
        (
            f"{len(affected)} of {len(frames)} frame(s) may depend on the edited "
            f"anchors; {len(frames) - len(affected)} published frame solutions "
            "will be copied byte-for-byte."
        ),
        34,
        90,
        completed=0,
        total=len(affected),
    )
    (
        affected_resolved,
        affected_anchor_by_sample,
        affected_uncertainty,
        _,
    ) = resolve_temporal_frame_calibrations(
        frames,
        frame_sizes,
        all_direct,
        motion_edges,
        evidence,
        [([], scene_time) for _, scene_time in frames],
        scene["payload"]["pitch"],
        max_gap_seconds=max_gap_seconds,
        observed_mask_loader=cached_pitch_line_mask_loader(
            Path(get_settings().media_root) / "pitch-line-masks",
            enabled=bool(get_settings().pitch_line_mask_cache_enabled),
        ),
        target_sample_indices=affected,
    )
    resolved = {
        sample: value
        for sample, value in baseline_resolved.items()
        if sample not in affected
    }
    resolved.update(affected_resolved)
    anchor_by_sample = {
        sample: value
        for sample, value in baseline_anchor_by_sample.items()
        if sample not in affected
    }
    anchor_by_sample.update(affected_anchor_by_sample)
    uncertainty = {
        sample: value
        for sample, value in baseline_uncertainty.items()
        if sample not in affected
    }
    uncertainty.update(affected_uncertainty)

    progress.update(
        "calibration",
        2,
        "Affected-frame QA complete",
        (
            f"Recomputed {len(affected)} affected frame(s), reused "
            f"{len(frames) - len(affected)}, and ran 0 PnLCalib inference(s)."
        ),
        90,
        96,
        completed=len(affected),
        total=len(affected),
    )
    if progress.run_log is not None:
        progress.run_log.event(
            "calibration-incremental-finalization",
            editedSampleIndices=sorted(edited),
            affectedSampleIndices=sorted(affected),
            reusedSampleCount=len(frames) - len(affected),
            pnlcalibInferenceCount=0,
            baseDataFingerprint=session.get("baseDataFingerprint"),
            baseArtifactSha256=session.get("baseArtifactSha256"),
        )

    accepted_by_source = {
        source_frame_index(frames[sample][0]): calibration
        for sample, calibration in all_direct.items()
    }
    sampled = SampledCalibrationAnalysis(
        frame_size=frame_sizes[max(frame_sizes)],
        frame_sizes=frame_sizes,
        camera_motion_edges=motion_edges,
        camera_transforms=camera_transforms,
        accepted_frame_calibrations=accepted_by_source,
        accepted_automatic_direct_by_sample=automatic_direct,
        accepted_manual_direct_by_sample=manual_direct,
        frame_evidence=evidence,
        rejected_frame_count=sum(
            str(item.get("observationStatus") or "") == "direct-rejected"
            for item in evidence
        ),
    )
    temporal_recovered = sum(
        str(item.get("solutionStatus") or "") == "temporal-accepted"
        for item in evidence
    )
    temporal = TemporalCalibrationResult(
        resolved_by_sample=resolved,
        anchor_by_sample=anchor_by_sample,
        uncertainty_by_sample=uncertainty,
        recovered_frame_count=temporal_recovered,
        metric_person_sample_count=0,
    )
    inputs = SampledCalibrationInputs(
        manual_reference=overrides[-1] if overrides else {},
        frame_calibrations={},
        calibration_warnings=[
            (
                f"Incremental finalization applied {len(edited)} staged manual "
                f"correction(s), recomputed {len(affected)} affected frame(s), "
                f"reused {len(frames) - len(affected)} published solution(s), "
                "and ran no PnLCalib inference."
            )
        ],
        manual_stabilized_by_sample=manual_stabilized,
        manual_override_by_sample=manual_override_by_sample,
    )
    selection = select_representative_calibration(
        frames=frames,
        frame_size=sampled.frame_size,
        frame_evidence=evidence,
        accepted_frame_calibrations=accepted_by_source,
        accepted_manual_direct_by_sample=manual_direct,
        camera_transforms=camera_transforms,
        manual_stabilized_by_sample=manual_stabilized,
        manual_reference=inputs.manual_reference,
        rejected_frame_count=sampled.rejected_frame_count,
        temporal_recovered_frame_count=temporal_recovered,
        warnings=inputs.calibration_warnings,
    )
    return project_calibration_phase_result(
        sampled,
        inputs,
        temporal,
        skipped_dense_ball_result("calibrate-only"),
        selection,
    )


__all__ = (
    "affected_calibration_samples",
    "finalize_staged_calibration_phase",
)
