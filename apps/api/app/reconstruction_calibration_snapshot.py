from __future__ import annotations

"""Strict immutable calibration input consumed by full reconstruction."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_store import canonical_json_bytes, digest_from_reference
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_artifact_manifest import artifact_references
from .reconstruction_detection_contract import CalibrationPhaseResult
from .reconstruction_errors import ReconstructionError
from .reconstruction_inputs import source_frame_index
from .temporal_calibration_contract import TemporalCalibrationResult


CALIBRATION_DATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PersistedCalibrationSnapshot:
    result: CalibrationPhaseResult
    temporal: TemporalCalibrationResult
    frame_sizes: dict[int, tuple[int, int]]
    camera_transforms: dict[int, np.ndarray]
    provenance: dict[str, Any]


def calibration_data_fingerprint(reconstruction: Mapping[str, Any]) -> str:
    """Hash only calibration output, excluding later detection/runtime data."""

    calibration = reconstruction.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ReconstructionError("Calibration output is missing")
    frame_evidence = calibration.get("frameEvidence")
    if not isinstance(frame_evidence, Sequence) or isinstance(
        frame_evidence, (str, bytes)
    ):
        raise ReconstructionError(
            "Calibration frame evidence is not materialized"
        )
    orientation = reconstruction.get("pitchOrientation")
    orientation = orientation if isinstance(orientation, Mapping) else {}
    payload = {
        "schemaVersion": CALIBRATION_DATA_SCHEMA_VERSION,
        "calibrationInputFingerprint": reconstruction.get(
            "calibrationInputFingerprint"
        ),
        "pitchCalibration": reconstruction.get("pitchCalibration"),
        "pitchOrientation": {
            "visiblePitchSide": orientation.get("visiblePitchSide"),
            "visiblePitchSideSource": orientation.get(
                "visiblePitchSideSource"
            ),
            "visiblePitchSideAgreement": orientation.get(
                "visiblePitchSideAgreement"
            ),
        },
        "calibration": {
            key: value
            for key, value in calibration.items()
            if key != "frameEvidenceCount"
        },
    }
    return f"sha256:{sha256(canonical_json_bytes(payload)).hexdigest()}"


def calibration_artifact_input(
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin the completed calibration product into a queued full run."""

    provenance = reconstruction.get("calibrationProvenance")
    if not isinstance(provenance, Mapping):
        raise ReconstructionError(
            "Completed calibration provenance is missing; run calibration again"
        )
    data_fingerprint = str(provenance.get("dataFingerprint") or "")
    calibration_input_fingerprint = str(
        reconstruction.get("calibrationInputFingerprint") or ""
    )
    if not data_fingerprint or str(
        provenance.get("calibrationInputFingerprint") or ""
    ) != calibration_input_fingerprint:
        raise ReconstructionError(
            "Completed calibration does not match the current calibration inputs"
        )
    if calibration_data_fingerprint(reconstruction) != data_fingerprint:
        raise ReconstructionError(
            "Completed calibration data does not match its published fingerprint; "
            "run calibration again"
        )
    if not provenance.get("runId") or not provenance.get("producedAt"):
        raise ReconstructionError(
            "Completed calibration has no producer identity; run calibration again"
        )
    reference = artifact_references(reconstruction).get("calibrationFrames")
    if not isinstance(reference, Mapping):
        raise ReconstructionError(
            "Completed calibration artifact is missing; run calibration again"
        )
    digest_from_reference(reference)
    return {
        "schemaVersion": 1,
        "producerRunId": provenance.get("runId"),
        "producedAt": provenance.get("producedAt"),
        "calibrationInputFingerprint": calibration_input_fingerprint,
        "dataFingerprint": data_fingerprint,
        "artifact": deepcopy(dict(reference)),
        "producerArtifact": deepcopy(provenance.get("artifact")),
        "samplingFrameRate": provenance.get("samplingFrameRate"),
        "directCalibrationMaxGapSeconds": provenance.get(
            "directCalibrationMaxGapSeconds"
        ),
        "totalFrames": provenance.get("totalFrames"),
        "resolvedFrames": provenance.get("resolvedFrames"),
        "unresolvedFrames": provenance.get("unresolvedFrames"),
        "coordinateSpace": reconstruction.get("coordinateSpace"),
    }


def _matrix(value: object, *, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(f"{label} is not a numeric matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ReconstructionError(f"{label} must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ReconstructionError(f"{label} is singular")
    return matrix


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _value_fingerprint(value: object) -> str | None:
    if value is None:
        return None
    return f"sha256:{sha256(canonical_json_bytes({'value': value})).hexdigest()}"


def pitch_calibration_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=_matrix(payload.get("imageToPitch"), label=label),
        confidence=float(payload.get("confidence") or 0.0),
        supported_lines=int(payload.get("supportedLines") or 0),
        mean_line_score=float(payload.get("meanLineScore") or 0.0),
        rectangle=str(payload.get("rectangle") or "persisted-calibration"),
        matched_curves=int(payload.get("matchedCurves") or 0),
        method=str(
            payload.get("source")
            or payload.get("method")
            or payload.get("backend")
            or "persisted-calibration"
        ),
        keypoint_count=int(payload.get("keypointCount") or 0),
        inlier_count=int(payload.get("inlierCount") or 0),
        reprojection_error=_optional_float(payload.get("reprojectionError")),
        frame_index=(
            int(payload["sourceFrameIndex"])
            if payload.get("sourceFrameIndex") is not None
            else None
        ),
        detected_keypoint_count=int(payload.get("detectedKeypointCount") or 0),
        completed_keypoint_count=int(payload.get("completedKeypointCount") or 0),
        inlier_ratio=_optional_float(payload.get("inlierRatio")),
        reprojection_p95=_optional_float(payload.get("reprojectionP95")),
        raw_line_count=int(payload.get("rawLineCount") or 0),
        ground_error_p50=_optional_float(payload.get("groundErrorP50Metres")),
        ground_error_p95=_optional_float(payload.get("groundErrorP95Metres")),
        confidence_kind=str(
            payload.get("confidenceKind") or "persisted-calibration-score"
        ),
        raw_keypoints=tuple(
            deepcopy(payload.get("keypoints") or payload.get("rawKeypoints") or [])
        ),
        raw_lines=tuple(deepcopy(payload.get("rawLines") or [])),
    )


def _frame_evidence(
    reconstruction: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    calibration = reconstruction.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ReconstructionError("Calibration output is missing")
    raw = calibration.get("frameEvidence")
    if not isinstance(raw, list) or not all(
        isinstance(item, Mapping) for item in raw
    ):
        raise ReconstructionError("Calibration frame evidence is malformed")
    return list(raw)


def _validate_pinned_input(
    reconstruction: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pinned = reconstruction.get("calibrationArtifactInput")
    if not isinstance(pinned, Mapping):
        raise ReconstructionError(
            "Reconstruction has no pinned calibration artifact input"
        )
    if str(pinned.get("calibrationInputFingerprint") or "") != str(
        reconstruction.get("calibrationInputFingerprint") or ""
    ):
        raise ReconstructionError(
            "Pinned calibration inputs do not match this reconstruction"
        )
    actual_data_fingerprint = calibration_data_fingerprint(reconstruction)
    if str(pinned.get("dataFingerprint") or "") != actual_data_fingerprint:
        raise ReconstructionError(
            "Pinned calibration data failed fingerprint validation"
        )
    pinned_reference = pinned.get("artifact")
    current_reference = artifact_references(reconstruction).get(
        "calibrationFrames"
    )
    if not isinstance(pinned_reference, Mapping) or not isinstance(
        current_reference, Mapping
    ):
        raise ReconstructionError("Pinned calibration artifact is missing")
    if digest_from_reference(pinned_reference) != digest_from_reference(
        current_reference
    ):
        raise ReconstructionError(
            "Calibration artifact changed after reconstruction was queued"
        )
    if pinned.get("totalFrames") is not None and int(
        pinned["totalFrames"]
    ) != len(evidence):
        raise ReconstructionError(
            "Pinned calibration frame count does not match its artifact"
        )
    return dict(pinned)


def load_persisted_calibration_snapshot(
    scene: Mapping[str, Any],
    frames: Sequence[tuple[Path, float]],
) -> PersistedCalibrationSnapshot:
    """Validate and materialize calibration without solving or inference."""

    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction")
        or {}
    )
    evidence = _frame_evidence(reconstruction)
    pinned = _validate_pinned_input(reconstruction, evidence)
    if len(evidence) != len(frames):
        raise ReconstructionError(
            "Calibration artifact does not cover the current sampled frames"
        )

    resolved: dict[int, PitchCalibration] = {}
    anchor_by_sample: dict[int, int] = {}
    uncertainty_by_sample: dict[int, float] = {}
    frame_sizes: dict[int, tuple[int, int]] = {}
    camera_transforms: dict[int, np.ndarray] = {}
    accepted_by_source: dict[int, PitchCalibration] = {}
    accepted_automatic: dict[int, PitchCalibration] = {}
    accepted_manual: dict[int, PitchCalibration] = {}
    seen_samples: set[int] = set()

    for expected_sample, ((path, scene_time), item) in enumerate(
        zip(frames, evidence)
    ):
        sample_index = int(item.get("sampleIndex", expected_sample))
        if sample_index != expected_sample or sample_index in seen_samples:
            raise ReconstructionError(
                "Calibration artifact sample indices are not canonical"
            )
        seen_samples.add(sample_index)
        expected_source = source_frame_index(path)
        if int(item.get("sourceFrameIndex", -1)) != expected_source:
            raise ReconstructionError(
                f"Calibration sample {sample_index} refers to a different source frame"
            )
        if abs(float(item.get("sceneTime", -1.0)) - float(scene_time)) > 0.002:
            raise ReconstructionError(
                f"Calibration sample {sample_index} refers to a different scene time"
            )
        width = int(item.get("frameWidth") or 0)
        height = int(item.get("frameHeight") or 0)
        if width <= 0 or height <= 0:
            raise ReconstructionError(
                f"Calibration sample {sample_index} has no valid frame size"
            )
        frame_sizes[sample_index] = (width, height)
        camera = item.get("cameraMotion")
        if not isinstance(camera, Mapping):
            raise ReconstructionError(
                f"Calibration sample {sample_index} has no camera transform"
            )
        camera_transforms[expected_source] = _matrix(
            camera.get("currentToReference"),
            label=f"Calibration sample {sample_index} camera transform",
        )

        status = str(item.get("solutionStatus") or "")
        projection_source = str(item.get("projectionSource") or "none")
        if "accepted" not in status or projection_source == "none":
            continue
        calibration = pitch_calibration_from_payload(
            item,
            label=f"Calibration sample {sample_index} imageToPitch",
        )
        resolved[sample_index] = calibration
        temporal = item.get("temporal")
        anchor_frames = (
            temporal.get("anchorFrameIndices")
            if isinstance(temporal, Mapping)
            else None
        )
        anchor_by_sample[sample_index] = (
            int(anchor_frames[0])
            if isinstance(anchor_frames, list) and anchor_frames
            else expected_source
        )
        uncertainty = _optional_float(item.get("positionUncertaintyMetres"))
        uncertainty_by_sample[sample_index] = (
            uncertainty if uncertainty is not None else 0.0
        )
        if status == "direct-accepted":
            accepted_by_source[expected_source] = calibration
            if projection_source == "manual-direct":
                accepted_manual[sample_index] = calibration
            else:
                accepted_automatic[sample_index] = calibration

    calibration_contract = reconstruction.get("calibration") or {}
    summary = (
        dict(calibration_contract.get("summary") or {})
        if isinstance(calibration_contract, Mapping)
        else {}
    )
    unresolved_count = len(frames) - len(resolved)
    pitch_metadata = reconstruction.get("pitchCalibration")
    representative = (
        pitch_calibration_from_payload(
            pitch_metadata,
            label="Representative calibration",
        )
        if isinstance(pitch_metadata, Mapping)
        and pitch_metadata.get("imageToPitch") is not None
        else next(iter(resolved.values()), None)
    )
    warnings = [
        str(value)
        for value in reconstruction.get("calibrationWarnings") or []
    ]
    result = CalibrationPhaseResult(
        calibration=representative,
        quality={
            "verdict": "pass" if unresolved_count == 0 else "review",
            "summary": summary,
        },
        coordinate_mode="metric" if resolved else "unavailable",
        metric_calibration=unresolved_count == 0 and bool(resolved),
        frame_evidence=[deepcopy(dict(item)) for item in evidence],
        accepted_frame_calibrations=accepted_by_source,
        accepted_automatic_direct_by_sample=accepted_automatic,
        accepted_manual_direct_by_sample=accepted_manual,
        resolved_calibrations_by_sample=resolved,
        manual_override_by_sample={},
        representative_manual_sample=None,
        rejected_frame_count=sum(
            str(item.get("observationStatus") or "") == "direct-rejected"
            for item in evidence
        ),
        temporal_recovered_frame_count=sum(
            str(item.get("solutionStatus") or "") == "temporal-accepted"
            for item in evidence
        ),
        metric_person_sample_count=0,
        metric_ball_sample_count=0,
        warnings=warnings,
    )
    temporal = TemporalCalibrationResult(
        resolved_by_sample=resolved,
        anchor_by_sample=anchor_by_sample,
        uncertainty_by_sample=uncertainty_by_sample,
        recovered_frame_count=result.temporal_recovered_frame_count,
        metric_person_sample_count=0,
    )
    source_counts: dict[str, int] = {}
    for item in evidence:
        source = str(item.get("projectionSource") or "none")
        source_counts[source] = source_counts.get(source, 0) + 1
    provenance = {
        **pinned,
        "artifactSha256": (pinned.get("artifact") or {}).get("sha256"),
        "frameCount": len(evidence),
        "resolvedFrameCount": len(resolved),
        "unresolvedFrameCount": unresolved_count,
        "projectionSourceCounts": source_counts,
        "calibrationMethods": sorted(
            {calibration.method for calibration in resolved.values()}
        ),
        "sampleUsage": [
            {
                "sampleIndex": int(item.get("sampleIndex", index)),
                "sourceFrameIndex": item.get("sourceFrameIndex"),
                "sceneTime": item.get("sceneTime"),
                "solutionStatus": item.get("solutionStatus"),
                "projectionSource": item.get("projectionSource"),
                "anchorFrameIndices": list(
                    ((item.get("temporal") or {}).get("anchorFrameIndices") or [])
                ),
                "positionUncertaintyMetres": item.get(
                    "positionUncertaintyMetres"
                ),
                "imageToPitchFingerprint": _value_fingerprint(
                    item.get("imageToPitch")
                ),
                "cameraTransformFingerprint": _value_fingerprint(
                    (item.get("cameraMotion") or {}).get(
                        "currentToReference"
                    )
                ),
            }
            for index, item in enumerate(evidence)
        ],
    }
    return PersistedCalibrationSnapshot(
        result=result,
        temporal=temporal,
        frame_sizes=frame_sizes,
        camera_transforms=camera_transforms,
        provenance=provenance,
    )


__all__ = (
    "PersistedCalibrationSnapshot",
    "calibration_artifact_input",
    "calibration_data_fingerprint",
    "load_persisted_calibration_snapshot",
    "pitch_calibration_from_payload",
)
