from __future__ import annotations

"""Pure construction of the compact Scene state for one queued run."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .reconstruction_errors import ReconstructionError
from .reconstruction_coordinate_policy import (
    METRIC_REQUIRED,
    TRACKING_COORDINATE_POLICIES,
)
from .reconstruction_progress import queued_progress
from .scene_document import reconstruction_input_fingerprint


@dataclass(frozen=True)
class ReconstructionQueueInputs:
    """Every resolved input needed to describe a queued reconstruction run."""

    model: str
    ball_backend: str
    ball_detection_input: Mapping[str, Any]
    frame_count: int
    run_id: str
    match_snapshot_ref: Mapping[str, Any] | None
    sampling_frame_rate: float
    direct_calibration_max_gap_seconds: float = 0.0
    ball_detection_profile: str = "automatic"
    jersey_ocr_profile: str = "automatic"
    contact_point_profile: str = "bbox-bottom"
    mode: str = "full"
    tracking_coordinate_policy: str = METRIC_REQUIRED
    calibration_fallback_consent: Mapping[str, Any] | None = None
    calibration_input_fingerprint: str | None = None
    calibration_artifact_input: Mapping[str, Any] | None = None
    calibration_trigger: str | None = None


def validate_reconstruction_queue_scene(scene: Mapping[str, Any]) -> None:
    payload = scene.get("payload")
    video = payload.get("videoAsset") if isinstance(payload, Mapping) else None
    if not isinstance(video, Mapping):
        raise ReconstructionError("Scene has no source video")
    if video.get("multiPass"):
        raise ReconstructionError(
            "Multi-pass composites must be rebuilt with multi-angle analysis, "
            "not single-pass reconstruction"
        )


def _ball_sample_count(ball: object) -> int:
    if not isinstance(ball, Mapping):
        return 0
    keyframes = ball.get("keyframes")
    if isinstance(keyframes, list):
        return len(keyframes)
    try:
        return max(0, int(ball.get("keyframeCount") or 0))
    except (TypeError, ValueError):
        return 0


def prepare_reconstruction_queue_draft(
    scene: Mapping[str, Any],
    inputs: ReconstructionQueueInputs,
) -> dict[str, Any]:
    """Return a new queued Scene document without storage or artifact I/O."""

    validate_reconstruction_queue_scene(scene)
    if not inputs.model.strip():
        raise ReconstructionError("Reconstruction model must be selected")
    if not inputs.ball_backend.strip():
        raise ReconstructionError("Ball detection backend must be selected")
    if not inputs.run_id.strip():
        raise ReconstructionError("Reconstruction run ID must be assigned")
    if inputs.frame_count < 0:
        raise ReconstructionError("Reconstruction frame count cannot be negative")
    if inputs.ball_detection_profile not in {
        "automatic",
        "skip-manual-authoritative",
    }:
        raise ReconstructionError("Unknown ball detection profile")
    if inputs.ball_detection_profile == "skip-manual-authoritative":
        ball = scene.get("payload", {}).get("ball")
        if not isinstance(ball, Mapping) or ball.get("mode") != "manual":
            raise ReconstructionError(
                "Ball detection can be skipped only while the manual ball "
                "trajectory is the authoritative channel"
            )
    if inputs.jersey_ocr_profile not in {"automatic", "off"}:
        raise ReconstructionError("Unknown jersey OCR profile")
    if inputs.contact_point_profile not in {"bbox-bottom", "pose-feet"}:
        raise ReconstructionError("Unknown contact point profile")
    if inputs.sampling_frame_rate <= 0.0:
        raise ReconstructionError("Analysis sampling FPS must be positive")
    if not 0.0 <= inputs.direct_calibration_max_gap_seconds <= 5.0:
        raise ReconstructionError(
            "Direct calibration sampling gap must be between 0 and 5 seconds"
        )
    if inputs.mode not in {"calibrate", "full"}:
        raise ReconstructionError("Unknown reconstruction mode")
    if inputs.tracking_coordinate_policy not in TRACKING_COORDINATE_POLICIES:
        raise ReconstructionError("Unknown tracking coordinate policy")
    if (
        inputs.tracking_coordinate_policy != METRIC_REQUIRED
        and not isinstance(inputs.calibration_fallback_consent, Mapping)
    ):
        raise ReconstructionError(
            "Image fallback requires explicit calibration consent"
        )
    if inputs.mode == "full" and not isinstance(
        inputs.calibration_artifact_input, Mapping
    ):
        raise ReconstructionError(
            "Full reconstruction requires a pinned calibration artifact"
        )
    if inputs.mode == "calibrate" and inputs.calibration_artifact_input is not None:
        raise ReconstructionError(
            "Calibration runs cannot consume a reconstruction calibration input"
        )
    if inputs.calibration_trigger not in {
        None,
        "full-request",
        "manual-draft-finalize",
    }:
        raise ReconstructionError("Unknown calibration trigger")
    if inputs.mode != "calibrate" and inputs.calibration_trigger is not None:
        raise ReconstructionError(
            "Only a calibration process may carry a calibration trigger"
        )

    draft = deepcopy(scene)
    payload = draft.setdefault("payload", {})
    video = payload["videoAsset"]
    previous = video.get("reconstruction") or {}
    previous_result = {
        "completedAt": previous.get("completedAt"),
        "trackCount": len(payload.get("tracks") or []),
        "ballSamples": _ball_sample_count(payload.get("ball")),
        "calibrationStatus": (previous.get("pitchCalibration") or {}).get(
            "status"
        ),
    }
    run_revision = int(previous.get("runRevision") or 0) + 1
    previous_diagnostics = {
        key: value
        for key, value in (previous.get("diagnostics") or {}).items()
        if key != "identityCorrections"
    }

    video["processingState"] = "reconstructing"
    reconstruction = {
        **previous,
        "model": inputs.model,
        "ballBackend": inputs.ball_backend,
        "ballDetectionInput": deepcopy(dict(inputs.ball_detection_input)),
        "ballDetectionProfile": inputs.ball_detection_profile,
        "jerseyOcrProfile": inputs.jersey_ocr_profile,
        "contactPointProfile": inputs.contact_point_profile,
        "samplingFrameRate": inputs.sampling_frame_rate,
        "directCalibrationMaxGapSeconds": (
            inputs.direct_calibration_max_gap_seconds
        ),
        "mode": inputs.mode,
        **(
            {"calibrationTrigger": inputs.calibration_trigger}
            if inputs.calibration_trigger is not None
            else {}
        ),
        "trackingCoordinatePolicy": inputs.tracking_coordinate_policy,
        **(
            {"calibrationInputFingerprint": inputs.calibration_input_fingerprint}
            if inputs.calibration_input_fingerprint
            else {}
        ),
        # A calibrate run stops at the inspection gate; a full run continues past
        # it. Published calibration evidence is replaced atomically only after a
        # successful run. Keeping the previous gate while a new run computes
        # makes failures inspectable without making that gate current: its
        # calibrationInputFingerprint is compared with the newly queued input.
        # A full run consumes the pinned artifact behind that gate, so its
        # review/timeline remains valid operator evidence as well.
        "stage": "calibration" if inputs.mode == "calibrate" else "reconstruction",
        "identityCorrectionDiagnostics": [],
        "diagnostics": previous_diagnostics,
    }
    if inputs.calibration_trigger is None:
        reconstruction.pop("calibrationTrigger", None)
    if inputs.calibration_fallback_consent is None:
        reconstruction.pop("calibrationFallbackConsent", None)
    else:
        reconstruction["calibrationFallbackConsent"] = deepcopy(
            dict(inputs.calibration_fallback_consent)
        )
    if inputs.calibration_artifact_input is None:
        reconstruction.pop("calibrationArtifactInput", None)
    else:
        reconstruction["calibrationArtifactInput"] = deepcopy(
            dict(inputs.calibration_artifact_input)
        )
    if inputs.match_snapshot_ref is None:
        reconstruction.pop("matchSnapshotRef", None)
    else:
        reconstruction["matchSnapshotRef"] = deepcopy(
            dict(inputs.match_snapshot_ref)
        )
    video["reconstruction"] = reconstruction

    input_fingerprint = reconstruction_input_fingerprint(draft)
    reconstruction.update(
        {
            "status": "queued",
            "processingStatus": "queued",
            "qualityVerdict": "pending",
            "quality": None,
            "runId": inputs.run_id,
            "runRevision": run_revision,
            "inputFingerprint": input_fingerprint,
            "error": None,
            "startedAt": None,
            "completedAt": None,
            "frameCount": inputs.frame_count,
            "trackCount": previous_result["trackCount"],
            "ballSamples": previous_result["ballSamples"],
            "warnings": [],
            "previousResult": previous_result,
            "progress": queued_progress(inputs.frame_count, mode=inputs.mode),
        }
    )
    return draft


__all__ = (
    "ReconstructionQueueInputs",
    "prepare_reconstruction_queue_draft",
    "validate_reconstruction_queue_scene",
)
