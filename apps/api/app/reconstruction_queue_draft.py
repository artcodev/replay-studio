from __future__ import annotations

"""Pure construction of the compact Scene state for one queued run."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .reconstruction_errors import ReconstructionError
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
    ball_detection_profile: str = "automatic"
    jersey_ocr_profile: str = "automatic"


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
        "identityCorrectionDiagnostics": [],
        "diagnostics": previous_diagnostics,
    }
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
            "progress": queued_progress(inputs.frame_count),
        }
    )
    return draft


__all__ = (
    "ReconstructionQueueInputs",
    "prepare_reconstruction_queue_draft",
    "validate_reconstruction_queue_scene",
)
