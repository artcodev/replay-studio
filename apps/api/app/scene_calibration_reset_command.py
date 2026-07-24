from __future__ import annotations

"""Reset the calibration stage so the operator can recalibrate from scratch.

Clears the manual overrides, the review gate, and the calibration outputs.
Dropping ``pitchCalibrationOverrides`` changes the input fingerprint, so any
prior gate is cleanly invalidated. The manually chosen pitch orientation is
preserved.
"""

from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


# Calibration-stage outputs to drop. Job-level status fields are left untouched
# (a fresh calibrate run overwrites them); `pitchOrientation` is preserved.
_CLEARED_RECONSTRUCTION_FIELDS = (
    "pitchCalibrationOverrides",
    "pendingCalibrationEditSession",
    "calibrationReview",
    "calibrationInputFingerprint",
    "calibrationProvenance",
    "calibrationArtifactInput",
    "calibrationWarnings",
    "trackingCoordinatePolicy",
    "calibrationFallbackConsent",
    "calibration",
    "pitchCalibration",
    "ballDetection",
    "coordinateSpace",
    "stage",
    "mode",
)


def reset_scene_calibration(scene: dict) -> dict:
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for calibration to finish before resetting it"
        )
    for field in _CLEARED_RECONSTRUCTION_FIELDS:
        reconstruction.pop(field, None)
    manifest = reconstruction.get("artifactManifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("artifacts"), dict):
        manifest["artifacts"].pop("calibrationFrames", None)
    video["reconstruction"] = reconstruction
    video["processingState"] = "frames-ready"
    return scenes.put(scene)


__all__ = ("reset_scene_calibration",)
