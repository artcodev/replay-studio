"""Reconstruction artifact kinds, references, and manifest validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifact_store import ReconstructionArtifactError


ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
IDENTITY_DIAGNOSTICS_ARTIFACT_KIND = "reconstruction.identity-diagnostics"
IDENTITY_DIAGNOSTICS_SCHEMA_VERSION = 1
IDENTITY_TIMELINE_ARTIFACT_KIND = "reconstruction.identity-timeline"
IDENTITY_TIMELINE_SCHEMA_VERSION = 1
BALL_TRAJECTORY_ARTIFACT_KIND = "reconstruction.ball-trajectory"
BALL_TRAJECTORY_SCHEMA_VERSION = 1
CALIBRATION_FRAMES_ARTIFACT_KIND = "reconstruction.calibration-frames"
CALIBRATION_FRAMES_SCHEMA_VERSION = 1

DENSE_ARTIFACT_CONTRACTS = {
    "identityTimeline": (
        IDENTITY_TIMELINE_ARTIFACT_KIND,
        IDENTITY_TIMELINE_SCHEMA_VERSION,
    ),
    "ballTrajectory": (
        BALL_TRAJECTORY_ARTIFACT_KIND,
        BALL_TRAJECTORY_SCHEMA_VERSION,
    ),
    "calibrationFrames": (
        CALIBRATION_FRAMES_ARTIFACT_KIND,
        CALIBRATION_FRAMES_SCHEMA_VERSION,
    ),
}


def artifact_references(reconstruction: Mapping[str, Any]) -> dict[str, Any]:
    manifest = reconstruction.get("artifactManifest")
    if manifest is None:
        return {}
    if not isinstance(manifest, Mapping):
        raise ReconstructionArtifactError(
            "Reconstruction artifact manifest is malformed"
        )
    if manifest.get("schemaVersion") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ReconstructionArtifactError(
            "Unsupported reconstruction artifact manifest schema"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReconstructionArtifactError(
            "Reconstruction artifact manifest is malformed"
        )
    return dict(artifacts)


def existing_artifact_reference(
    references: Mapping[str, Any],
    name: str,
) -> dict[str, Any] | None:
    reference = references.get(name)
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise ReconstructionArtifactError(
            f"Artifact reference {name!r} is malformed"
        )
    return deepcopy(dict(reference))


def merge_artifact_manifest(
    manifest: Mapping[str, Any] | None,
    **references: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest is None:
        existing: dict[str, Any] = {}
    else:
        if manifest.get("schemaVersion") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ReconstructionArtifactError(
                "Unsupported reconstruction artifact manifest schema"
            )
        values = manifest.get("artifacts")
        if not isinstance(values, Mapping):
            raise ReconstructionArtifactError(
                "Reconstruction artifact manifest is malformed"
            )
        existing = dict(values)
    existing.update({key: dict(value) for key, value in references.items()})
    return {
        "schemaVersion": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifacts": existing,
    }


def required_identity_diagnostics_reference(
    reconstruction: Mapping[str, Any],
) -> Mapping[str, Any]:
    references = artifact_references(reconstruction)
    reference = references.get("identityDiagnostics")
    if not isinstance(reference, Mapping):
        raise ReconstructionArtifactError(
            "Identity diagnostics artifact reference is missing"
        )
    return reference
