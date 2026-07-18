"""Atomic Scene compaction coordinator for all dense reconstruction domains."""

from __future__ import annotations

from typing import Any

from .artifact_store import ArtifactStore, reconstruction_artifact_store
from .reconstruction_artifact_codec import (
    materialized_artifacts,
    set_materialized_artifacts,
)
from .reconstruction_artifact_manifest import (
    artifact_references,
    merge_artifact_manifest,
)
from .reconstruction_ball_artifacts import publish_ball_trajectory_content
from .reconstruction_calibration_artifacts import publish_calibration_frames_artifact
from .reconstruction_identity_artifacts import publish_identity_timeline_artifact


def publish_dense_reconstruction_artifacts(
    scene: dict[str, Any],
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Publish every materialized domain, then atomically compact the Scene view."""

    artifact_store = store or reconstruction_artifact_store()
    payload = scene.setdefault("payload", {})
    reconstruction = payload.setdefault("videoAsset", {}).setdefault(
        "reconstruction", {}
    )
    references = artifact_references(reconstruction)
    materialized = materialized_artifacts(reconstruction)

    identity = publish_identity_timeline_artifact(
        scene,
        payload,
        references,
        materialized,
        store=artifact_store,
    )
    ball = publish_ball_trajectory_content(
        scene,
        payload,
        references,
        materialized,
        store=artifact_store,
    )
    calibration = publish_calibration_frames_artifact(
        scene,
        reconstruction,
        references,
        materialized,
        store=artifact_store,
    )

    reconstruction["artifactManifest"] = merge_artifact_manifest(
        reconstruction.get("artifactManifest"),
        identityTimeline=identity.reference,
        ballTrajectory=ball.reference,
        calibrationFrames=calibration.reference,
    )
    set_materialized_artifacts(reconstruction, ())
    if identity.encoding is not None:
        payload["tracks"] = identity.encoding.compact_tracks
        payload["canonicalPeople"] = identity.encoding.compact_people
    if ball.compact is not None:
        payload["ball"] = ball.compact
    if calibration.encoding is not None:
        reconstruction["calibration"] = calibration.encoding.compact_calibration
        reconstruction["ballDetection"] = calibration.encoding.compact_ball_detection
    return reconstruction["artifactManifest"]


__all__ = ("publish_dense_reconstruction_artifacts",)
