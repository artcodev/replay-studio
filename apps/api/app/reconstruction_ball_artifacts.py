"""Publish immutable ball trajectory artifacts and compact their Scene view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_store import ArtifactStore, reconstruction_artifact_store
from .reconstruction_artifact_codec import (
    ball_artifact_payload,
    compact_ball,
    materialized_artifacts,
    set_materialized_artifacts,
)
from .reconstruction_artifact_manifest import (
    BALL_TRAJECTORY_ARTIFACT_KIND,
    BALL_TRAJECTORY_SCHEMA_VERSION,
    artifact_references,
    existing_artifact_reference,
    merge_artifact_manifest,
)


@dataclass(frozen=True)
class PublishedBallTrajectory:
    reference: dict[str, Any]
    compact: dict[str, Any] | None


def publish_ball_trajectory_content(
    scene: Mapping[str, Any],
    payload: Mapping[str, Any],
    references: Mapping[str, Any],
    materialized: set[str],
    *,
    store: ArtifactStore,
) -> PublishedBallTrajectory:
    """Publish materialized ball samples, or retain the current immutable ref."""

    ball = payload.get("ball") if isinstance(payload.get("ball"), Mapping) else {}
    reference = existing_artifact_reference(references, "ballTrajectory")
    needs_publication = reference is None or "ballTrajectory" in materialized
    needs_publication = needs_publication or any(
        key in ball for key in ("keyframes", "automaticKeyframes", "manualKeyframes")
    )
    if not needs_publication:
        assert reference is not None
        return PublishedBallTrajectory(reference, None)

    reference = store.put_json(
        kind=BALL_TRAJECTORY_ARTIFACT_KIND,
        schema_version=BALL_TRAJECTORY_SCHEMA_VERSION,
        payload=ball_artifact_payload(str(scene.get("id") or ""), ball),
    )
    return PublishedBallTrajectory(reference, compact_ball(ball))


def publish_ball_trajectory_artifact(
    scene: dict[str, Any],
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Publish and compact only the edited ball domain of one Scene."""

    artifact_store = store or reconstruction_artifact_store()
    payload = scene.setdefault("payload", {})
    reconstruction = payload.setdefault("videoAsset", {}).setdefault(
        "reconstruction", {}
    )
    references = artifact_references(reconstruction)
    materialized = materialized_artifacts(reconstruction)
    published = publish_ball_trajectory_content(
        scene,
        payload,
        references,
        materialized,
        store=artifact_store,
    )
    if published.compact is None:
        return published.reference

    reconstruction["artifactManifest"] = merge_artifact_manifest(
        reconstruction.get("artifactManifest"),
        ballTrajectory=published.reference,
    )
    set_materialized_artifacts(
        reconstruction,
        materialized - {"ballTrajectory"},
    )
    payload["ball"] = published.compact
    return published.reference


__all__ = (
    "PublishedBallTrajectory",
    "publish_ball_trajectory_artifact",
    "publish_ball_trajectory_content",
)
