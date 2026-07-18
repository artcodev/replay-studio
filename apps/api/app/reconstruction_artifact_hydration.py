"""Strict loading and temporary in-memory hydration of dense artifacts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .artifact_store import (
    ArtifactStore,
    ReconstructionArtifactError,
    reconstruction_artifact_store,
)
from .reconstruction_artifact_codec import (
    hydrate_ball,
    hydrate_calibration,
    hydrate_identity_timeline,
    mark_materialized_artifacts,
)
from .reconstruction_artifact_manifest import (
    DENSE_ARTIFACT_CONTRACTS,
    artifact_references,
)


def load_dense_reconstruction_artifacts(
    reconstruction: Mapping[str, Any],
    *,
    names: Iterable[str] | None = None,
    store: ArtifactStore | None = None,
) -> dict[str, dict[str, Any]]:
    """Load selected dense payloads and validate every immutable reference."""

    references = artifact_references(reconstruction)
    selected = tuple(names or DENSE_ARTIFACT_CONTRACTS)
    artifact_store = store or reconstruction_artifact_store()
    result: dict[str, dict[str, Any]] = {}
    for name in selected:
        contract = DENSE_ARTIFACT_CONTRACTS.get(name)
        if contract is None:
            raise ReconstructionArtifactError(f"Unknown dense artifact {name!r}")
        reference = references.get(name)
        if reference is None:
            continue
        if not isinstance(reference, Mapping):
            raise ReconstructionArtifactError(
                f"Artifact reference {name!r} is malformed"
            )
        result[name] = artifact_store.get_json(
            reference,
            expected_kind=contract[0],
            expected_schema_version=contract[1],
        )
    return result


def hydrate_scene_reconstruction(
    scene: dict[str, Any],
    *,
    names: Iterable[str] | None = None,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Materialize selected dense domains in a mutable working Scene."""

    payload = scene.setdefault("payload", {})
    reconstruction = payload.get("videoAsset", {}).get("reconstruction") or {}
    loaded = load_dense_reconstruction_artifacts(
        reconstruction,
        names=names,
        store=store,
    )
    if loaded and not isinstance(reconstruction, dict):
        raise ReconstructionArtifactError(
            "Reconstruction must be mutable while artifacts are hydrated"
        )
    if identity := loaded.get("identityTimeline"):
        hydrate_identity_timeline(payload, identity)
    if ball := loaded.get("ballTrajectory"):
        hydrate_ball(payload, ball)
    if calibration := loaded.get("calibrationFrames"):
        hydrate_calibration(reconstruction, calibration)
    if loaded:
        mark_materialized_artifacts(reconstruction, loaded)
    return scene


__all__ = ("hydrate_scene_reconstruction", "load_dense_reconstruction_artifacts")
