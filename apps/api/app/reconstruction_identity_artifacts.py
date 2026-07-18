"""Publish and read immutable identity reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_store import ArtifactStore, reconstruction_artifact_store
from .reconstruction_artifact_codec import (
    IdentityTimelineEncoding,
    compact_identity_diagnostics,
    encode_identity_timeline,
)
from .reconstruction_artifact_manifest import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    IDENTITY_DIAGNOSTICS_ARTIFACT_KIND,
    IDENTITY_DIAGNOSTICS_SCHEMA_VERSION,
    IDENTITY_TIMELINE_ARTIFACT_KIND,
    IDENTITY_TIMELINE_SCHEMA_VERSION,
    existing_artifact_reference,
    required_identity_diagnostics_reference,
)


@dataclass(frozen=True)
class PublishedIdentityTimeline:
    reference: dict[str, Any]
    encoding: IdentityTimelineEncoding | None


def publish_identity_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    store: ArtifactStore | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish full review evidence and return its compact Scene summary."""

    artifact_store = store or reconstruction_artifact_store()
    reference = artifact_store.put_json(
        kind=IDENTITY_DIAGNOSTICS_ARTIFACT_KIND,
        schema_version=IDENTITY_DIAGNOSTICS_SCHEMA_VERSION,
        payload=diagnostics,
    )
    return (
        {
            "schemaVersion": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "artifacts": {"identityDiagnostics": reference},
        },
        compact_identity_diagnostics(diagnostics),
    )


def publish_identity_timeline_artifact(
    scene: Mapping[str, Any],
    payload: Mapping[str, Any],
    references: Mapping[str, Any],
    materialized: set[str],
    *,
    store: ArtifactStore,
) -> PublishedIdentityTimeline:
    """Publish a materialized identity timeline, or retain its immutable ref."""

    tracks = payload.get("tracks") or []
    people = payload.get("canonicalPeople") or []
    reference = existing_artifact_reference(references, "identityTimeline")
    needs_publication = reference is None or "identityTimeline" in materialized
    needs_publication = needs_publication or any(
        isinstance(item, Mapping)
        and ("keyframes" in item or "observations" in item)
        for item in tracks
    ) or any(
        isinstance(item, Mapping) and "observations" in item for item in people
    )
    if not needs_publication:
        assert reference is not None
        return PublishedIdentityTimeline(reference, None)

    encoding = encode_identity_timeline(
        str(scene.get("id") or ""),
        tracks,
        people,
    )
    reference = store.put_json(
        kind=IDENTITY_TIMELINE_ARTIFACT_KIND,
        schema_version=IDENTITY_TIMELINE_SCHEMA_VERSION,
        payload=encoding.payload,
    )
    return PublishedIdentityTimeline(reference, encoding)


def load_identity_diagnostics(
    reconstruction: Mapping[str, Any],
    *,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Load full identity evidence through its strict manifest contract."""

    reference = required_identity_diagnostics_reference(reconstruction)
    return (store or reconstruction_artifact_store()).get_json(
        reference,
        expected_kind=IDENTITY_DIAGNOSTICS_ARTIFACT_KIND,
        expected_schema_version=IDENTITY_DIAGNOSTICS_SCHEMA_VERSION,
    )


__all__ = (
    "PublishedIdentityTimeline",
    "load_identity_diagnostics",
    "publish_identity_diagnostics",
    "publish_identity_timeline_artifact",
)
