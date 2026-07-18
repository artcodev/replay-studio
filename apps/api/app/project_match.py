from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .project_match_persistence_contract import MatchSnapshotDocument, MatchSnapshotSummary


def project_parent_scene_id(scene: dict) -> str | None:
    """Return the direct project owner advertised by a derived scene."""

    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    parent_id = multi_pass.get("parentSceneId") or video.get("parentSceneId")
    normalized = str(parent_id or "").strip()
    return normalized or None


def is_multi_pass_scene(scene: dict) -> bool:
    video = scene.get("payload", {}).get("videoAsset") or {}
    return bool(video.get("multiPass"))


def is_single_pass_reconstruction_scene(scene: dict) -> bool:
    video = scene.get("payload", {}).get("videoAsset") or {}
    return bool(video.get("selectedSegmentId")) and not bool(video.get("multiPass"))


def match_snapshot_reference(
    snapshot: MatchSnapshotDocument | MatchSnapshotSummary | None,
) -> dict[str, Any] | None:
    """Return the compact immutable match input captured by an analysis run."""

    if snapshot is None:
        return None
    return {
        "id": str(snapshot.id),
        "contentHash": str(snapshot.content_hash),
        "schemaVersion": int(snapshot.schema_version),
    }


def normalized_match_snapshot_reference(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and canonicalize an already persisted snapshot reference."""

    if not isinstance(value, Mapping):
        return None
    snapshot_id = str(value.get("id") or "").strip()
    content_hash = str(value.get("contentHash") or "").strip()
    if not snapshot_id or not content_hash:
        return None
    try:
        schema_version = max(1, int(value.get("schemaVersion") or 1))
    except (TypeError, ValueError):
        return None
    return {
        "id": snapshot_id,
        "contentHash": content_hash,
        "schemaVersion": schema_version,
    }


def reconstruction_match_snapshot_reference(scene: Mapping[str, Any]) -> dict[str, Any] | None:
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    return normalized_match_snapshot_reference(reconstruction.get("matchSnapshotRef"))


def snapshot_matches_reference(
    snapshot: MatchSnapshotDocument,
    reference: Mapping[str, Any] | None,
) -> bool:
    return match_snapshot_reference(snapshot) == normalized_match_snapshot_reference(
        reference
    )
