"""Artifact-backed composition of the identity-review read model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifact_store import ArtifactStore, ReconstructionArtifactError
from .identity_review_errors import IdentityReviewArtifactUnavailableError
from .identity_review_observation_projection import crop_evidence_by_observation
from .identity_review_person_projection import identity_review_items
from .project_match_persistence_contract import MatchSnapshotDocument
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_identity_artifacts import load_identity_diagnostics
from .reconstruction_artifact_manifest import artifact_references


def _reconstruction(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    video = scene.get("payload", {}).get("videoAsset", {})
    reconstruction = (
        video.get("reconstruction", {}) if isinstance(video, Mapping) else {}
    )
    return reconstruction if isinstance(reconstruction, Mapping) else {}


def _availability(scene: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    reconstruction = _reconstruction(scene)
    references = artifact_references(reconstruction)
    status = str(reconstruction.get("status") or "not-started")
    if status in {"queued", "processing", "failed", "cancelled"}:
        return {"state": status, "available": False}, False
    if status == "not-started":
        return {"state": "not-started", "available": False}, False
    if status == "ready":
        has_diagnostics = "identityDiagnostics" in references
        has_timeline = "identityTimeline" in references
        if has_diagnostics != has_timeline:
            raise ReconstructionArtifactError(
                "Identity review artifact publication is incomplete"
            )
        if has_diagnostics:
            return {"state": "ready", "available": True}, True
        return {
            "state": "unavailable",
            "available": False,
            "reasonCode": "identity-review-artifacts-not-published",
        }, False
    return {
        "state": "unavailable",
        "available": False,
        "reasonCode": "reconstruction-state-unrecognized",
    }, False


def _identity_diagnostics(
    reconstruction: Mapping[str, Any],
    artifact_store: ArtifactStore | None,
) -> dict[str, Any]:
    try:
        return load_identity_diagnostics(reconstruction, store=artifact_store)
    except ReconstructionArtifactError as exc:
        raise IdentityReviewArtifactUnavailableError(
            f"Identity review evidence is unavailable: {exc}"
        ) from exc


def _roster_status(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    players = snapshot.get("roster") or []
    sync = snapshot.get("sync") if isinstance(snapshot.get("sync"), dict) else {}
    warnings = [str(value) for value in sync.get("warnings") or []]
    quality = (
        snapshot.get("rosterQuality")
        if isinstance(snapshot.get("rosterQuality"), dict)
        else {}
    )
    automatic_eligible = quality.get("automaticIdentityEligible")
    if automatic_eligible is True:
        status = "ready"
    elif not players:
        status = "unavailable"
    elif automatic_eligible is False:
        status = "incomplete"
    else:
        status = "review"
    return {
        "status": status,
        "playerCount": len(players),
        "complete": automatic_eligible is True,
        "automaticIdentityEligible": automatic_eligible is True,
        "manualIdentityEligible": bool(
            quality.get("manualIdentityEligible", bool(players))
        ),
        "reasons": [str(reason) for reason in quality.get("reasons") or []],
        # Warnings are display-only; never parse prose into product state.
        "warnings": warnings,
    }


def build_identity_review_projection(
    scene: Mapping[str, Any],
    *,
    match_snapshot: MatchSnapshotDocument | None,
    worker_health: Mapping[str, Any] | None = None,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Build a deterministic review queue without HTTP links or scene writes."""

    scene = deepcopy(scene)
    try:
        availability, has_identity_diagnostics = _availability(scene)
    except ReconstructionArtifactError as exc:
        raise IdentityReviewArtifactUnavailableError(
            f"Identity review evidence is unavailable: {exc}"
        ) from exc
    canonical_match = match_snapshot.payload if match_snapshot is not None else {}
    if not has_identity_diagnostics:
        return {
            "sceneId": scene.get("id"),
            "revision": scene.get("revision", 0),
            "availability": availability,
            "matchSnapshot": {
                "id": match_snapshot.id if match_snapshot is not None else None,
                "contentHash": (
                    match_snapshot.content_hash if match_snapshot is not None else None
                ),
                "matchId": match_snapshot.match_id if match_snapshot is not None else None,
                "roster": _roster_status(canonical_match),
            },
            "workers": deepcopy(worker_health or {}),
            "summary": {
                "canonicalPersonCount": 0,
                "boundCount": 0,
                "suggestedCount": 0,
                "conflictCount": 0,
                "anonymousCount": 0,
                "excludedCount": 0,
            },
            "items": [],
        }
    try:
        hydrate_scene_reconstruction(
            scene,
            names=("identityTimeline",),
            store=artifact_store,
        )
    except ReconstructionArtifactError as exc:
        raise IdentityReviewArtifactUnavailableError(
            f"Identity review evidence is unavailable: {exc}"
        ) from exc
    payload = scene.get("payload", {})
    diagnostics = _identity_diagnostics(_reconstruction(scene), artifact_store)
    reid_by_observation, jersey_by_observation = crop_evidence_by_observation(
        diagnostics
    )
    items = identity_review_items(
        scene,
        payload.get("canonicalPeople"),
        reid_by_observation,
        jersey_by_observation,
    )
    return {
        "sceneId": scene.get("id"),
        "revision": scene.get("revision", 0),
        "availability": availability,
        "matchSnapshot": {
            "id": match_snapshot.id if match_snapshot is not None else None,
            "contentHash": (
                match_snapshot.content_hash if match_snapshot is not None else None
            ),
            "matchId": match_snapshot.match_id if match_snapshot is not None else None,
            "roster": _roster_status(canonical_match),
        },
        "workers": deepcopy(worker_health or {}),
        "summary": {
            "canonicalPersonCount": len(items),
            "boundCount": sum(item["resolutionState"] == "bound" for item in items),
            "suggestedCount": sum(
                item["resolutionState"] == "suggested" for item in items
            ),
            "conflictCount": sum(
                item["resolutionState"] == "conflict" for item in items
            ),
            "anonymousCount": sum(
                item["resolutionState"] == "anonymous" for item in items
            ),
            "excludedCount": sum(
                item["resolutionState"] == "excluded" for item in items
            ),
        },
        "items": items,
    }


__all__ = ("build_identity_review_projection",)
