from __future__ import annotations

from typing import Mapping

from .jersey_ocr_contract import JerseyEvidenceSummary
from .reconstruction_canonical_person_id import derive_canonical_person_id
from .reconstruction_track_state import TrackState
from .reconstruction_identity_document_projection import build_identity_document
from .reconstruction_identity_publication_diagnostics import (
    identity_publication_diagnostics,
)
from .reconstruction_roster_identity_resolution import (
    apply_closed_set_roster_resolution,
    match_snapshot_roster,
)


def canonical_people_documents(
    tracks: list[TrackState],
    mapping: dict[int, str],
    rendered_tracks: list[dict],
    scene: dict,
    resolver_diagnostics: dict | None = None,
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None = None,
    match_snapshot: Mapping[str, object] | None = None,
) -> tuple[list[dict], dict]:
    source_start = float(
        scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0
    )
    rendered_by_identity = {
        str(item.get("canonicalPersonId")): item
        for item in rendered_tracks
        if item.get("canonicalPersonId")
    }
    roster, roster_diagnostics = match_snapshot_roster(match_snapshot)
    roster_by_external_id = {
        player.external_player_id: player for player in roster
    }
    jersey_by_identity = jersey_evidence or {}
    documents: list[dict] = []
    source_tracklets: set[str] = set()
    total_observations = 0
    total_reid_observations = 0
    for track in tracks:
        canonical_id = str(
            track.canonical_person_id or derive_canonical_person_id(track)
        )
        source_tracklets.update(track.source_tracklet_ids or {track.local_tracklet_id})
        document = build_identity_document(
            track,
            mapping=mapping,
            rendered=rendered_by_identity.get(canonical_id),
            canonical_id=canonical_id,
            source_start=source_start,
            roster_by_external_id=roster_by_external_id,
            roster_diagnostics=roster_diagnostics,
            jersey_summary=jersey_by_identity.get(canonical_id),
            resolver_diagnostics=resolver_diagnostics,
        )
        documents.append(document)
        total_observations += len(document["observations"])
        total_reid_observations += track.reid_observation_count

    closed_set_diagnostics = apply_closed_set_roster_resolution(
        documents,
        scene,
        match_snapshot,
        roster,
        roster_diagnostics,
        jersey_by_identity,
    )
    documents.sort(
        key=lambda item: (
            item.get("teamId") or "unknown",
            item.get("displayName") or item["id"],
            item["id"],
        )
    )
    return documents, identity_publication_diagnostics(
        documents,
        tracks,
        resolver_diagnostics=resolver_diagnostics,
        roster_diagnostics=roster_diagnostics,
        closed_set_diagnostics=closed_set_diagnostics,
        jersey_evidence=jersey_by_identity,
        source_tracklet_count=len(source_tracklets),
        total_observations=total_observations,
        total_reid_observations=total_reid_observations,
    )
