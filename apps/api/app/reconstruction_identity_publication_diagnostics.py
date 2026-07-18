from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from .jersey_ocr_contract import JerseyEvidenceSummary
from .reconstruction_track_state import TrackState


def identity_publication_diagnostics(
    documents: list[dict],
    tracks: list[TrackState],
    *,
    resolver_diagnostics: Mapping | None,
    roster_diagnostics: dict,
    closed_set_diagnostics: dict,
    jersey_evidence: Mapping[str, JerseyEvidenceSummary],
    source_tracklet_count: int,
    total_observations: int,
    total_reid_observations: int,
) -> dict:
    return {
        **(deepcopy(resolver_diagnostics) if resolver_diagnostics else {}),
        "sourceTrackletCount": source_tracklet_count,
        "canonicalPersonCount": len(documents),
        "resolvedPersonCount": sum(
            item["identityStatus"] == "resolved" for item in documents
        ),
        "provisionalPersonCount": sum(
            item["identityStatus"] == "provisional" for item in documents
        ),
        "excludedPersonCount": sum(
            item["identityStatus"] == "excluded" for item in documents
        ),
        "conflictPersonCount": sum(bool(item["conflicts"]) for item in documents),
        "manualRosterJerseyConflictCount": sum(
            any(
                conflict.get("code") == "manual-roster-jersey-conflict"
                for conflict in item["conflicts"]
            )
            for item in documents
        ),
        "manualRosterMissingConflictCount": sum(
            any(
                conflict.get("code") == "manual-roster-player-missing"
                for conflict in item["conflicts"]
            )
            for item in documents
        ),
        "manualDecisionCount": sum(
            item["identitySource"] == "manual" for item in documents
        ),
        "identityObservationCount": total_observations,
        "reidUsableObservationCount": total_reid_observations,
        "reidSelectedIndependentSampleCount": sum(
            track.reid_feature_count for track in tracks
        ),
        "reidCropCoverage": round(
            total_reid_observations / max(1, total_observations), 3
        ),
        "jerseyReadablePersonCount": sum(
            summary.selected_sample_count > 0 for summary in jersey_evidence.values()
        ),
        "jerseyReliablePersonCount": sum(
            summary.status == "reliable" for summary in jersey_evidence.values()
        ),
        "jerseyProvisionalPersonCount": sum(
            summary.status == "provisional" for summary in jersey_evidence.values()
        ),
        "jerseyConflictPersonCount": sum(
            summary.status == "conflict" for summary in jersey_evidence.values()
        ),
        "jerseyReadableCoverage": round(
            sum(
                summary.selected_sample_count > 0
                for summary in jersey_evidence.values()
            )
            / max(1, len(documents)),
            3,
        ),
        "rosterCandidateCount": sum(
            len(item.get("rosterCandidates") or []) for item in documents
        ),
        "rosterPrior": roster_diagnostics,
        "closedSetRoster": closed_set_diagnostics,
    }
