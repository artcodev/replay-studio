from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from .jersey_ocr_contract import JerseyEvidenceSummary
from .reconstruction_track_state import TrackState
from .reconstruction_identity_evidence_projection import (
    append_jersey_conflicts,
    identity_evidence_projection,
    published_identity_observations,
)
from .reconstruction_identity_semantics import annotation_role, annotation_team


def _identity_status(track: TrackState) -> tuple[str, float | None]:
    manually_resolved = bool(track.positive_annotation_ids or track.manual_external_player_id)
    status = (
        "resolved"
        if manually_resolved or track.identity_status == "resolved"
        else "excluded"
        if track.identity_status == "excluded"
        else "provisional"
    )
    confidence = 1.0 if manually_resolved else track.identity_confidence
    if confidence is not None:
        confidence = round(max(0.0, min(1.0, float(confidence))), 3)
    return status, confidence


def build_identity_document(
    track: TrackState,
    *,
    mapping: Mapping[int, str],
    rendered: dict | None,
    canonical_id: str,
    source_start: float,
    roster_by_external_id: Mapping[str, object],
    roster_diagnostics: Mapping,
    jersey_summary: JerseyEvidenceSummary | None,
    resolver_diagnostics: Mapping | None,
) -> dict:
    observations = published_identity_observations(
        track, rendered, canonical_id, source_start
    )
    team = mapping.get(track.id) or annotation_team(track.manual_kind)
    role = annotation_role(track.manual_kind) or track.role or "player"
    conflicts = deepcopy(track.identity_conflicts)
    bound_external_id = str(track.manual_external_player_id or "")
    bound_player = roster_by_external_id.get(bound_external_id)
    if bound_external_id and bound_player is None:
        conflicts.append(
            {
                "id": f"{canonical_id}:manual-roster-player-missing",
                "code": "manual-roster-player-missing",
                "message": (
                    "The confirmed roster player is absent or ambiguous in the "
                    "current persisted match roster; the manual binding was retained "
                    "for review."
                ),
                "severity": "review",
                "externalPlayerId": bound_external_id,
                "bindingAnnotationIds": sorted(track.roster_binding_annotation_ids),
                "rosterStatus": roster_diagnostics["status"],
            }
        )
    if jersey_summary is not None:
        append_jersey_conflicts(
            conflicts,
            canonical_id=canonical_id,
            track=track,
            summary=jersey_summary,
            bound_roster_player=bound_player,
        )
    status, confidence = _identity_status(track)
    default_label = track.manual_label or (
        rendered.get("label") if rendered is not None else None
    )
    if not default_label:
        default_label = f"{str(team).title()} person" if team else "Unassigned person"
    member_tracklets = sorted(track.source_tracklet_ids or {track.local_tracklet_id})
    identity_source = (
        "manual"
        if track.positive_annotation_ids or track.manual_external_player_id
        else "reid+trajectory"
        if track.reid_feature_count
        else "jersey-ocr+trajectory"
        if jersey_summary is not None and jersey_summary.status == "reliable"
        else "tracker+trajectory"
    )
    return {
        "id": canonical_id,
        "canonicalPersonId": canonical_id,
        "displayName": default_label,
        "identityStatus": status,
        "identityConfidence": confidence,
        "identitySource": identity_source,
        "teamId": team,
        "role": role,
        "jerseyNumber": jersey_summary.jersey_number if jersey_summary else None,
        "candidateNumber": jersey_summary.candidate_number if jersey_summary else None,
        "externalPlayerId": track.manual_external_player_id,
        "annotationIds": sorted(track.annotation_ids),
        "sourceTrackletIds": member_tracklets,
        "memberTrackletIds": member_tracklets,
        "observationCount": len(observations),
        "observations": observations,
        "renderTrackId": rendered.get("id") if rendered is not None else None,
        "evidence": identity_evidence_projection(
            track,
            canonical_id=canonical_id,
            observation_count=len(observations),
            jersey_summary=jersey_summary,
            resolver_diagnostics=resolver_diagnostics,
        ),
        "rosterCandidates": [],
        "conflicts": conflicts,
        "provenance": (
            "manual"
            if track.positive_annotation_ids
            else "mixed"
            if len(member_tracklets) > 1
            else "automatic"
        ),
    }
