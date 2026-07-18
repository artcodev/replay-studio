from __future__ import annotations

"""Closed-set roster hypotheses derived from the persisted match snapshot."""

from typing import Mapping

from .identity_decisions import rejected_roster_candidate_ids
from .jersey_ocr_contract import JerseyEvidenceSummary
from .jersey_roster_candidates import RosterPlayer
from .closed_set_roster_resolution import resolve_closed_set_roster
from .roster_identity_contract import (
    AttributeEvidence as RosterAttributeEvidence,
    CanonicalPersonEvidence as RosterCanonicalPersonEvidence,
    PersistedRosterPlayer,
)

def match_snapshot_roster(
    match_snapshot: Mapping[str, object] | None,
) -> tuple[list[RosterPlayer], dict]:
    snapshot = match_snapshot if isinstance(match_snapshot, Mapping) else {}
    roster_quality = (
        snapshot.get("rosterQuality")
        if isinstance(snapshot.get("rosterQuality"), dict)
        else {}
    )
    raw_players = snapshot.get("roster") or []
    players: list[RosterPlayer] = []
    invalid_count = 0
    identifiers: set[str] = set()
    duplicate_identifiers: set[str] = set()
    for item in raw_players:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        identifier = str(item.get("id") or "").strip()
        if not identifier:
            invalid_count += 1
            continue
        if identifier in identifiers:
            duplicate_identifiers.add(identifier)
            continue
        identifiers.add(identifier)
        try:
            players.append(
                RosterPlayer(
                    external_player_id=identifier,
                    display_name=str(item.get("name") or identifier),
                    jersey_number=item.get("number"),
                    team_id=item.get("team_id") or item.get("teamId"),
                    role=item.get("position") or item.get("role"),
                )
            )
        except ValueError:
            invalid_count += 1
    if duplicate_identifiers:
        # Duplicate external IDs make any candidate confirmation ambiguous.
        players = []
    return players, {
        "availablePlayerCount": len(raw_players),
        "usablePlayerCount": len(players),
        "invalidPlayerCount": invalid_count,
        "duplicateExternalPlayerIds": sorted(duplicate_identifiers),
        "automaticIdentityEligible": bool(
            roster_quality.get("automaticIdentityEligible")
        ),
        "manualIdentityEligible": bool(
            roster_quality.get("manualIdentityEligible", bool(players))
        ),
        "qualityStatus": roster_quality.get("status") or "unknown",
        "qualityReasons": list(roster_quality.get("reasons") or []),
        "status": (
            "invalid-duplicate-ids"
            if duplicate_identifiers
            else "ready"
            if players
            else "unavailable"
        ),
    }


def _external_roster_team_id(
    match_snapshot: Mapping[str, object] | None,
    local_team_id: str | None,
) -> str | None:
    if local_team_id is None:
        return None
    snapshot = match_snapshot if isinstance(match_snapshot, Mapping) else {}
    team = snapshot.get("homeTeam" if local_team_id == "home" else "awayTeam")
    if isinstance(team, dict) and team.get("id"):
        return str(team["id"])
    return str(local_team_id)


def apply_closed_set_roster_resolution(
    documents: list[dict],
    scene: dict,
    match_snapshot: Mapping[str, object] | None,
    roster: list[RosterPlayer],
    roster_diagnostics: dict,
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None,
) -> dict:
    """Publish review-only, globally unique roster hypotheses.

    The persisted match snapshot is the only closed set. Incomplete snapshots
    remain usable for explicit manual binding but never constrain automatic
    hypotheses. The resolver itself cannot write ``externalPlayerId``.
    """

    for document in documents:
        document["rosterCandidates"] = []

    base_diagnostics = {
        "status": "unavailable",
        "schemaVersion": 1,
        "automaticBindingCount": 0,
        "requiresManualConfirmation": True,
        "matchClockAligned": False,
        "reasons": [],
    }
    if not roster_diagnostics.get("automaticIdentityEligible"):
        base_diagnostics["status"] = "disabled-incomplete-roster"
        base_diagnostics["reasons"] = list(
            roster_diagnostics.get("qualityReasons")
            or ["persisted-roster-not-eligible-for-automatic-identity"]
        )
        for document in documents:
            document["rosterResolution"] = {
                "status": "abstain",
                "suggestedExternalPlayerId": None,
                "requiresManualConfirmation": False,
                "reasons": list(base_diagnostics["reasons"]),
                "conflicts": [],
            }
        return base_diagnostics
    if not roster:
        base_diagnostics["reasons"] = ["persisted-roster-empty-or-invalid"]
        return base_diagnostics

    persisted_players = [
        PersistedRosterPlayer(
            external_player_id=player.external_player_id,
            display_name=player.display_name,
            team_id=player.team_id,
            jersey_number=player.jersey_number,
            role=player.role,
            # Video source time is not match-clock time. Availability windows
            # are deliberately omitted until clock alignment is explicit.
            active_intervals=(),
        )
        for player in roster
    ]
    resolver_people: list[RosterCanonicalPersonEvidence] = []
    skipped_ids: set[str] = set()
    for document in documents:
        canonical_id = str(document["canonicalPersonId"])
        if document.get("identityStatus") == "excluded":
            skipped_ids.add(canonical_id)
            document["rosterResolution"] = {
                "status": "abstain",
                "suggestedExternalPlayerId": None,
                "requiresManualConfirmation": False,
                "reasons": ["canonical-identity-excluded"],
                "conflicts": [],
            }
            continue
        manual = document.get("provenance") == "manual"
        local_team_id = document.get("teamId")
        external_team_id = _external_roster_team_id(match_snapshot, local_team_id)
        team_evidence = (
            RosterAttributeEvidence(
                value=external_team_id,
                confidence=0.96 if manual else 0.76,
                source="manual-team-label" if manual else "team-clustering",
                confirmed=manual,
            )
            if external_team_id
            else None
        )
        role = document.get("role")
        role_evidence = (
            RosterAttributeEvidence(
                value=str(role),
                confidence=0.96 if manual else 0.72,
                source="manual-role-label" if manual else "role-classifier",
                confirmed=manual,
            )
            if role
            else None
        )
        jersey_summary = (jersey_evidence or {}).get(canonical_id)
        jersey_value = (
            jersey_summary.jersey_number or jersey_summary.candidate_number
            if jersey_summary is not None
            else None
        )
        jersey_attribute = (
            RosterAttributeEvidence(
                value=jersey_value,
                confidence=float(jersey_summary.confidence),
                source="jersey-ocr-worker",
                support_count=max(1, int(jersey_summary.support_count)),
                confirmed=False,
            )
            if jersey_summary is not None and jersey_value is not None
            else None
        )
        resolver_people.append(
            RosterCanonicalPersonEvidence(
                canonical_person_id=canonical_id,
                visible_intervals=(),
                team=team_evidence,
                role=role_evidence,
                jersey_number=jersey_attribute,
                confirmed_external_player_id=document.get("externalPlayerId"),
                excluded_external_player_ids=tuple(
                    sorted(rejected_roster_candidate_ids(scene, canonical_id))
                ),
            )
        )

    result = resolve_closed_set_roster(resolver_people, persisted_players)
    resolutions = {
        resolution.canonical_person_id: resolution
        for resolution in result.people
    }
    for document in documents:
        canonical_id = str(document["canonicalPersonId"])
        if canonical_id in skipped_ids:
            continue
        resolution = resolutions[canonical_id]
        resolution_payload = resolution.to_payload()
        document["rosterResolution"] = {
            key: value
            for key, value in resolution_payload.items()
            if key != "candidates"
        }
        if resolution.status == "suggested":
            published_candidates = []
            for candidate in resolution.candidates:
                if not candidate.eligible or candidate.identity_signal_score <= 0.0:
                    continue
                payload = candidate.to_payload()
                # UI confidence is a hypothesis score, never a probability or
                # an accepted roster binding.
                payload["confidence"] = payload["score"]
                published_candidates.append(payload)
            document["rosterCandidates"] = published_candidates
        for code in resolution.conflicts:
            conflict_id = f"{canonical_id}:roster-resolution:{code}"
            if any(item.get("id") == conflict_id for item in document["conflicts"]):
                continue
            document["conflicts"].append(
                {
                    "id": conflict_id,
                    "code": code,
                    "message": (
                        "The closed-set roster resolver found contradictory identity evidence; "
                        "the existing manual decision was retained."
                    ),
                    "severity": "review",
                }
            )

    return {
        **result.to_payload()["diagnostics"],
        "status": "ready",
        "schemaVersion": 1,
        "matchClockAligned": False,
        "skippedExcludedIdentityCount": len(skipped_ids),
    }
