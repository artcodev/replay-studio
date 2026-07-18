"""Observable quality diagnostics for roster identity review."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .roster_identity_contract import (
    CanonicalPersonEvidence,
    PersistedRosterPlayer,
    PersonRosterResolution,
)
from .roster_identity_conflicts import ConfirmedBindingConflicts


def build_roster_resolution_diagnostics(
    people: Sequence[CanonicalPersonEvidence],
    players: Sequence[PersistedRosterPlayer],
    resolutions: Sequence[PersonRosterResolution],
    binding_conflicts: ConfirmedBindingConflicts,
    assignment_objective: float,
) -> dict[str, object]:
    conflict_codes = Counter(
        conflict for resolution in resolutions for conflict in resolution.conflicts
    )
    candidate_conflict_codes = Counter(
        conflict
        for resolution in resolutions
        for candidate in resolution.candidates
        for conflict in candidate.conflicts
    )
    suggestion_ids = {
        item.suggested_external_player_id
        for item in resolutions
        if item.suggested_external_player_id is not None
    }
    suggestion_count = sum(item.status == "suggested" for item in resolutions)
    return {
        "canonicalPersonCount": len(people),
        "persistedRosterPlayerCount": len(players),
        "confirmedBindingCount": sum(
            item.status == "confirmed" for item in resolutions
        ),
        "suggestionCount": suggestion_count,
        "abstainCount": sum(item.status == "abstain" for item in resolutions),
        "confirmedBindingConflictCount": sum(
            item.status == "confirmed" and bool(item.conflicts)
            for item in resolutions
        ),
        "globalConflictCount": len(binding_conflicts.conflicts),
        "conflictCounts": dict(sorted(conflict_codes.items())),
        "candidateConflictCounts": dict(sorted(candidate_conflict_codes.items())),
        "reservedExternalPlayerIds": sorted(binding_conflicts.reserved_player_ids),
        "assignmentObjective": round(assignment_objective, 6),
        "oneToOneSuggestions": len(suggestion_ids) == suggestion_count,
        "automaticBindingCount": 0,
        "requiresManualConfirmation": True,
    }

