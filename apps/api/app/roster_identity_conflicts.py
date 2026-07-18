"""Manual-binding reservations and cross-person roster conflicts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Mapping, Sequence

from .roster_identity_contract import (
    CanonicalPersonEvidence,
    RosterIdentityCandidate,
    RosterResolutionConflict,
)
from .roster_identity_temporal import intervals_overlap


@dataclass(frozen=True)
class ConfirmedBindingConflicts:
    reserved_player_ids: frozenset[str]
    conflicts: tuple[RosterResolutionConflict, ...]
    codes_by_person: Mapping[str, tuple[str, ...]]


def enforce_manual_binding_reservations(
    person: CanonicalPersonEvidence,
    candidate: RosterIdentityCandidate,
    reserved_player_ids: frozenset[str],
) -> RosterIdentityCandidate:
    """Block automatic edges that contradict or reuse a manual binding."""

    player_id = candidate.external_player_id
    if person.confirmed_external_player_id == player_id:
        return candidate
    conflict: str | None = None
    if person.confirmed_external_player_id is not None:
        conflict = "different-player-manually-confirmed"
    elif player_id in reserved_player_ids:
        conflict = "player-reserved-by-confirmed-binding"
    if conflict is None:
        return candidate
    return replace(
        candidate,
        conflicts=tuple(dict.fromkeys((*candidate.conflicts, conflict))),
        eligible=False,
        proposal_status="blocked",
    )


def analyze_confirmed_binding_conflicts(
    people: Sequence[CanonicalPersonEvidence],
) -> ConfirmedBindingConflicts:
    """Retain manual facts while making one-to-one violations observable."""

    confirmed_owners: dict[str, list[CanonicalPersonEvidence]] = defaultdict(list)
    for person in people:
        if person.confirmed_external_player_id is not None:
            confirmed_owners[person.confirmed_external_player_id].append(person)

    conflicts: list[RosterResolutionConflict] = []
    codes_by_person: dict[str, list[str]] = defaultdict(list)
    for external_id, owners in sorted(confirmed_owners.items()):
        if len(owners) <= 1:
            continue
        owner_ids = tuple(sorted(item.canonical_person_id for item in owners))
        conflicts.append(
            RosterResolutionConflict(
                code="duplicate-confirmed-player-binding",
                message=(
                    "The same real player is manually bound to multiple canonical "
                    "identities; the bindings were retained and require review."
                ),
                canonical_person_ids=owner_ids,
                external_player_id=external_id,
            )
        )
        for owner_id in owner_ids:
            codes_by_person[owner_id].append("duplicate-confirmed-player-binding")

        for left, right in combinations(owners, 2):
            if not intervals_overlap(left.visible_intervals, right.visible_intervals):
                continue
            simultaneous_ids = tuple(
                sorted((left.canonical_person_id, right.canonical_person_id))
            )
            conflicts.append(
                RosterResolutionConflict(
                    code="simultaneous-confirmed-player-duplicate",
                    message=(
                        "Two simultaneously visible identities carry the same "
                        "confirmed real player. Manual bindings remain authoritative "
                        "but invalid."
                    ),
                    canonical_person_ids=simultaneous_ids,
                    external_player_id=external_id,
                )
            )
            for owner_id in simultaneous_ids:
                codes_by_person[owner_id].append(
                    "simultaneous-confirmed-player-duplicate"
                )

    return ConfirmedBindingConflicts(
        reserved_player_ids=frozenset(confirmed_owners),
        conflicts=tuple(conflicts),
        codes_by_person={
            person_id: tuple(dict.fromkeys(codes))
            for person_id, codes in codes_by_person.items()
        },
    )
