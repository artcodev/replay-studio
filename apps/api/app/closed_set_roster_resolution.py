"""Application orchestration for review-only closed-set roster identity."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .roster_identity_assignment import solve_global_roster_assignment
from .roster_identity_conflicts import (
    ConfirmedBindingConflicts,
    analyze_confirmed_binding_conflicts,
    enforce_manual_binding_reservations,
)
from .roster_identity_contract import (
    CanonicalPersonEvidence,
    ClosedSetRosterResolution,
    PersistedRosterPlayer,
    PersonRosterResolution,
    RosterIdentityCandidate,
    RosterResolverConfig,
)
from .roster_identity_diagnostics import build_roster_resolution_diagnostics
from .roster_identity_scoring import (
    candidate_evidence,
    evaluate_roster_candidate,
    rank_roster_candidates,
)


def _validate_unique_inputs(
    people: tuple[CanonicalPersonEvidence, ...],
    players: tuple[PersistedRosterPlayer, ...],
) -> None:
    person_ids = [item.canonical_person_id for item in people]
    player_ids = [item.external_player_id for item in players]
    if len(person_ids) != len(set(person_ids)):
        raise ValueError("canonical_person_id values must be unique")
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("persisted external_player_id values must be unique")


def _score_candidate_matrix(
    people: tuple[CanonicalPersonEvidence, ...],
    players: tuple[PersistedRosterPlayer, ...],
    config: RosterResolverConfig,
    binding_conflicts: ConfirmedBindingConflicts,
) -> dict[str, dict[str, RosterIdentityCandidate]]:
    matrix: dict[str, dict[str, RosterIdentityCandidate]] = {}
    for person in people:
        rows: dict[str, RosterIdentityCandidate] = {}
        for player in players:
            candidate = evaluate_roster_candidate(
                person,
                player,
                config,
                confirmed_binding=(
                    person.confirmed_external_player_id == player.external_player_id
                ),
            )
            rows[player.external_player_id] = enforce_manual_binding_reservations(
                person, candidate, binding_conflicts.reserved_player_ids
            )
        matrix[person.canonical_person_id] = rows
    return matrix


def _confirmed_resolution(
    person: CanonicalPersonEvidence,
    rows: dict[str, RosterIdentityCandidate],
    player_ids: frozenset[str],
    person_conflicts: tuple[str, ...],
    config: RosterResolverConfig,
) -> PersonRosterResolution:
    external_id = person.confirmed_external_player_id
    assert external_id is not None
    conflicts = list(person_conflicts)
    if external_id not in player_ids:
        conflicts.append("confirmed-player-missing-from-persisted-roster")
        missing = RosterIdentityCandidate(
            external_player_id=external_id,
            display_name=external_id,
            team_id=None,
            jersey_number=None,
            role=None,
            score=1.0,
            identity_signal_score=1.0,
            evidence=(
                candidate_evidence(
                    "manual-confirmed-binding",
                    1.0,
                    1.0,
                    "manual-roster-binding",
                ),
            ),
            reasons=("manual-confirmed-binding-authoritative",),
            conflicts=("confirmed-player-missing-from-persisted-roster",),
            eligible=False,
            proposal_status="confirmed",
        )
        rows = {external_id: missing, **rows}
    bound_candidate = rows.get(external_id)
    if bound_candidate is not None:
        conflicts.extend(bound_candidate.conflicts)
    return PersonRosterResolution(
        canonical_person_id=person.canonical_person_id,
        status="confirmed",
        confirmed_external_player_id=external_id,
        suggested_external_player_id=None,
        candidates=rank_roster_candidates(
            rows.values(), limit=config.candidate_limit, force_ids=(external_id,)
        ),
        reasons=("manual-binding-retained",),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


def _review_resolution(
    person: CanonicalPersonEvidence,
    rows: dict[str, RosterIdentityCandidate],
    players: tuple[PersistedRosterPlayer, ...],
    binding_conflicts: ConfirmedBindingConflicts,
    selected_player_id: str | None,
    ambiguous_edges: frozenset[tuple[str, str]],
    config: RosterResolverConfig,
) -> PersonRosterResolution:
    reasons: list[str] = []
    if selected_player_id is not None:
        candidate = rows[selected_player_id]
        if (person.canonical_person_id, selected_player_id) in ambiguous_edges:
            rows[selected_player_id] = replace(
                candidate,
                proposal_status="ambiguous",
                reasons=tuple(
                    (*candidate.reasons, "global-assignment-margin-too-small")
                ),
            )
            selected_player_id = None
            reasons.append("global-assignment-ambiguous")
        else:
            rows[selected_player_id] = replace(candidate, proposal_status="selected")
            reasons.extend(
                ("globally-unique-roster-suggestion", "manual-confirmation-required")
            )

    if selected_player_id is None and not reasons:
        eligible = [
            item
            for item in rows.values()
            if item.eligible
            and item.external_player_id not in binding_conflicts.reserved_player_ids
            and item.score >= config.min_candidate_score
        ]
        identity_eligible = [
            item
            for item in eligible
            if item.identity_signal_score >= config.min_identity_signal_score
        ]
        if not players:
            reasons.append("persisted-roster-empty")
        elif not identity_eligible:
            reasons.append("insufficient-identity-evidence")
        else:
            reasons.append("global-one-to-one-abstain")

    return PersonRosterResolution(
        canonical_person_id=person.canonical_person_id,
        status="suggested" if selected_player_id is not None else "abstain",
        confirmed_external_player_id=None,
        suggested_external_player_id=selected_player_id,
        candidates=rank_roster_candidates(
            rows.values(),
            limit=config.candidate_limit,
            force_ids=(selected_player_id,) if selected_player_id is not None else (),
        ),
        reasons=tuple(dict.fromkeys(reasons)),
        conflicts=binding_conflicts.codes_by_person.get(
            person.canonical_person_id, ()
        ),
    )


def resolve_closed_set_roster(
    canonical_people: Iterable[CanonicalPersonEvidence],
    persisted_players: Iterable[PersistedRosterPlayer],
    config: RosterResolverConfig | None = None,
) -> ClosedSetRosterResolution:
    """Rank unique roster candidates without accepting any binding.

    Suggestions are UI review hints. Durable identity remains an explicit
    manual decision; this capability never emits an automatic binding.
    """

    policy = config or RosterResolverConfig()
    people = tuple(
        sorted(tuple(canonical_people), key=lambda item: item.canonical_person_id)
    )
    players = tuple(
        sorted(tuple(persisted_players), key=lambda item: item.external_player_id)
    )
    _validate_unique_inputs(people, players)

    binding_conflicts = analyze_confirmed_binding_conflicts(people)
    candidate_matrix = _score_candidate_matrix(
        people, players, policy, binding_conflicts
    )
    unconfirmed_people = tuple(
        item for item in people if item.confirmed_external_player_id is None
    )
    unreserved_players = tuple(
        item
        for item in players
        if item.external_player_id not in binding_conflicts.reserved_player_ids
    )
    assignment = solve_global_roster_assignment(
        unconfirmed_people,
        unreserved_players,
        {
            person.canonical_person_id: {
                player.external_player_id: candidate_matrix[
                    person.canonical_person_id
                ][player.external_player_id]
                for player in unreserved_players
            }
            for person in unconfirmed_people
        },
        policy,
    )

    player_ids = frozenset(item.external_player_id for item in players)
    resolutions: list[PersonRosterResolution] = []
    for person in people:
        rows = candidate_matrix[person.canonical_person_id]
        person_conflicts = binding_conflicts.codes_by_person.get(
            person.canonical_person_id, ()
        )
        if person.confirmed_external_player_id is not None:
            resolution = _confirmed_resolution(
                person, rows, player_ids, person_conflicts, policy
            )
        else:
            resolution = _review_resolution(
                person,
                rows,
                players,
                binding_conflicts,
                assignment.selected_player_by_person.get(person.canonical_person_id),
                assignment.ambiguous_edges,
                policy,
            )
        resolutions.append(resolution)

    diagnostics = build_roster_resolution_diagnostics(
        people,
        players,
        resolutions,
        binding_conflicts,
        assignment.objective,
    )
    return ClosedSetRosterResolution(
        people=tuple(resolutions),
        conflicts=binding_conflicts.conflicts,
        diagnostics=diagnostics,
    )

