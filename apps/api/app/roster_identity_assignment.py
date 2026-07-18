"""Fail-closed one-to-one assignment for scored roster hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .roster_identity_contract import (
    CanonicalPersonEvidence,
    PersistedRosterPlayer,
    RosterIdentityCandidate,
    RosterResolverConfig,
)


@dataclass(frozen=True)
class GlobalRosterAssignment:
    selected_player_by_person: Mapping[str, str]
    ambiguous_edges: frozenset[tuple[str, str]]
    objective: float


def solve_global_roster_assignment(
    people: Sequence[CanonicalPersonEvidence],
    players: Sequence[PersistedRosterPlayer],
    candidates: Mapping[str, Mapping[str, RosterIdentityCandidate]],
    config: RosterResolverConfig,
) -> GlobalRosterAssignment:
    """Select unique suggestions while retaining an explicit abstain option."""

    if not people:
        return GlobalRosterAssignment({}, frozenset(), 0.0)

    player_columns = {item.external_player_id: index for index, item in enumerate(players)}
    column_count = len(players) + len(people)
    forbidden = -1_000_000.0
    utility = np.full((len(people), column_count), forbidden, dtype=np.float64)
    for row, person in enumerate(people):
        for player_id, candidate in candidates[person.canonical_person_id].items():
            column = player_columns[player_id]
            if (
                candidate.eligible
                and candidate.score >= config.min_candidate_score
                and candidate.identity_signal_score >= config.min_identity_signal_score
            ):
                utility[row, column] = candidate.score
        utility[row, len(players) :] = max(
            0.0, config.min_candidate_score - config.assignment_margin
        )

    rows, columns = linear_sum_assignment(-utility)
    objective = float(sum(utility[row, column] for row, column in zip(rows, columns)))
    selected: dict[str, str] = {}
    selected_edges: list[tuple[int, int]] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= len(players) or utility[row, column] < config.min_candidate_score:
            continue
        selected[people[row].canonical_person_id] = players[column].external_player_id
        selected_edges.append((row, column))

    ambiguous: set[tuple[str, str]] = set()
    for selected_row, selected_column in selected_edges:
        alternative = utility.copy()
        alternative[selected_row, selected_column] = forbidden
        alternative_rows, alternative_columns = linear_sum_assignment(-alternative)
        alternative_objective = float(
            sum(
                alternative[row, column]
                for row, column in zip(alternative_rows, alternative_columns)
            )
        )
        # A tie must never be converted into an identity decision, including
        # when experiments configure an assignment margin of zero.
        if objective - alternative_objective <= max(config.assignment_margin, 1e-9):
            ambiguous.add(
                (
                    people[selected_row].canonical_person_id,
                    players[selected_column].external_player_id,
                )
            )

    return GlobalRosterAssignment(selected, frozenset(ambiguous), objective)

