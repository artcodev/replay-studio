"""Fail-closed global assignment and transitive component validation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .identity_resolution_components import IdentityNode
from .identity_resolution_contract import (
    IdentityEdge,
    IdentityResolverConfig,
    IdentityTracklet,
)


def mark_ambiguous_identity_edges(
    candidates: list[tuple[int, int, IdentityEdge]],
    config: IdentityResolverConfig,
) -> list[tuple[int, int, IdentityEdge]]:
    """Demote competing edges whose evidence does not clear the margin."""

    def ambiguity_rank(edge: IdentityEdge) -> float:
        gap_penalty = min(
            config.ambiguity_gap_penalty_cap,
            max(0.0, float(edge.gap_seconds or 0.0))
            * config.ambiguity_gap_penalty_per_second,
        )
        return float(edge.score or 0.0) - gap_penalty

    ambiguous: dict[int, set[str]] = defaultdict(set)
    for coordinate, reason in ((0, "ambiguous-successor"), (1, "ambiguous-predecessor")):
        grouped: dict[int, list[int]] = defaultdict(list)
        for edge_index, candidate in enumerate(candidates):
            grouped[candidate[coordinate]].append(edge_index)
        for edge_indices in grouped.values():
            ranked = sorted(
                edge_indices,
                key=lambda index: (
                    -ambiguity_rank(candidates[index][2]),
                    candidates[index][2].predecessor_id,
                    candidates[index][2].successor_id,
                ),
            )
            if len(ranked) < 2:
                continue
            best_score = ambiguity_rank(candidates[ranked[0]][2])
            second_score = ambiguity_rank(candidates[ranked[1]][2])
            if best_score - second_score >= config.ambiguity_margin:
                continue
            for edge_index in ranked:
                score = ambiguity_rank(candidates[edge_index][2])
                if best_score - score < config.ambiguity_margin:
                    ambiguous[edge_index].add(reason)

    result = []
    for index, (predecessor, successor, edge) in enumerate(candidates):
        reasons = ambiguous.get(index)
        if reasons and edge.status == "accepted":
            edge = replace(
                edge,
                status="review",
                reasons=(*edge.reasons, *sorted(reasons)),
            )
        result.append((predecessor, successor, edge))
    return result


def solve_identity_assignment(
    nodes: Sequence[IdentityNode],
    candidates: list[tuple[int, int, IdentityEdge]],
    config: IdentityResolverConfig,
) -> tuple[list[IdentityEdge], list[IdentityEdge]]:
    """Select at most one predecessor and successor for each identity node."""

    accepted_candidates = [item for item in candidates if item[2].status == "accepted"]
    review = [item[2] for item in candidates if item[2].status == "review"]
    eligible_indices = [index for index, node in enumerate(nodes) if not node.excluded]
    if not eligible_indices or not accepted_candidates:
        return [], review

    compact_index = {node_index: compact for compact, node_index in enumerate(eligible_indices)}
    count = len(eligible_indices)
    large = 1e6
    costs = np.full((count, count * 2), large, dtype=np.float64)
    edge_by_cell: dict[tuple[int, int], IdentityEdge] = {}
    for predecessor, successor, edge in accepted_candidates:
        row, column = compact_index[predecessor], compact_index[successor]
        costs[row, column] = 1.0 - float(edge.score or 0.0) + (row * count + column) * 1e-12
        edge_by_cell[(row, column)] = edge
    no_link_cost = 1.0 - config.accept_score + 1e-6
    for row in range(count):
        costs[row, count + row] = no_link_cost + row * 1e-12

    rows, columns = linear_sum_assignment(costs)
    selected_cells = {
        (row, column)
        for row, column in zip(rows.tolist(), columns.tolist())
        if column < count and (row, column) in edge_by_cell
    }
    accepted = [edge_by_cell[cell] for cell in sorted(selected_cells)]
    selected_ids = {
        (edge.predecessor_id, edge.successor_id) for edge in accepted
    }
    for _, _, edge in accepted_candidates:
        if (edge.predecessor_id, edge.successor_id) in selected_ids:
            continue
        review.append(
            replace(
                edge,
                status="review",
                reasons=(*edge.reasons, "not-selected-by-global-assignment"),
            )
        )
    return accepted, review


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _reid_dimensions(members: Sequence[IdentityTracklet]) -> set[int]:
    dimensions: set[int] = set()
    for item in members:
        if item.mean_reid_embedding is not None:
            dimensions.add(len(item.mean_reid_embedding))
        dimensions.update(len(sample) for sample in item.reid_embeddings)
    return dimensions


def _reliable_component_jerseys(
    members: Sequence[IdentityTracklet],
    config: IdentityResolverConfig,
) -> set[str]:
    evidence: dict[str, list[IdentityTracklet]] = defaultdict(list)
    for item in members:
        if item.jersey_number is not None:
            evidence[item.jersey_number].append(item)
    return {
        jersey
        for jersey, rows in evidence.items()
        if any(item.manual_jersey for item in rows)
        or (
            max((item.jersey_confidence for item in rows), default=0.0)
            >= config.reliable_jersey_confidence
            and sum(item.jersey_sample_count for item in rows)
            >= config.reliable_jersey_samples
        )
    }


def _transitive_component_conflicts(
    left: Sequence[IdentityTracklet],
    right: Sequence[IdentityTracklet],
    config: IdentityResolverConfig,
) -> tuple[str, ...]:
    """Validate all hard evidence after joining two compatible components."""

    members = (*left, *right)
    manual_identity_ids = {
        item.manual_identity_id for item in members if item.manual_identity_id
    }
    external_player_ids = {
        item.external_player_id for item in members if item.external_player_id
    }
    manual_teams = {
        item.team_id
        for item in members
        if item.manual_team and item.team_id is not None
    }
    manual_roles = {
        item.role
        for item in members
        if item.manual_role and item.role is not None
    }
    reliable_jerseys = _reliable_component_jerseys(members, config)
    reid_dimensions = _reid_dimensions(members)
    return tuple(
        reason
        for conflict, reason in (
            (
                len(manual_identity_ids) > 1,
                "transitive-manual-identity-conflict",
            ),
            (
                len(external_player_ids) > 1,
                "transitive-external-player-conflict",
            ),
            (len(reliable_jerseys) > 1, "transitive-jersey-conflict"),
            (len(manual_teams) > 1, "transitive-team-conflict"),
            (len(manual_roles) > 1, "transitive-role-conflict"),
            (
                len(reid_dimensions) > 1,
                "transitive-reid-dimension-conflict",
            ),
        )
        if conflict
    )


def filter_transitive_identity_conflicts(
    tracklets: Sequence[IdentityTracklet],
    manual_edges: Sequence[IdentityEdge],
    automatic_edges: Sequence[IdentityEdge],
    config: IdentityResolverConfig,
) -> tuple[list[IdentityEdge], list[IdentityEdge], dict[str, str]]:
    """Build components without allowing an unlabeled transitive bridge."""

    union = _UnionFind(item.id for item in tracklets)
    members_by_root: dict[str, list[IdentityTracklet]] = {
        item.id: [item] for item in tracklets
    }

    def join(left_id: str, right_id: str) -> None:
        left_root, right_root = union.find(left_id), union.find(right_id)
        if left_root == right_root:
            return
        combined = [*members_by_root[left_root], *members_by_root[right_root]]
        union.union(left_root, right_root)
        root = union.find(left_root)
        obsolete = right_root if root == left_root else left_root
        members_by_root[root] = combined
        members_by_root.pop(obsolete, None)

    for edge in manual_edges:
        join(edge.predecessor_id, edge.successor_id)

    accepted: list[IdentityEdge] = []
    demoted: list[IdentityEdge] = []
    ranked_edges = sorted(
        automatic_edges,
        key=lambda edge: (
            -float(edge.score or 0.0),
            edge.predecessor_id,
            edge.successor_id,
            edge.source,
        ),
    )
    for edge in ranked_edges:
        left_root = union.find(edge.predecessor_id)
        right_root = union.find(edge.successor_id)
        if left_root == right_root:
            accepted.append(edge)
            continue
        conflicts = _transitive_component_conflicts(
            members_by_root[left_root],
            members_by_root[right_root],
            config,
        )
        if conflicts:
            demoted.append(
                replace(
                    edge,
                    status="review",
                    reasons=(*edge.reasons, *conflicts),
                )
            )
            continue
        join(edge.predecessor_id, edge.successor_id)
        accepted.append(edge)
    root_by_tracklet = {item.id: union.find(item.id) for item in tracklets}
    return accepted, demoted, root_by_tracklet


__all__ = [
    "filter_transitive_identity_conflicts",
    "mark_ambiguous_identity_edges",
    "solve_identity_assignment",
]
