"""Manual must-link aggregation and canonical identity group materialization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .identity_resolution_contract import (
    GroupStatus,
    IdentityEdge,
    IdentityGroup,
    IdentityTracklet,
    normalize_reid_embedding,
)


@dataclass(frozen=True)
class IdentityNode:
    """A manually connected component used as one assignment vertex."""

    key: str
    members: tuple[IdentityTracklet, ...]
    start_time: float
    end_time: float
    team_id: str | None
    role: str | None
    external_player_id: str | None
    jersey_number: str | None
    jersey_confidence: float
    jersey_sample_count: int
    jersey_manual: bool
    mean_reid_embedding: tuple[float, ...] | None
    reid_embeddings: tuple[tuple[float, ...], ...]
    start_pitch: tuple[float, float] | None
    end_pitch: tuple[float, float] | None
    start_uncertainty_metres: float | None
    end_uncertainty_metres: float | None
    manual_identity_id: str | None
    manual_confirmed: bool
    manual_team: bool
    manual_role: bool
    excluded: bool
    conflict_reasons: tuple[str, ...]


def _one_or_none(values: Iterable[str | None]) -> tuple[str | None, bool]:
    unique = {value for value in values if value is not None}
    return (next(iter(unique)), False) if len(unique) == 1 else (None, len(unique) > 1)


def _node_from_members(key: str, members: Sequence[IdentityTracklet]) -> IdentityNode:
    ordered = tuple(sorted(members, key=lambda item: (item.start_time, item.end_time, item.id)))
    team_id, _ = _one_or_none(item.team_id for item in ordered)
    role, _ = _one_or_none(item.role for item in ordered)
    external_player_id, external_conflict = _one_or_none(
        item.external_player_id for item in ordered
    )
    manual_identity_id, manual_identity_conflict = _one_or_none(
        item.manual_identity_id for item in ordered
    )

    jersey_rows = [item for item in ordered if item.jersey_number is not None]
    jersey_number, jersey_conflict = _one_or_none(item.jersey_number for item in jersey_rows)
    jersey_confidence = max((item.jersey_confidence for item in jersey_rows), default=0.0)
    jersey_sample_count = sum(item.jersey_sample_count for item in jersey_rows)

    means = [item.mean_reid_embedding for item in ordered if item.mean_reid_embedding is not None]
    dimensions = {len(item) for item in means}
    samples = [sample for item in ordered for sample in item.reid_embeddings]
    dimensions.update(len(item) for item in samples)
    reid_conflict = len(dimensions) > 1
    if reid_conflict:
        mean_embedding = None
        normalized_samples: tuple[tuple[float, ...], ...] = ()
    else:
        weighted = []
        for item in ordered:
            if item.mean_reid_embedding is None:
                continue
            weighted.extend(
                [item.mean_reid_embedding] * max(1, min(20, item.observation_count))
            )
        mean_embedding = (
            normalize_reid_embedding(
                np.mean(np.asarray(weighted, dtype=np.float64), axis=0)
            )
            if weighted
            else None
        )
        normalized_samples = tuple(samples)

    earliest = min(ordered, key=lambda item: (item.start_time, item.id))
    latest = max(ordered, key=lambda item: (item.end_time, item.id))
    manual_team_values = {
        item.team_id for item in ordered if item.manual_team and item.team_id is not None
    }
    manual_role_values = {
        item.role for item in ordered if item.manual_role and item.role is not None
    }
    conflicts = tuple(
        reason
        for enabled, reason in (
            (len(manual_team_values) > 1, "manual-component-team-conflict"),
            (len(manual_role_values) > 1, "manual-component-role-conflict"),
            (external_conflict, "manual-component-external-player-conflict"),
            (jersey_conflict, "manual-component-jersey-conflict"),
            (manual_identity_conflict, "manual-component-id-conflict"),
            (reid_conflict, "manual-component-reid-dimension-conflict"),
        )
        if enabled
    )
    return IdentityNode(
        key=key,
        members=ordered,
        start_time=min(item.start_time for item in ordered),
        end_time=max(item.end_time for item in ordered),
        team_id=team_id,
        role=role,
        external_player_id=external_player_id,
        jersey_number=jersey_number,
        jersey_confidence=jersey_confidence,
        jersey_sample_count=jersey_sample_count,
        jersey_manual=any(item.manual_jersey for item in ordered),
        mean_reid_embedding=mean_embedding,
        reid_embeddings=normalized_samples,
        start_pitch=earliest.start_pitch,
        end_pitch=latest.end_pitch,
        start_uncertainty_metres=earliest.start_uncertainty_metres,
        end_uncertainty_metres=latest.end_uncertainty_metres,
        manual_identity_id=manual_identity_id,
        manual_confirmed=any(item.manual_confirmed for item in ordered),
        manual_team=bool(
            team_id is not None
            and any(item.manual_team and item.team_id == team_id for item in ordered)
        ),
        manual_role=bool(
            role is not None
            and any(item.manual_role and item.role == role for item in ordered)
        ),
        excluded=any(item.manual_excluded for item in ordered),
        conflict_reasons=conflicts,
    )


def build_identity_nodes(
    tracklets: Sequence[IdentityTracklet],
) -> tuple[list[IdentityNode], list[IdentityEdge]]:
    """Collapse manual must-links into assignment nodes and their audit edges."""

    components: dict[str, list[IdentityTracklet]] = defaultdict(list)
    for item in tracklets:
        key = (
            f"manual:{item.manual_identity_id}"
            if item.manual_identity_id is not None
            else f"tracklet:{item.id}"
        )
        components[key].append(item)

    nodes = [
        _node_from_members(key, members)
        for key, members in sorted(components.items(), key=lambda item: item[0])
    ]
    nodes.sort(key=lambda item: (item.start_time, item.end_time, item.key))
    manual_edges: list[IdentityEdge] = []
    for node in nodes:
        if node.manual_identity_id is None or len(node.members) < 2:
            continue
        ordered = sorted(node.members, key=lambda item: (item.start_time, item.end_time, item.id))
        for left, right in zip(ordered, ordered[1:]):
            manual_edges.append(
                IdentityEdge(
                    predecessor_id=left.id,
                    successor_id=right.id,
                    status="accepted",
                    score=1.0,
                    source="manual",
                    reasons=("manual-identity", *node.conflict_reasons),
                    gap_seconds=round(right.start_time - left.end_time, 6),
                )
            )
    return nodes, manual_edges


def identity_node_endpoint(
    node: IdentityNode,
    *,
    successor: bool,
) -> IdentityTracklet:
    if successor:
        return min(node.members, key=lambda item: (item.start_time, item.id))
    return max(node.members, key=lambda item: (item.end_time, item.id))


def _group_identifier(members: Sequence[IdentityTracklet]) -> str:
    manual_ids = sorted({item.manual_identity_id for item in members if item.manual_identity_id})
    if len(manual_ids) == 1:
        return f"identity:manual:{manual_ids[0]}"
    external_ids = sorted({item.external_player_id for item in members if item.external_player_id})
    if len(external_ids) == 1:
        return f"identity:external:{external_ids[0]}"
    return f"identity:{min(item.id for item in members)}"


def resolve_identity_group(
    members: Sequence[IdentityTracklet],
    accepted_edges: Sequence[IdentityEdge],
) -> IdentityGroup:
    """Materialize one canonical output group from a validated component."""

    ordered = tuple(sorted(members, key=lambda item: (item.start_time, item.end_time, item.id)))
    ids = {item.id for item in ordered}
    group_edges = [
        edge
        for edge in accepted_edges
        if edge.predecessor_id in ids and edge.successor_id in ids
    ]
    manual_ids = {item.manual_identity_id for item in ordered if item.manual_identity_id}
    external_ids = {item.external_player_id for item in ordered if item.external_player_id}
    team_ids = {item.team_id for item in ordered if item.team_id}
    roles = {item.role for item in ordered if item.role}
    jerseys = {item.jersey_number for item in ordered if item.jersey_number}
    excluded = any(item.manual_excluded for item in ordered)
    if excluded:
        status: GroupStatus = "excluded"
        source = "manual-excluded"
        confidence = 1.0
    elif any(item.manual_confirmed for item in ordered):
        status = "resolved"
        source = "manual"
        confidence = 1.0
    elif len(external_ids) == 1:
        status = "resolved"
        source = "external-player"
        confidence = 0.99
    elif any(edge.source == "automatic" for edge in group_edges):
        status = "resolved"
        source = "auto-stitch"
        confidence = min(
            float(edge.score or 0.0)
            for edge in group_edges
            if edge.source == "automatic"
        )
    else:
        status = "provisional"
        source = "local-tracklet"
        confidence = 0.0
    return IdentityGroup(
        id=_group_identifier(ordered),
        tracklet_ids=tuple(item.id for item in ordered),
        status=status,
        confidence=round(confidence, 6),
        source=source,
        team_id=next(iter(team_ids)) if len(team_ids) == 1 else None,
        role=next(iter(roles)) if len(roles) == 1 else None,
        external_player_id=next(iter(external_ids)) if len(external_ids) == 1 else None,
        jersey_number=next(iter(jerseys)) if len(jerseys) == 1 else None,
        manual_identity_id=next(iter(manual_ids)) if len(manual_ids) == 1 else None,
        observation_count=sum(item.observation_count for item in ordered),
    )


__all__ = [
    "IdentityNode",
    "build_identity_nodes",
    "identity_node_endpoint",
    "resolve_identity_group",
]
