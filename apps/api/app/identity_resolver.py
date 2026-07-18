"""Offline canonical identity resolution orchestration.

The online tracker stays local and conservative.  This application service
coordinates manual components, pairwise evidence scoring, fail-closed global
assignment, transitive validation, and result diagnostics.  Each algorithm is
owned by its dedicated module so this composition layer has one reason to
change: the resolution workflow.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from . import identity_assignment
from . import identity_pairwise_scoring
from . import identity_resolution_components
from . import identity_resolution_contract as contract
from . import identity_resolution_diagnostics


def _edge_sort_key(edge: contract.IdentityEdge) -> tuple:
    return (edge.predecessor_id, edge.successor_id, edge.source, edge.reasons)


def resolve_identities(
    tracklets: Sequence[contract.IdentityTracklet],
    config: contract.IdentityResolverConfig | None = None,
) -> contract.IdentityResolution:
    """Resolve local tracklets into deterministic canonical identity groups."""

    config = config or contract.IdentityResolverConfig()
    ordered = tuple(sorted(tracklets, key=lambda item: (item.start_time, item.end_time, item.id)))
    if len({item.id for item in ordered}) != len(ordered):
        raise ValueError("IdentityTracklet ids must be unique")
    if not ordered:
        return contract.IdentityResolution(
            groups=(),
            accepted_edges=(),
            review_edges=(),
            rejected_edges=(),
            diagnostics=identity_resolution_diagnostics.empty_identity_diagnostics(),
            tracklet_to_identity={},
        )

    nodes, manual_edges = identity_resolution_components.build_identity_nodes(ordered)
    candidates, rejected = identity_pairwise_scoring.pair_identity_nodes(nodes, config)
    candidates = identity_assignment.mark_ambiguous_identity_edges(candidates, config)
    auto_accepted, review = identity_assignment.solve_identity_assignment(
        nodes,
        candidates,
        config,
    )
    auto_accepted, transitive_review, root_by_tracklet = (
        identity_assignment.filter_transitive_identity_conflicts(
            ordered,
            manual_edges,
            auto_accepted,
            config,
        )
    )
    review.extend(transitive_review)
    accepted = [*manual_edges, *auto_accepted]

    members_by_root: dict[str, list[contract.IdentityTracklet]] = defaultdict(list)
    for item in ordered:
        members_by_root[root_by_tracklet[item.id]].append(item)
    groups = [
        identity_resolution_components.resolve_identity_group(members, accepted)
        for _, members in sorted(
            members_by_root.items(),
            key=lambda item: min(
                (member.start_time, member.end_time, member.id) for member in item[1]
            ),
        )
    ]
    groups.sort(
        key=lambda group: min(
            (item.start_time, item.end_time, item.id)
            for item in ordered
            if item.id in group.tracklet_ids
        )
    )
    tracklet_to_identity = {
        tracklet_id: group.id
        for group in groups
        for tracklet_id in group.tracklet_ids
    }

    accepted = sorted(accepted, key=_edge_sort_key)
    review = sorted(review, key=_edge_sort_key)
    rejected = sorted(rejected, key=_edge_sort_key)
    diagnostics = identity_resolution_diagnostics.build_identity_diagnostics(
        tracklets=ordered,
        nodes=nodes,
        groups=groups,
        accepted=accepted,
        review=review,
        rejected=rejected,
        tracklet_to_identity=tracklet_to_identity,
        config=config,
    )
    return contract.IdentityResolution(
        groups=tuple(groups),
        accepted_edges=tuple(accepted),
        review_edges=tuple(review),
        rejected_edges=tuple(rejected),
        diagnostics=diagnostics,
        tracklet_to_identity=tracklet_to_identity,
    )


__all__ = ["resolve_identities"]
