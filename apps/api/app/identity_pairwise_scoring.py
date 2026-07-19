"""Pairwise evidence scoring for identity tracklet components.

This module proposes candidate edges only.  It does not choose a global
assignment or mutate identity components.
"""

from __future__ import annotations

from math import hypot
from typing import Sequence

import numpy as np

from .identity_resolution_components import IdentityNode, identity_node_endpoint
from .identity_resolution_contract import (
    EdgeStatus,
    IdentityEdge,
    IdentityResolverConfig,
)


def _reliable_jersey(node: IdentityNode, config: IdentityResolverConfig) -> bool:
    return bool(
        node.jersey_number is not None
        and (
            node.jersey_manual
            or (
                node.jersey_confidence >= config.reliable_jersey_confidence
                and node.jersey_sample_count >= config.reliable_jersey_samples
            )
        )
    )


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        return None
    return max(0.0, min(2.0, 1.0 - float(np.dot(left, right))))


def _reid_distances(
    left: IdentityNode,
    right: IdentityNode,
    strong_threshold: float,
) -> tuple[float | None, float | None, float | None, int, int]:
    """Return mean, optimistic and robust bidirectional ReID evidence."""

    mean_distance = None
    if left.mean_reid_embedding is not None and right.mean_reid_embedding is not None:
        mean_distance = _cosine_distance(left.mean_reid_embedding, right.mean_reid_embedding)
    matrix = [
        [
            _cosine_distance(left_sample, right_sample)
            for right_sample in right.reid_embeddings
        ]
        for left_sample in left.reid_embeddings
    ]
    finite_rows = [
        [distance for distance in row if distance is not None]
        for row in matrix
    ]
    finite_rows = [row for row in finite_rows if row]
    if not finite_rows:
        return mean_distance, None, None, 0, 0
    best = min(distance for row in finite_rows for distance in row)
    left_nearest = [min(row) for row in finite_rows]
    column_count = min(len(row) for row in finite_rows)
    right_nearest = [
        min(row[column] for row in finite_rows)
        for column in range(column_count)
    ]
    robust = max(float(np.median(left_nearest)), float(np.median(right_nearest)))
    return (
        mean_distance,
        best,
        robust,
        sum(distance <= strong_threshold for distance in left_nearest),
        sum(distance <= strong_threshold for distance in right_nearest),
    )


def _candidate_edge(
    predecessor: IdentityNode,
    successor: IdentityNode,
    config: IdentityResolverConfig,
) -> IdentityEdge:
    predecessor_endpoint = identity_node_endpoint(predecessor, successor=False)
    successor_endpoint = identity_node_endpoint(successor, successor=True)
    gap = successor.start_time - predecessor.end_time
    hard_reasons: list[str] = []
    soft_reasons: list[str] = []

    if gap <= config.temporal_epsilon_seconds:
        hard_reasons.append("temporal-overlap")
    if predecessor.conflict_reasons:
        hard_reasons.extend(predecessor.conflict_reasons)
    if successor.conflict_reasons:
        hard_reasons.extend(successor.conflict_reasons)
    if (
        predecessor.manual_identity_id is not None
        and successor.manual_identity_id is not None
        and predecessor.manual_identity_id != successor.manual_identity_id
    ):
        hard_reasons.append("manual-identity-conflict")
    if predecessor.team_id and successor.team_id and predecessor.team_id != successor.team_id:
        if predecessor.manual_team and successor.manual_team:
            hard_reasons.append("team-conflict")
        else:
            soft_reasons.append("automatic-team-disagreement")
    if predecessor.role and successor.role and predecessor.role != successor.role:
        if predecessor.manual_role and successor.manual_role:
            hard_reasons.append("role-conflict")
        else:
            soft_reasons.append("automatic-role-disagreement")
    if (
        predecessor.external_player_id
        and successor.external_player_id
        and predecessor.external_player_id != successor.external_player_id
    ):
        hard_reasons.append("external-player-conflict")

    predecessor_jersey_reliable = _reliable_jersey(predecessor, config)
    successor_jersey_reliable = _reliable_jersey(successor, config)
    if (
        predecessor_jersey_reliable
        and successor_jersey_reliable
        and predecessor.jersey_number != successor.jersey_number
    ):
        hard_reasons.append("jersey-conflict")

    pitch_distance = None
    reachable_distance = None
    if predecessor.end_pitch is not None and successor.start_pitch is not None and gap > 0.0:
        pitch_distance = hypot(
            successor.start_pitch[0] - predecessor.end_pitch[0],
            successor.start_pitch[1] - predecessor.end_pitch[1],
        )
        uncertainty = float(predecessor.end_uncertainty_metres or 0.0) + float(
            successor.start_uncertainty_metres or 0.0
        )
        reachable_distance = (
            config.max_player_speed_metres_per_second * gap
            + config.motion_slack_metres
            + uncertainty
        )
        if pitch_distance > reachable_distance:
            hard_reasons.append("physically-impossible-transition")

    if hard_reasons:
        # The pair is rejected unconditionally, so the O(samples²) ReID
        # matrix is never computed for it. Most pairs in a real clip end
        # here (temporal overlap, team conflict, impossible transition).
        return IdentityEdge(
            predecessor_id=predecessor_endpoint.id,
            successor_id=successor_endpoint.id,
            status="rejected",
            score=None,
            source="constraints",
            reasons=tuple(dict.fromkeys(hard_reasons)),
            gap_seconds=round(gap, 6),
            pitch_distance_metres=pitch_distance,
            reachable_distance_metres=reachable_distance,
        )

    (
        mean_distance,
        sample_distance,
        robust_sample_distance,
        strong_support_left,
        strong_support_right,
    ) = _reid_distances(
        predecessor,
        successor,
        config.strong_sample_reid_distance,
    )

    external_match = bool(
        predecessor.external_player_id
        and predecessor.external_player_id == successor.external_player_id
    )
    jersey_match = bool(
        predecessor_jersey_reliable
        and successor_jersey_reliable
        and predecessor.jersey_number == successor.jersey_number
    )
    weak_jersey_match = bool(
        predecessor.jersey_number
        and predecessor.jersey_number == successor.jersey_number
        and not jersey_match
    )
    strong_reid = bool(
        robust_sample_distance is not None
        and len(predecessor.reid_embeddings) >= config.min_strong_reid_samples
        and len(successor.reid_embeddings) >= config.min_strong_reid_samples
        and strong_support_left >= config.min_strong_reid_samples
        and strong_support_right >= config.min_strong_reid_samples
        and robust_sample_distance <= config.strong_sample_reid_distance
        and (mean_distance is None or mean_distance <= config.review_reid_distance)
    )
    review_reid = bool(
        not strong_reid
        and (
            (mean_distance is not None and mean_distance <= config.review_reid_distance)
            or (
                sample_distance is not None
                and sample_distance <= config.strong_reid_distance
            )
        )
    )

    signals: list[tuple[str, float]] = []
    if external_match:
        signals.append(("external-player-match", 0.995))
    if jersey_match:
        jersey_score = 0.94 + min(
            0.035,
            max(0.0, min(predecessor.jersey_confidence, successor.jersey_confidence) - 0.8)
            * 0.175,
        )
        signals.append(("reliable-jersey-match", jersey_score))
    if strong_reid:
        distances = [
            item
            for item in (mean_distance, robust_sample_distance)
            if item is not None
        ]
        signals.append(("strong-reid", 1.0 - max(distances)))
    elif review_reid:
        distances = [item for item in (mean_distance, sample_distance) if item is not None]
        signals.append(("review-reid", 1.0 - min(distances)))
    if weak_jersey_match:
        signals.append(("low-confidence-jersey-match", 0.65))

    if not signals:
        return IdentityEdge(
            predecessor_id=predecessor_endpoint.id,
            successor_id=successor_endpoint.id,
            status="rejected",
            score=0.0,
            source="identity-evidence",
            reasons=("insufficient-identity-evidence",),
            gap_seconds=round(gap, 6),
            reid_mean_distance=mean_distance,
            reid_best_sample_distance=sample_distance,
            reid_robust_sample_distance=robust_sample_distance,
            reid_strong_support_left=strong_support_left,
            reid_strong_support_right=strong_support_right,
            pitch_distance_metres=pitch_distance,
            reachable_distance_metres=reachable_distance,
        )

    strong_identity = external_match or jersey_match or strong_reid
    score = max(value for _, value in signals)
    strong_signal_count = sum(
        reason in {"external-player-match", "reliable-jersey-match", "strong-reid"}
        for reason, _ in signals
    )
    if strong_signal_count > 1:
        score += min(0.04, (strong_signal_count - 1) * 0.02)
    if pitch_distance is not None and reachable_distance is not None:
        feasibility = max(0.0, 1.0 - pitch_distance / max(1e-6, reachable_distance))
        score += feasibility * 0.015
    score -= 0.08 * len(soft_reasons)
    score = round(max(0.0, min(1.0, score)), 6)
    status: EdgeStatus = (
        "accepted"
        if strong_identity
        and score >= config.accept_score
        and (not soft_reasons or external_match)
        else "review"
    )
    return IdentityEdge(
        predecessor_id=predecessor_endpoint.id,
        successor_id=successor_endpoint.id,
        status=status,
        score=score,
        source="automatic",
        reasons=tuple([*(reason for reason, _ in signals), *soft_reasons]),
        gap_seconds=round(gap, 6),
        reid_mean_distance=mean_distance,
        reid_best_sample_distance=sample_distance,
        reid_robust_sample_distance=robust_sample_distance,
        reid_strong_support_left=strong_support_left,
        reid_strong_support_right=strong_support_right,
        pitch_distance_metres=pitch_distance,
        reachable_distance_metres=reachable_distance,
    )


def pair_identity_nodes(
    nodes: Sequence[IdentityNode],
    config: IdentityResolverConfig,
) -> tuple[list[tuple[int, int, IdentityEdge]], list[IdentityEdge]]:
    """Score every eligible node pair and separate candidates from rejections."""

    candidates: list[tuple[int, int, IdentityEdge]] = []
    rejected: list[IdentityEdge] = []
    eligible = [(index, node) for index, node in enumerate(nodes) if not node.excluded]
    for offset, (left_index, left) in enumerate(eligible):
        for right_index, right in eligible[offset + 1 :]:
            if left.end_time + config.temporal_epsilon_seconds < right.start_time:
                predecessor_index, predecessor = left_index, left
                successor_index, successor = right_index, right
            elif right.end_time + config.temporal_epsilon_seconds < left.start_time:
                predecessor_index, predecessor = right_index, right
                successor_index, successor = left_index, left
            else:
                predecessor_index, predecessor = min(
                    ((left_index, left), (right_index, right)),
                    key=lambda item: (item[1].start_time, item[1].key),
                )
                successor_index, successor = (
                    (right_index, right)
                    if predecessor_index == left_index
                    else (left_index, left)
                )
            edge = _candidate_edge(predecessor, successor, config)
            if edge.status == "rejected":
                rejected.append(edge)
            else:
                candidates.append((predecessor_index, successor_index, edge))
    return candidates, rejected


__all__ = ["pair_identity_nodes"]
