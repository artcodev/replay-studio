"""Offline canonical identity resolution for fragmented person tracklets.

The online tracker deliberately stays local and conservative.  This module is
the second, offline stage: it may join non-overlapping tracklets when there is
actual identity evidence, while treating pitch motion only as a feasibility
gate.  It has no dependency on reconstruction state or persistence so it can
be tested independently and later reused by single- and multi-pass pipelines.

Two invariants are intentionally strict:

* every input tracklet is present in exactly one output group; and
* automatic stitching never relies on proximity or team colour alone.

Manual ``manual_identity_id`` components are the sole must-link override.  All
other links pass temporal, semantic, jersey, and physical constraints before a
global one-predecessor/one-successor assignment is solved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import hypot, isfinite
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


EdgeStatus = Literal["accepted", "review", "rejected"]
GroupStatus = Literal["resolved", "provisional", "excluded"]


def _normalized_embedding(value: Sequence[float] | np.ndarray | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return None
    return tuple(float(item) for item in array / norm)


def _normalized_jersey(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(int(text)) if text.isdigit() else text.upper()


def _pitch_position(value: Sequence[float] | None, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain finite coordinates")
    return result


@dataclass(frozen=True)
class IdentityTracklet:
    """Immutable evidence summary for one local online-tracker trajectory."""

    id: str
    start_time: float
    end_time: float
    team_id: str | None = None
    role: str | None = None
    external_player_id: str | None = None
    jersey_number: str | int | None = None
    jersey_confidence: float = 0.0
    jersey_sample_count: int = 0
    mean_reid_embedding: Sequence[float] | np.ndarray | None = None
    reid_embeddings: Sequence[Sequence[float] | np.ndarray] = field(default_factory=tuple)
    start_pitch: Sequence[float] | None = None
    end_pitch: Sequence[float] | None = None
    start_uncertainty_metres: float | None = None
    end_uncertainty_metres: float | None = None
    observation_count: int = 0
    manual_confirmed: bool = False
    manual_excluded: bool = False
    manual_identity_id: str | None = None
    manual_team: bool = False
    manual_role: bool = False
    manual_jersey: bool = False

    def __post_init__(self) -> None:
        identifier = str(self.id).strip()
        if not identifier:
            raise ValueError("IdentityTracklet.id must not be empty")
        start, end = float(self.start_time), float(self.end_time)
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError("IdentityTracklet times must be finite and end_time >= start_time")
        confidence = float(self.jersey_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("jersey_confidence must be between 0 and 1")
        if int(self.jersey_sample_count) < 0 or int(self.observation_count) < 0:
            raise ValueError("sample and observation counts must be non-negative")

        mean = _normalized_embedding(self.mean_reid_embedding)
        samples = tuple(
            item
            for value in self.reid_embeddings
            if (item := _normalized_embedding(value)) is not None
        )
        dimensions = {len(item) for item in samples}
        if mean is not None:
            dimensions.add(len(mean))
        if len(dimensions) > 1:
            raise ValueError("All ReID embeddings in one tracklet must have the same dimension")
        if mean is None and samples:
            mean = _normalized_embedding(np.mean(np.asarray(samples, dtype=np.float64), axis=0))

        manual_identity_id = (
            str(self.manual_identity_id).strip() if self.manual_identity_id is not None else None
        )
        external_player_id = (
            str(self.external_player_id).strip() if self.external_player_id is not None else None
        )
        team_id = str(self.team_id).strip() if self.team_id is not None else None
        role = str(self.role).strip() if self.role is not None else None

        for value, name in (
            (self.start_uncertainty_metres, "start_uncertainty_metres"),
            (self.end_uncertainty_metres, "end_uncertainty_metres"),
        ):
            if value is not None and (not isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "team_id", team_id or None)
        object.__setattr__(self, "role", role or None)
        object.__setattr__(self, "external_player_id", external_player_id or None)
        object.__setattr__(self, "manual_identity_id", manual_identity_id or None)
        object.__setattr__(self, "jersey_number", _normalized_jersey(self.jersey_number))
        object.__setattr__(self, "jersey_confidence", confidence)
        object.__setattr__(self, "jersey_sample_count", int(self.jersey_sample_count))
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(self, "mean_reid_embedding", mean)
        object.__setattr__(self, "reid_embeddings", samples)
        object.__setattr__(self, "start_pitch", _pitch_position(self.start_pitch, "start_pitch"))
        object.__setattr__(self, "end_pitch", _pitch_position(self.end_pitch, "end_pitch"))
        object.__setattr__(
            self,
            "start_uncertainty_metres",
            float(self.start_uncertainty_metres)
            if self.start_uncertainty_metres is not None
            else None,
        )
        object.__setattr__(
            self,
            "end_uncertainty_metres",
            float(self.end_uncertainty_metres)
            if self.end_uncertainty_metres is not None
            else None,
        )


@dataclass(frozen=True)
class IdentityResolverConfig:
    reliable_jersey_confidence: float = 0.80
    reliable_jersey_samples: int = 2
    strong_reid_distance: float = 0.18
    review_reid_distance: float = 0.30
    strong_sample_reid_distance: float = 0.10
    min_strong_reid_samples: int = 2
    accept_score: float = 0.78
    ambiguity_margin: float = 0.08
    ambiguity_gap_penalty_per_second: float = 0.20
    ambiguity_gap_penalty_cap: float = 0.25
    max_player_speed_metres_per_second: float = 12.0
    motion_slack_metres: float = 2.0
    temporal_epsilon_seconds: float = 1e-6

    def __post_init__(self) -> None:
        probability_values = (
            self.reliable_jersey_confidence,
            self.strong_reid_distance,
            self.review_reid_distance,
            self.strong_sample_reid_distance,
            self.accept_score,
            self.ambiguity_margin,
            self.ambiguity_gap_penalty_per_second,
            self.ambiguity_gap_penalty_cap,
        )
        if any(not isfinite(float(item)) or float(item) < 0.0 for item in probability_values):
            raise ValueError("Identity resolver thresholds must be finite and non-negative")
        if self.strong_reid_distance > self.review_reid_distance:
            raise ValueError("strong_reid_distance must not exceed review_reid_distance")
        if not 0.0 <= self.accept_score <= 1.0:
            raise ValueError("accept_score must be between 0 and 1")
        if self.reliable_jersey_samples < 1:
            raise ValueError("reliable_jersey_samples must be positive")
        if self.min_strong_reid_samples < 2:
            raise ValueError("min_strong_reid_samples must be at least two")
        if self.max_player_speed_metres_per_second <= 0.0 or self.motion_slack_metres < 0.0:
            raise ValueError("Motion limits must be positive")


@dataclass(frozen=True)
class IdentityEdge:
    predecessor_id: str
    successor_id: str
    status: EdgeStatus
    score: float | None
    source: str
    reasons: tuple[str, ...] = ()
    gap_seconds: float | None = None
    reid_mean_distance: float | None = None
    reid_best_sample_distance: float | None = None
    reid_robust_sample_distance: float | None = None
    reid_strong_support_left: int = 0
    reid_strong_support_right: int = 0
    pitch_distance_metres: float | None = None
    reachable_distance_metres: float | None = None


@dataclass(frozen=True)
class IdentityGroup:
    id: str
    tracklet_ids: tuple[str, ...]
    status: GroupStatus
    confidence: float
    source: str
    team_id: str | None
    role: str | None
    external_player_id: str | None
    jersey_number: str | None
    manual_identity_id: str | None
    observation_count: int


@dataclass(frozen=True)
class IdentityResolution:
    groups: tuple[IdentityGroup, ...]
    accepted_edges: tuple[IdentityEdge, ...]
    review_edges: tuple[IdentityEdge, ...]
    rejected_edges: tuple[IdentityEdge, ...]
    diagnostics: dict
    tracklet_to_identity: dict[str, str]


@dataclass(frozen=True)
class _Node:
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


def _node_from_members(key: str, members: Sequence[IdentityTracklet]) -> _Node:
    ordered = tuple(sorted(members, key=lambda item: (item.start_time, item.end_time, item.id)))
    team_id, team_conflict = _one_or_none(item.team_id for item in ordered)
    role, role_conflict = _one_or_none(item.role for item in ordered)
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
            _normalized_embedding(np.mean(np.asarray(weighted, dtype=np.float64), axis=0))
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
    return _Node(
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


def _nodes(tracklets: Sequence[IdentityTracklet]) -> tuple[list[_Node], list[IdentityEdge]]:
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


def _endpoint(node: _Node, *, successor: bool) -> IdentityTracklet:
    if successor:
        return min(node.members, key=lambda item: (item.start_time, item.id))
    return max(node.members, key=lambda item: (item.end_time, item.id))


def _reliable_jersey(node: _Node, config: IdentityResolverConfig) -> bool:
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
    left: _Node,
    right: _Node,
    strong_threshold: float,
) -> tuple[float | None, float | None, float | None, int, int]:
    """Return mean, optimistic and robust bidirectional ReID evidence.

    The minimum of an N×M distance matrix is useful for review, but it is not
    identity proof: one contaminated crop can coincide by accident. Robust
    support uses the median nearest-neighbour distance in both directions, so
    a strong sample signal requires several independent crops on both sides.
    """

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
    predecessor: _Node,
    successor: _Node,
    config: IdentityResolverConfig,
) -> IdentityEdge:
    predecessor_endpoint = _endpoint(predecessor, successor=False)
    successor_endpoint = _endpoint(successor, successor=True)
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
    if hard_reasons:
        return IdentityEdge(
            predecessor_id=predecessor_endpoint.id,
            successor_id=successor_endpoint.id,
            status="rejected",
            score=None,
            source="constraints",
            reasons=tuple(dict.fromkeys(hard_reasons)),
            gap_seconds=round(gap, 6),
            reid_mean_distance=mean_distance,
            reid_best_sample_distance=sample_distance,
            reid_robust_sample_distance=robust_sample_distance,
            reid_strong_support_left=strong_support_left,
            reid_strong_support_right=strong_support_right,
            pitch_distance_metres=pitch_distance,
            reachable_distance_metres=reachable_distance,
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
    # A mean embedding is useful for review, but cannot be identity proof on
    # its own: it may represent a single blurry or contaminated crop.  An
    # automatic link requires several mutually supported samples in both
    # directions.  This also prevents one coincident sample from hiding behind
    # a deceptively good aggregate mean.
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
        # Score from the least favourable aggregate, not the most optimistic
        # one.  The edge may still be accepted, but its confidence must expose
        # disagreement between the mean and the robust sample statistic.
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


def _pair_nodes(
    nodes: Sequence[_Node], config: IdentityResolverConfig
) -> tuple[list[tuple[int, int, IdentityEdge]], list[IdentityEdge]]:
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


def _mark_ambiguities(
    candidates: list[tuple[int, int, IdentityEdge]],
    config: IdentityResolverConfig,
) -> list[tuple[int, int, IdentityEdge]]:
    def ambiguity_rank(edge: IdentityEdge) -> float:
        # When identity evidence is equally strong, the chronologically nearest
        # compatible fragment is the natural predecessor/successor.  This term
        # is used only to decide whether two alternatives are ambiguous; it
        # never upgrades weak evidence into an accepted stitch.
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


def _global_assignment(
    nodes: Sequence[_Node],
    candidates: list[tuple[int, int, IdentityEdge]],
    config: IdentityResolverConfig,
) -> tuple[list[IdentityEdge], list[IdentityEdge]]:
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
        # A deterministic epsilon resolves numerical ties after the explicit
        # ambiguity gate has already failed closed on semantically close edges.
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
    """Return jersey values that are reliable within a prospective component.

    OCR evidence may be spread over several fragments.  Aggregate sample counts
    per normalized number, but require at least one sufficiently confident crop
    (or an explicit manual jersey) before treating the value as a cannot-link
    constraint.
    """

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
    """Validate evidence after joining two already-compatible components.

    Pairwise constraints are insufficient for an unlabeled bridge: A can match
    an unlabeled B and B can match C even when A and C carry incompatible hard
    identity evidence.  Every selected automatic edge therefore has to keep
    the complete union semantically coherent.
    """

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


def _filter_transitive_component_conflicts(
    tracklets: Sequence[IdentityTracklet],
    manual_edges: Sequence[IdentityEdge],
    automatic_edges: Sequence[IdentityEdge],
    config: IdentityResolverConfig,
) -> tuple[list[IdentityEdge], list[IdentityEdge], _UnionFind]:
    """Union selected edges without allowing an unlabeled transitive bridge."""

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

    # Manual same-owner links are the explicit must-link override.  They form
    # the initial components; automatic links may not introduce any additional
    # hard-evidence conflict around them.
    for edge in manual_edges:
        join(edge.predecessor_id, edge.successor_id)

    accepted: list[IdentityEdge] = []
    demoted: list[IdentityEdge] = []
    # Prefer the highest-confidence selected associations when two otherwise
    # valid edges would form an incompatible chain.  The remaining key makes
    # equal-score outcomes independent of input order.
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
    return accepted, demoted, union


def _group_identifier(members: Sequence[IdentityTracklet]) -> str:
    manual_ids = sorted({item.manual_identity_id for item in members if item.manual_identity_id})
    if len(manual_ids) == 1:
        return f"identity:manual:{manual_ids[0]}"
    external_ids = sorted({item.external_player_id for item in members if item.external_player_id})
    if len(external_ids) == 1:
        return f"identity:external:{external_ids[0]}"
    # The lexicographically smallest source id keeps the identifier stable when
    # a later offline pass appends another tracklet to an existing component.
    return f"identity:{min(item.id for item in members)}"


def _resolved_group(
    members: Sequence[IdentityTracklet],
    accepted_edges: Sequence[IdentityEdge],
) -> IdentityGroup:
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


def _edge_sort_key(edge: IdentityEdge) -> tuple:
    return (edge.predecessor_id, edge.successor_id, edge.source, edge.reasons)


def resolve_identities(
    tracklets: Sequence[IdentityTracklet],
    config: IdentityResolverConfig | None = None,
) -> IdentityResolution:
    """Resolve local tracklets into deterministic canonical identity groups."""

    config = config or IdentityResolverConfig()
    ordered = tuple(sorted(tracklets, key=lambda item: (item.start_time, item.end_time, item.id)))
    if len({item.id for item in ordered}) != len(ordered):
        raise ValueError("IdentityTracklet ids must be unique")
    if not ordered:
        return IdentityResolution(
            groups=(),
            accepted_edges=(),
            review_edges=(),
            rejected_edges=(),
            diagnostics={
                "schemaVersion": 1,
                "trackletCount": 0,
                "canonicalIdentityCount": 0,
                "allTrackletsPreserved": True,
                "identityObservationCoverage": 1.0,
                "associationConfidenceP10": None,
                "associationConfidenceP50": None,
                "acceptedAssociationConfidenceP10": None,
                "reviewAssociationConfidenceP50": None,
                "strongReidBidirectionalEdgeCount": 0,
                "groundTruthAvailable": False,
                "estimatedIdSwitchCount": None,
                "duplicateOverlapSeconds": None,
            },
            tracklet_to_identity={},
        )

    nodes, manual_edges = _nodes(ordered)
    candidates, rejected = _pair_nodes(nodes, config)
    candidates = _mark_ambiguities(candidates, config)
    auto_accepted, review = _global_assignment(nodes, candidates, config)
    auto_accepted, transitive_review, union = _filter_transitive_component_conflicts(
        ordered,
        manual_edges,
        auto_accepted,
        config,
    )
    review.extend(transitive_review)
    accepted = [*manual_edges, *auto_accepted]

    members_by_root: dict[str, list[IdentityTracklet]] = defaultdict(list)
    for item in ordered:
        members_by_root[union.find(item.id)].append(item)
    groups = [
        _resolved_group(members, accepted)
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
    rejection_counts = Counter(reason for edge in rejected for reason in edge.reasons)
    observation_count = sum(item.observation_count for item in ordered)
    accepted_scores = [float(edge.score) for edge in accepted if edge.score is not None]
    review_scores = [float(edge.score) for edge in review if edge.score is not None]
    evidence_scores = [*accepted_scores, *review_scores]
    diagnostics = {
        "schemaVersion": 1,
        "trackletCount": len(ordered),
        "eligibleTrackletCount": sum(not item.manual_excluded for item in ordered),
        "excludedTrackletCount": sum(item.manual_excluded for item in ordered),
        "canonicalIdentityCount": len(groups),
        "resolvedIdentityCount": sum(group.status == "resolved" for group in groups),
        "provisionalIdentityCount": sum(group.status == "provisional" for group in groups),
        "excludedIdentityCount": sum(group.status == "excluded" for group in groups),
        "acceptedEdgeCount": len(accepted),
        "autoStitchCount": sum(edge.source == "automatic" for edge in accepted),
        "manualStitchCount": sum(edge.source == "manual" for edge in accepted),
        "reviewEdgeCount": len(review),
        "rejectedEdgeCount": len(rejected),
        "ambiguousEdgeCount": sum(
            any(reason.startswith("ambiguous-") for reason in edge.reasons)
            for edge in review
        ),
        "manualConflictCount": sum(bool(node.conflict_reasons) for node in nodes),
        "rejectionReasonCounts": dict(sorted(rejection_counts.items())),
        "observationCount": observation_count,
        "preservedObservationCount": observation_count,
        "identityObservationCoverage": 1.0,
        "associationConfidenceP10": (
            round(float(np.percentile(evidence_scores, 10)), 6)
            if evidence_scores
            else None
        ),
        "associationConfidenceP50": (
            round(float(np.percentile(evidence_scores, 50)), 6)
            if evidence_scores
            else None
        ),
        "acceptedAssociationConfidenceP10": (
            round(float(np.percentile(accepted_scores, 10)), 6)
            if accepted_scores
            else None
        ),
        "reviewAssociationConfidenceP50": (
            round(float(np.percentile(review_scores, 50)), 6)
            if review_scores
            else None
        ),
        "strongReidBidirectionalEdgeCount": sum(
            "strong-reid" in edge.reasons
            and edge.reid_strong_support_left >= config.min_strong_reid_samples
            and edge.reid_strong_support_right >= config.min_strong_reid_samples
            for edge in accepted
        ),
        # These are label-dependent accuracy values. Keep them explicitly
        # unknown for ordinary clips instead of deriving them from continuity.
        "groundTruthAvailable": False,
        "estimatedIdSwitchCount": None,
        "duplicateOverlapSeconds": None,
        "allTrackletsPreserved": len(tracklet_to_identity) == len(ordered),
        "inputTrackletDigest": sha256(
            "\n".join(sorted(item.id for item in ordered)).encode("utf-8")
        ).hexdigest(),
    }
    return IdentityResolution(
        groups=tuple(groups),
        accepted_edges=tuple(accepted),
        review_edges=tuple(review),
        rejected_edges=tuple(rejected),
        diagnostics=diagnostics,
        tracklet_to_identity=tracklet_to_identity,
    )


# The longer name is convenient at the reconstruction integration seam.
resolve_global_identities = resolve_identities


__all__ = [
    "IdentityEdge",
    "IdentityGroup",
    "IdentityResolution",
    "IdentityResolverConfig",
    "IdentityTracklet",
    "resolve_global_identities",
    "resolve_identities",
]
