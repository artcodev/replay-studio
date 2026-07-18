"""Candidate evidence scoring for closed-set roster identity review."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .jersey_ocr_contract import normalize_jersey_number
from .roster_identity_contract import (
    AttributeEvidence,
    CandidateEvidence,
    CanonicalPersonEvidence,
    ParticipationEvidence,
    PersistedRosterPlayer,
    PlayerLikelihoodEvidence,
    RosterIdentityCandidate,
    RosterResolverConfig,
)
from .roster_identity_temporal import (
    expanded_intervals,
    interval_coverage,
    intervals_overlap,
    point_inside_intervals,
)


def _role_family(value: str | int | None) -> str | None:
    if value is None:
        return None
    role = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    if not role:
        return None
    if role in {"gk", "goalkeeper", "goalie", "keeper"} or "goalkeeper" in role:
        return "goalkeeper"
    if role in {"ref", "referee", "official", "assistant referee"} or "referee" in role:
        return "official"
    if any(
        token in role
        for token in (
            "coach",
            "manager",
            "staff",
            "spectator",
            "supporter",
            "fan",
            "medical",
            "physio",
            "ball boy",
            "ball girl",
            "non player",
            "other person",
        )
    ) or role == "other":
        return "non-player"
    if any(
        token in role
        for token in (
            "player",
            "defender",
            "back",
            "midfield",
            "wing",
            "forward",
            "striker",
            "attacker",
            "sweeper",
        )
    ):
        return "outfield"
    return None


def _attribute_value(evidence: AttributeEvidence | None) -> str | None:
    if evidence is None:
        return None
    return str(evidence.value).strip()


def candidate_evidence(
    code: str,
    score_delta: float,
    confidence: float,
    source: str,
    *details: str,
) -> CandidateEvidence:
    return CandidateEvidence(
        code=code,
        score_delta=float(score_delta),
        confidence=float(confidence),
        source=source,
        details=tuple(details),
    )


def evaluate_roster_candidate(
    person: CanonicalPersonEvidence,
    player: PersistedRosterPlayer,
    config: RosterResolverConfig,
    *,
    confirmed_binding: bool = False,
) -> RosterIdentityCandidate:
    evidence: list[CandidateEvidence] = []
    reasons: list[str] = []
    conflicts: list[str] = []
    identity_signal_score = 0.0

    if confirmed_binding:
        evidence.append(
            candidate_evidence(
                "manual-confirmed-binding", 1.0, 1.0, "manual-roster-binding"
            )
        )
        identity_signal_score = 1.0
        reasons.append("manual-confirmed-binding-authoritative")
    elif player.external_player_id in person.excluded_external_player_ids:
        conflicts.append("manually-excluded-player")

    team_value = _attribute_value(person.team)
    if team_value is not None and player.team_id is not None:
        confidence = person.team.confidence
        if team_value == player.team_id:
            evidence.append(
                candidate_evidence(
                    "team-match",
                    config.team_weight * confidence,
                    confidence,
                    person.team.source,
                    player.team_id,
                )
            )
            reasons.append("team-match")
        elif person.team.confirmed or confidence >= config.hard_team_confidence:
            conflicts.append("team-mismatch-hard")
        else:
            evidence.append(
                candidate_evidence(
                    "team-mismatch-soft",
                    -config.soft_team_penalty * confidence,
                    confidence,
                    person.team.source,
                    team_value,
                    player.team_id,
                )
            )
            reasons.append("team-mismatch-soft")
    elif team_value is not None:
        reasons.append("roster-team-unavailable")
    else:
        reasons.append("canonical-team-unavailable")

    canonical_number = (
        normalize_jersey_number(person.jersey_number.value)
        if person.jersey_number is not None
        else None
    )
    if canonical_number is not None and player.jersey_number is not None:
        jersey = person.jersey_number
        support_factor = 1.0 if jersey.confirmed else min(1.0, jersey.support_count / 2.0)
        effective_confidence = jersey.confidence * support_factor
        if canonical_number == player.jersey_number:
            delta = config.jersey_weight * effective_confidence
            evidence.append(
                candidate_evidence(
                    "jersey-number-match",
                    delta,
                    jersey.confidence,
                    jersey.source,
                    canonical_number,
                    f"support:{jersey.support_count}",
                )
            )
            identity_signal_score += max(0.0, delta)
            reasons.append("jersey-number-match")
        elif jersey.confirmed or (
            jersey.confidence >= config.hard_jersey_confidence
            and jersey.support_count >= config.hard_jersey_support_count
        ):
            conflicts.append("jersey-number-mismatch-hard")
        else:
            evidence.append(
                candidate_evidence(
                    "jersey-number-mismatch-soft",
                    -config.soft_jersey_penalty * effective_confidence,
                    jersey.confidence,
                    jersey.source,
                    canonical_number,
                    player.jersey_number,
                )
            )
            reasons.append("jersey-number-mismatch-soft")
    elif canonical_number is not None:
        reasons.append("roster-jersey-number-unavailable")
    else:
        reasons.append("canonical-jersey-number-unavailable")

    canonical_role = _role_family(person.role.value if person.role is not None else None)
    roster_role = _role_family(player.role)
    if (
        person.role is not None
        and person.role.confirmed
        and canonical_role in {"official", "non-player"}
    ):
        conflicts.append("confirmed-non-player-role")
    elif canonical_role is not None and roster_role is not None:
        confidence = person.role.confidence
        if canonical_role == roster_role:
            evidence.append(
                candidate_evidence(
                    "role-match",
                    config.role_weight * confidence,
                    confidence,
                    person.role.source,
                    canonical_role,
                )
            )
            reasons.append("role-match")
        elif person.role.confirmed or confidence >= config.hard_role_confidence:
            conflicts.append("role-mismatch-hard")
        else:
            evidence.append(
                candidate_evidence(
                    "role-mismatch-soft",
                    -config.soft_role_penalty * confidence,
                    confidence,
                    person.role.source,
                    canonical_role,
                    roster_role,
                )
            )
            reasons.append("role-mismatch-soft")
    elif canonical_role is not None:
        reasons.append("roster-role-unavailable")
    else:
        reasons.append("canonical-role-unavailable")

    if person.visible_intervals and player.active_intervals:
        active_with_tolerance = expanded_intervals(
            player.active_intervals, config.availability_tolerance_seconds
        )
        coverage = interval_coverage(person.visible_intervals, active_with_tolerance)
        if not intervals_overlap(person.visible_intervals, active_with_tolerance):
            conflicts.append("player-inactive-at-observation-time")
        elif coverage < config.minimum_active_coverage:
            conflicts.append("player-not-active-for-full-visible-interval")
        else:
            evidence.append(
                candidate_evidence(
                    "active-time-coverage",
                    config.availability_weight * coverage,
                    coverage,
                    "match-participation-window",
                    f"minimum:{config.minimum_active_coverage:.3f}",
                    f"tolerance-seconds:{config.availability_tolerance_seconds:.3f}",
                )
            )
            reasons.append("active-time-coverage")
    elif person.visible_intervals:
        reasons.append("roster-active-interval-unavailable")
    else:
        reasons.append("canonical-visible-interval-unavailable")

    canonical_events = {
        (item.source, item.event_id): item for item in person.participation
    }
    roster_events = {
        (item.source, item.event_id): item for item in player.participation
    }
    shared_event_keys = sorted(set(canonical_events) & set(roster_events))
    compatible_events: list[
        tuple[tuple[str, str], ParticipationEvidence, ParticipationEvidence]
    ] = []
    for event_key in shared_event_keys:
        canonical_event = canonical_events[event_key]
        roster_event = roster_events[event_key]
        if canonical_event.kind != roster_event.kind:
            conflicts.append("event-kind-mismatch")
            continue
        if (
            canonical_event.match_time_seconds is None
            or roster_event.match_time_seconds is None
        ):
            reasons.append("event-time-unavailable")
            continue
        if (
            abs(canonical_event.match_time_seconds - roster_event.match_time_seconds)
            > config.event_time_tolerance_seconds
        ):
            conflicts.append("event-time-mismatch")
            continue
        if not person.visible_intervals or not point_inside_intervals(
            canonical_event.match_time_seconds,
            person.visible_intervals,
            config.event_visibility_tolerance_seconds,
        ):
            reasons.append("event-outside-canonical-visible-interval")
            continue
        if not player.active_intervals or not point_inside_intervals(
            roster_event.match_time_seconds,
            player.active_intervals,
            config.availability_tolerance_seconds,
        ):
            conflicts.append("event-outside-player-active-interval")
            continue
        compatible_events.append((event_key, canonical_event, roster_event))

    if compatible_events:
        confidence = max(
            min(canonical_event.confidence, roster_event.confidence)
            for _, canonical_event, roster_event in compatible_events
        )
        delta = config.participation_weight * confidence
        event_namespaces = [
            f"{source}:{event_id}" for (source, event_id), _, _ in compatible_events
        ]
        evidence.append(
            candidate_evidence(
                "player-event-match",
                delta,
                confidence,
                "matched-namespaced-event",
                *event_namespaces,
            )
        )
        identity_signal_score += delta
        reasons.append("player-event-match")
    elif canonical_events:
        canonical_ids = {event_id for _, event_id in canonical_events}
        roster_ids = {event_id for _, event_id in roster_events}
        if canonical_ids & roster_ids and not shared_event_keys:
            reasons.append("event-source-mismatch")
        if not any(reason.startswith("event-") for reason in reasons):
            reasons.append("no-player-event-match")

    likelihoods_by_player: dict[str, list[PlayerLikelihoodEvidence]] = defaultdict(list)
    for item in person.player_likelihoods:
        likelihoods_by_player[item.external_player_id].append(item)
    strongest_by_player = {
        identifier: max(items, key=lambda item: (item.confidence, item.evidence_id))
        for identifier, items in likelihoods_by_player.items()
    }
    direct = strongest_by_player.get(player.external_player_id)
    if direct is not None:
        delta = config.direct_player_weight * direct.confidence
        evidence.append(
            candidate_evidence(
                "direct-player-evidence",
                delta,
                direct.confidence,
                direct.source,
                direct.evidence_id,
            )
        )
        identity_signal_score += delta
        reasons.append("direct-player-evidence")
    if strongest_by_player:
        strongest = max(
            strongest_by_player.values(),
            key=lambda item: (item.confidence, item.external_player_id),
        )
        candidate_confidence = direct.confidence if direct is not None else 0.0
        if (
            strongest.external_player_id != player.external_player_id
            and strongest.confidence >= config.hard_direct_player_confidence
            and strongest.confidence - candidate_confidence
            >= config.direct_player_gate_margin
        ):
            conflicts.append("direct-player-evidence-conflict")

    score = max(0.0, min(1.0, sum(item.score_delta for item in evidence)))
    identity_signal_score = max(0.0, min(1.0, identity_signal_score))
    unique_conflicts = tuple(dict.fromkeys(conflicts))
    return RosterIdentityCandidate(
        external_player_id=player.external_player_id,
        display_name=player.display_name,
        team_id=player.team_id,
        jersey_number=(
            str(player.jersey_number) if player.jersey_number is not None else None
        ),
        role=player.role,
        score=score,
        identity_signal_score=identity_signal_score,
        evidence=tuple(evidence),
        reasons=tuple(dict.fromkeys(reasons)),
        conflicts=unique_conflicts,
        eligible=not unique_conflicts,
        proposal_status=(
            "confirmed"
            if confirmed_binding
            else "blocked"
            if unique_conflicts
            else "alternative"
        ),
    )


def rank_roster_candidates(
    candidates: Iterable[RosterIdentityCandidate],
    *,
    limit: int,
    force_ids: Iterable[str] = (),
) -> tuple[RosterIdentityCandidate, ...]:
    priority = {
        "confirmed": 0,
        "selected": 1,
        "ambiguous": 2,
        "alternative": 3,
        "blocked": 4,
    }
    rows = sorted(
        candidates,
        key=lambda item: (
            priority[item.proposal_status],
            not item.eligible,
            -item.score,
            -item.identity_signal_score,
            item.display_name.casefold(),
            item.external_player_id,
        ),
    )
    forced = set(force_ids)
    selected = rows[:limit]
    selected_ids = {item.external_player_id for item in selected}
    selected.extend(
        item
        for item in rows[limit:]
        if item.external_player_id in forced and item.external_player_id not in selected_ids
    )
    return tuple(replace(item, rank=index) for index, item in enumerate(selected, start=1))

