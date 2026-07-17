"""Conservative jersey-number OCR fusion and roster candidate generation.

The detector/OCR worker is expected to emit many correlated readings for a
person tracklet.  This module deliberately does not treat every video frame as
an independent vote.  It keeps the best readable frame in each temporal
window, combines the remaining evidence, and publishes a jersey number only
when both support and confidence thresholds pass.

The API is pure and reconstruction-agnostic.  In particular, roster data is a
review prior only: :func:`generate_roster_candidates` can rank possible
players, but its result has no field that can automatically bind a canonical
person to an external player.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from math import floor, isfinite
from typing import Iterable, Literal, Mapping, Sequence


JerseyEvidenceScope = Literal["tracklet", "canonical"]
JerseyEvidenceStatus = Literal["reliable", "provisional", "conflict", "no-evidence"]


def normalize_jersey_number(
    value: str | int | None,
    *,
    minimum: int = 0,
    maximum: int = 99,
) -> str | None:
    """Return a canonical decimal jersey number, or ``None`` when unsafe.

    OCR character substitutions (for example ``O -> 0``) are intentionally
    not guessed here.  A worker may provide such alternatives as separate OCR
    observations with their own confidence instead.
    """

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if text.startswith("#"):
            text = text[1:].strip()
        if not text or not text.isascii() or not text.isdigit():
            return None
        number = int(text)
    if number < minimum or number > maximum:
        return None
    return str(number)


def _probability(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return result


def _identifier(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True)
class JerseyOcrObservation:
    """One OCR reading from one crop of a local person tracklet."""

    id: str
    tracklet_id: str
    timestamp_seconds: float
    raw_number: str | int | None
    ocr_confidence: float
    frame_quality: float = 1.0
    back_visibility: float = 1.0
    frame_index: int | None = None
    source: str = "jersey-ocr"
    evidence_fingerprint: str | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_seconds)
        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        frame_index = None if self.frame_index is None else int(self.frame_index)
        if frame_index is not None and frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        object.__setattr__(self, "id", _identifier(self.id, "JerseyOcrObservation.id"))
        object.__setattr__(
            self,
            "tracklet_id",
            _identifier(self.tracklet_id, "JerseyOcrObservation.tracklet_id"),
        )
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(
            self,
            "ocr_confidence",
            _probability(self.ocr_confidence, "ocr_confidence"),
        )
        object.__setattr__(
            self,
            "frame_quality",
            _probability(self.frame_quality, "frame_quality"),
        )
        object.__setattr__(
            self,
            "back_visibility",
            _probability(self.back_visibility, "back_visibility"),
        )
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "source", str(self.source).strip() or "jersey-ocr")
        fingerprint = self.evidence_fingerprint
        if fingerprint is not None:
            if (
                not isinstance(fingerprint, str)
                or not fingerprint
                or len(fingerprint) > 160
                or not fingerprint.isascii()
                or any(character.isspace() for character in fingerprint)
            ):
                raise ValueError("evidence_fingerprint must be a stable opaque string")
        object.__setattr__(self, "evidence_fingerprint", fingerprint)

    @property
    def effective_score(self) -> float:
        """Quality-aware confidence used for sampling and voting."""

        return self.ocr_confidence * self.frame_quality * self.back_visibility


@dataclass(frozen=True)
class JerseyFusionConfig:
    """Thresholds for conservative sampling and evidence publication."""

    sampling_window_seconds: float = 0.50
    min_ocr_confidence: float = 0.55
    min_frame_quality: float = 0.35
    min_back_visibility: float = 0.35
    min_effective_score: float = 0.30
    max_selected_frames: int = 16
    reliable_confidence: float = 0.80
    reliable_support_count: int = 2
    # An exact 2:1 majority is allowed when its confidence/margin also passes;
    # two equally supported readings still fail closed as a conflict.
    conflict_vote_share: float = 0.34
    minimum_vote_margin: float = 0.20
    minimum_number: int = 0
    maximum_number: int = 99

    def __post_init__(self) -> None:
        if not isfinite(float(self.sampling_window_seconds)) or self.sampling_window_seconds <= 0:
            raise ValueError("sampling_window_seconds must be finite and positive")
        for name in (
            "min_ocr_confidence",
            "min_frame_quality",
            "min_back_visibility",
            "min_effective_score",
            "reliable_confidence",
            "conflict_vote_share",
            "minimum_vote_margin",
        ):
            _probability(getattr(self, name), name)
        if int(self.max_selected_frames) < 1:
            raise ValueError("max_selected_frames must be positive")
        if int(self.reliable_support_count) < 1:
            raise ValueError("reliable_support_count must be positive")
        if int(self.minimum_number) < 0 or int(self.maximum_number) < int(
            self.minimum_number
        ):
            raise ValueError("jersey number range is invalid")


@dataclass(frozen=True)
class JerseyVote:
    number: str
    support_count: int
    weight: float
    weight_share: float
    mean_effective_score: float
    observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class JerseyEvidenceSummary:
    """Fused evidence for one tracklet or one canonical person.

    ``jersey_number`` is fail-closed: it is non-null only for ``reliable``
    evidence.  ``candidate_number`` may expose a non-conflicting but
    provisional top OCR reading to a review UI.
    """

    subject_id: str
    scope: JerseyEvidenceScope
    status: JerseyEvidenceStatus
    jersey_number: str | None
    candidate_number: str | None
    confidence: float
    support_count: int
    selected_sample_count: int
    selected_observations: tuple[JerseyOcrObservation, ...]
    votes: tuple[JerseyVote, ...]
    tracklet_ids: tuple[str, ...]
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    conflict_reasons: tuple[str, ...] = ()

    def identity_resolver_fields(self) -> dict[str, str | float | int | None]:
        """Fields accepted by :class:`identity_resolver.IdentityTracklet`.

        Provisional/conflicting evidence returns a null jersey and zero sample
        count, so it cannot become weak or reliable identity evidence by
        accident at the integration boundary.
        """

        if self.jersey_number is None:
            return {
                "jersey_number": None,
                "jersey_confidence": 0.0,
                "jersey_sample_count": 0,
            }
        return {
            "jersey_number": self.jersey_number,
            "jersey_confidence": self.confidence,
            "jersey_sample_count": self.support_count,
        }

    def to_payload(self) -> dict:
        """JSON-ready diagnostics suitable for canonical-person documents."""

        return {
            "subjectId": self.subject_id,
            "scope": self.scope,
            "status": self.status,
            "jerseyNumber": self.jersey_number,
            "candidateNumber": self.candidate_number,
            "confidence": round(self.confidence, 6),
            "supportCount": self.support_count,
            "selectedSampleCount": self.selected_sample_count,
            "selectedObservationIds": [item.id for item in self.selected_observations],
            "trackletIds": list(self.tracklet_ids),
            "votes": [
                {
                    "number": item.number,
                    "supportCount": item.support_count,
                    "weight": round(item.weight, 6),
                    "weightShare": round(item.weight_share, 6),
                    "meanEffectiveScore": round(item.mean_effective_score, 6),
                    "observationIds": list(item.observation_ids),
                }
                for item in self.votes
            ],
            "rejectionCounts": dict(sorted(self.rejection_counts.items())),
            "conflictReasons": list(self.conflict_reasons),
        }


def _observation_order(item: JerseyOcrObservation) -> tuple:
    return (
        item.timestamp_seconds,
        item.frame_index if item.frame_index is not None else 2**63 - 1,
        item.tracklet_id,
        item.id,
    )


def _quality_order(item: JerseyOcrObservation) -> tuple:
    # ``min`` over this tuple selects the best item and gives deterministic
    # earliest-frame/id tie-breaking independent of input order.
    return (
        -item.effective_score,
        -item.ocr_confidence,
        -item.frame_quality,
        -item.back_visibility,
        *_observation_order(item),
    )


def _fuse(
    subject_id: str,
    scope: JerseyEvidenceScope,
    observations: Sequence[JerseyOcrObservation],
    tracklet_ids: Sequence[str],
    config: JerseyFusionConfig,
) -> JerseyEvidenceSummary:
    rejection_counts: Counter[str] = Counter()
    eligible: list[tuple[JerseyOcrObservation, str]] = []
    observation_key_counts = Counter((item.tracklet_id, item.id) for item in observations)
    fingerprint_groups: dict[str, list[JerseyOcrObservation]] = defaultdict(list)
    for item in observations:
        if (
            item.evidence_fingerprint is not None
            and observation_key_counts[(item.tracklet_id, item.id)] == 1
        ):
            fingerprint_groups[item.evidence_fingerprint].append(item)
    fingerprint_winners = {
        fingerprint: (winner.tracklet_id, winner.id)
        for fingerprint, items in fingerprint_groups.items()
        for winner in [min(items, key=_quality_order)]
    }
    for item in sorted(observations, key=_observation_order):
        if observation_key_counts[(item.tracklet_id, item.id)] > 1:
            # Replayed worker messages must not manufacture independent OCR
            # support.  Reject all ambiguous duplicates instead of guessing
            # which payload carrying the same identity is authoritative.
            rejection_counts["duplicate-observation-id"] += 1
            continue
        if (
            item.evidence_fingerprint is not None
            and fingerprint_winners[item.evidence_fingerprint]
            != (item.tracklet_id, item.id)
        ):
            # Different observation/crop IDs do not make identical decoded
            # pixels independent evidence. Keep one deterministic best row.
            rejection_counts["duplicate-evidence-fingerprint"] += 1
            continue
        number = normalize_jersey_number(
            item.raw_number,
            minimum=config.minimum_number,
            maximum=config.maximum_number,
        )
        if number is None:
            rejection_counts["invalid-or-missing-number"] += 1
        elif item.ocr_confidence < config.min_ocr_confidence:
            rejection_counts["ocr-confidence-low"] += 1
        elif item.frame_quality < config.min_frame_quality:
            rejection_counts["frame-quality-low"] += 1
        elif item.back_visibility < config.min_back_visibility:
            rejection_counts["back-visibility-low"] += 1
        elif item.effective_score < config.min_effective_score:
            rejection_counts["effective-score-low"] += 1
        else:
            eligible.append((item, number))

    # Per-tracklet binning avoids both dense-frame vote inflation and accidental
    # suppression when different tracklets happen at the same timestamp.
    windows: dict[tuple[str, int], list[tuple[JerseyOcrObservation, str]]] = defaultdict(list)
    for item, number in eligible:
        window = floor(item.timestamp_seconds / config.sampling_window_seconds)
        windows[(item.tracklet_id, window)].append((item, number))

    sampled: list[tuple[JerseyOcrObservation, str]] = []
    for candidates in windows.values():
        chosen = min(candidates, key=lambda row: _quality_order(row[0]))
        sampled.append(chosen)
        rejection_counts["inferior-frame-same-window"] += len(candidates) - 1

    sampled.sort(key=lambda row: _quality_order(row[0]))
    if len(sampled) > config.max_selected_frames:
        rejection_counts["sample-cap"] += len(sampled) - config.max_selected_frames
        sampled = sampled[: config.max_selected_frames]
    sampled.sort(key=lambda row: _observation_order(row[0]))

    if not sampled:
        return JerseyEvidenceSummary(
            subject_id=subject_id,
            scope=scope,
            status="no-evidence",
            jersey_number=None,
            candidate_number=None,
            confidence=0.0,
            support_count=0,
            selected_sample_count=0,
            selected_observations=(),
            votes=(),
            tracklet_ids=tuple(sorted(set(tracklet_ids))),
            rejection_counts=dict(rejection_counts),
        )

    by_number: dict[str, list[JerseyOcrObservation]] = defaultdict(list)
    for item, number in sampled:
        by_number[number].append(item)
    total_weight = sum(item.effective_score for item, _ in sampled)
    votes = []
    for number, items in by_number.items():
        weight = sum(item.effective_score for item in items)
        votes.append(
            JerseyVote(
                number=number,
                support_count=len(items),
                weight=weight,
                weight_share=weight / total_weight,
                mean_effective_score=weight / len(items),
                observation_ids=tuple(item.id for item in sorted(items, key=_observation_order)),
            )
        )
    votes.sort(key=lambda item: (-item.weight, -item.support_count, item.number))
    winner = votes[0]
    runner_share = votes[1].weight_share if len(votes) > 1 else 0.0
    margin = winner.weight_share - runner_share
    conflict = len(votes) > 1 and (
        runner_share >= config.conflict_vote_share or margin < config.minimum_vote_margin
    )
    confidence = min(
        1.0,
        0.60 * winner.mean_effective_score + 0.40 * winner.weight_share,
    )

    conflict_reasons: tuple[str, ...] = ()
    if conflict:
        reasons = ["competing-jersey-numbers"]
        if runner_share >= config.conflict_vote_share:
            reasons.append("runner-up-support-too-high")
        if margin < config.minimum_vote_margin:
            reasons.append("winner-margin-too-small")
        conflict_reasons = tuple(reasons)
        status: JerseyEvidenceStatus = "conflict"
        number = None
        candidate = None
        published_confidence = 0.0
    elif (
        winner.support_count >= config.reliable_support_count
        and confidence >= config.reliable_confidence
    ):
        status = "reliable"
        number = winner.number
        candidate = winner.number
        published_confidence = confidence
    else:
        status = "provisional"
        number = None
        candidate = winner.number
        published_confidence = confidence

    return JerseyEvidenceSummary(
        subject_id=subject_id,
        scope=scope,
        status=status,
        jersey_number=number,
        candidate_number=candidate,
        confidence=published_confidence,
        support_count=winner.support_count,
        selected_sample_count=len(sampled),
        selected_observations=tuple(item for item, _ in sampled),
        votes=tuple(votes),
        tracklet_ids=tuple(sorted(set(tracklet_ids))),
        rejection_counts=dict(rejection_counts),
        conflict_reasons=conflict_reasons,
    )


def aggregate_tracklet_evidence(
    tracklet_id: str,
    observations: Iterable[JerseyOcrObservation],
    *,
    config: JerseyFusionConfig | None = None,
) -> JerseyEvidenceSummary:
    """Sample and fuse OCR readings belonging to exactly one tracklet."""

    identifier = _identifier(tracklet_id, "tracklet_id")
    rows = tuple(observations)
    mismatches = sorted({item.tracklet_id for item in rows if item.tracklet_id != identifier})
    if mismatches:
        raise ValueError(
            f"observations for {identifier!r} include other tracklets: {mismatches!r}"
        )
    return _fuse(
        identifier,
        "tracklet",
        rows,
        (identifier,),
        config or JerseyFusionConfig(),
    )


def aggregate_tracklets(
    observations: Iterable[JerseyOcrObservation],
    *,
    config: JerseyFusionConfig | None = None,
) -> dict[str, JerseyEvidenceSummary]:
    """Group arbitrary OCR readings and return one summary per tracklet."""

    groups: dict[str, list[JerseyOcrObservation]] = defaultdict(list)
    for item in observations:
        groups[item.tracklet_id].append(item)
    return {
        tracklet_id: aggregate_tracklet_evidence(tracklet_id, rows, config=config)
        for tracklet_id, rows in sorted(groups.items())
    }


def aggregate_canonical_people(
    tracklet_summaries: Mapping[str, JerseyEvidenceSummary]
    | Iterable[JerseyEvidenceSummary],
    tracklet_to_canonical: Mapping[str, str],
    *,
    config: JerseyFusionConfig | None = None,
) -> dict[str, JerseyEvidenceSummary]:
    """Fuse sampled tracklet evidence into canonical-person summaries.

    Every supplied tracklet must have an explicit canonical mapping.  Reliable
    but disagreeing tracklet numbers are retained as a hard canonical conflict,
    even if raw vote weight would otherwise hide the disagreement.
    """

    if isinstance(tracklet_summaries, Mapping):
        rows = list(tracklet_summaries.values())
        for key, item in tracklet_summaries.items():
            if key != item.subject_id:
                raise ValueError("tracklet summary mapping key must equal summary.subject_id")
    else:
        rows = list(tracklet_summaries)
    by_canonical: dict[str, list[JerseyEvidenceSummary]] = defaultdict(list)
    for item in rows:
        if item.scope != "tracklet":
            raise ValueError("aggregate_canonical_people accepts tracklet summaries only")
        canonical_id = tracklet_to_canonical.get(item.subject_id)
        if canonical_id is None:
            raise ValueError(f"missing canonical mapping for tracklet {item.subject_id!r}")
        canonical_id = _identifier(canonical_id, "canonical person id")
        by_canonical[canonical_id].append(item)

    result: dict[str, JerseyEvidenceSummary] = {}
    fusion_config = config or JerseyFusionConfig()
    for canonical_id, members in sorted(by_canonical.items()):
        observations = tuple(
            observation
            for member in members
            for observation in member.selected_observations
        )
        tracklet_ids = tuple(sorted(member.subject_id for member in members))
        aggregate = _fuse(
            canonical_id,
            "canonical",
            observations,
            tracklet_ids,
            fusion_config,
        )
        member_rejections: Counter[str] = Counter()
        for member in members:
            member_rejections.update(member.rejection_counts)
        member_rejections.update(aggregate.rejection_counts)
        aggregate = replace(aggregate, rejection_counts=dict(member_rejections))
        reliable_numbers = {
            member.jersey_number
            for member in members
            if member.status == "reliable" and member.jersey_number is not None
        }
        if len(reliable_numbers) > 1:
            aggregate = replace(
                aggregate,
                status="conflict",
                jersey_number=None,
                candidate_number=None,
                confidence=0.0,
                conflict_reasons=tuple(
                    dict.fromkeys(
                        (*aggregate.conflict_reasons, "reliable-tracklet-jersey-conflict")
                    )
                ),
            )
        result[canonical_id] = aggregate
    return result


@dataclass(frozen=True)
class RosterPlayer:
    """Minimal roster record used only to propose review candidates."""

    external_player_id: str
    display_name: str
    jersey_number: str | int | None = None
    team_id: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_player_id",
            _identifier(self.external_player_id, "RosterPlayer.external_player_id"),
        )
        object.__setattr__(
            self,
            "display_name",
            _identifier(self.display_name, "RosterPlayer.display_name"),
        )
        team_id = str(self.team_id).strip() if self.team_id is not None else ""
        role = str(self.role).strip() if self.role is not None else ""
        object.__setattr__(self, "team_id", team_id or None)
        object.__setattr__(self, "role", role or None)
        object.__setattr__(
            self,
            "jersey_number",
            normalize_jersey_number(self.jersey_number),
        )


@dataclass(frozen=True)
class RosterCandidate:
    external_player_id: str
    display_name: str
    jersey_number: str
    team_id: str | None
    role: str | None
    score: float
    reasons: tuple[str, ...]
    requires_manual_confirmation: bool = field(default=True, init=False)

    def to_payload(self) -> dict:
        return {
            "externalPlayerId": self.external_player_id,
            "name": self.display_name,
            "number": self.jersey_number,
            "teamId": self.team_id,
            "position": self.role,
            "confidence": round(self.score, 6),
            "reasons": list(self.reasons),
            "requiresManualConfirmation": True,
        }


@dataclass(frozen=True)
class RosterCandidateSet:
    subject_id: str
    candidates: tuple[RosterCandidate, ...]
    reason: str
    requires_manual_confirmation: bool = field(default=True, init=False)

    @property
    def auto_bind_external_player_id(self) -> None:
        """Roster evidence is never sufficient for an automatic identity bind."""

        return None

    def to_payload(self) -> list[dict]:
        return [item.to_payload() for item in self.candidates]


def generate_roster_candidates(
    evidence: JerseyEvidenceSummary,
    roster: Iterable[RosterPlayer],
    *,
    team_id: str | None = None,
    limit: int = 10,
) -> RosterCandidateSet:
    """Rank exact-number roster matches without selecting or binding one.

    Team membership is only a ranking hint.  A team match without reliable OCR
    never creates a candidate, and even one exact team+number match still
    requires an explicit manual decision by the caller.
    """

    if int(limit) < 1:
        raise ValueError("limit must be positive")
    rows = tuple(roster)
    ids = [item.external_player_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("roster external_player_id values must be unique")
    if evidence.jersey_number is None or evidence.status != "reliable":
        return RosterCandidateSet(
            subject_id=evidence.subject_id,
            candidates=(),
            reason="reliable-jersey-required",
        )

    normalized_team = str(team_id).strip() or None if team_id is not None else None
    candidates: list[RosterCandidate] = []
    for player in rows:
        if player.jersey_number != evidence.jersey_number:
            continue
        if normalized_team is not None and player.team_id == normalized_team:
            multiplier = 1.0
            team_reason = "team-match"
        elif normalized_team is None or player.team_id is None:
            multiplier = 0.90
            team_reason = "team-unavailable"
        else:
            multiplier = 0.50
            team_reason = "team-conflict"
        candidates.append(
            RosterCandidate(
                external_player_id=player.external_player_id,
                display_name=player.display_name,
                jersey_number=player.jersey_number,
                team_id=player.team_id,
                role=player.role,
                score=evidence.confidence * multiplier,
                reasons=("reliable-jersey-number-match", team_reason),
            )
        )
    candidates.sort(
        key=lambda item: (-item.score, item.display_name.casefold(), item.external_player_id)
    )
    return RosterCandidateSet(
        subject_id=evidence.subject_id,
        candidates=tuple(candidates[: int(limit)]),
        reason="manual-confirmation-required" if candidates else "no-number-match",
    )


__all__ = [
    "JerseyEvidenceSummary",
    "JerseyFusionConfig",
    "JerseyOcrObservation",
    "JerseyVote",
    "RosterCandidate",
    "RosterCandidateSet",
    "RosterPlayer",
    "aggregate_canonical_people",
    "aggregate_tracklet_evidence",
    "aggregate_tracklets",
    "generate_roster_candidates",
    "normalize_jersey_number",
]
