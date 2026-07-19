"""Conservative sampling and fusion of correlated jersey OCR observations.

The detector/OCR worker is expected to emit many correlated readings for a
person tracklet.  This module deliberately does not treat every video frame as
an independent vote.  It keeps the best readable frame in each temporal
window, combines the remaining evidence, and publishes a jersey number only
when both support and confidence thresholds pass.

The algorithm is pure and reconstruction-agnostic. Contracts and review-only
roster ranking have their own owners so consumers do not depend on this engine
merely to exchange evidence DTOs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from math import floor
from typing import Iterable, Mapping, Sequence

from .jersey_ocr_contract import (
    JerseyEvidenceScope,
    JerseyEvidenceStatus,
    JerseyEvidenceSummary,
    JerseyFusionConfig,
    JerseyOcrObservation,
    JerseyVote,
    identifier,
    normalize_jersey_number,
)


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


def detect_jersey_number_switches(
    crops: Sequence[Mapping],
    *,
    min_stable_run: int = 3,
) -> list[dict]:
    """Flag tracklets whose recognized shirt number changes mid-track.

    A stable run of number A followed by a stable run of number B inside one
    tracklet is hard evidence of a tracker ID switch. This is deliberately a
    review flag, not an automatic split: noisy single misreads never qualify
    because both runs must be consecutive recognized reads of length
    ``min_stable_run``.
    """

    by_tracklet: dict[str, list[Mapping]] = {}
    for crop in crops:
        tracklet_id = str(crop.get("trackletId") or "")
        raw_number = crop.get("rawNumber")
        if not tracklet_id or crop.get("status") != "recognized" or not raw_number:
            continue
        by_tracklet.setdefault(tracklet_id, []).append(crop)
    suspects: list[dict] = []
    for tracklet_id, items in sorted(by_tracklet.items()):
        ordered = sorted(items, key=lambda item: float(item.get("timestamp") or 0.0))
        runs: list[dict] = []
        for item in ordered:
            number = str(item["rawNumber"])
            timestamp = float(item.get("timestamp") or 0.0)
            if runs and runs[-1]["number"] == number:
                runs[-1]["count"] += 1
                runs[-1]["lastTimestamp"] = timestamp
            else:
                runs.append(
                    {
                        "number": number,
                        "count": 1,
                        "firstTimestamp": timestamp,
                        "lastTimestamp": timestamp,
                    }
                )
        stable_runs = [run for run in runs if run["count"] >= min_stable_run]
        for first, second in zip(stable_runs, stable_runs[1:]):
            if first["number"] == second["number"]:
                continue
            suspects.append(
                {
                    "trackletId": tracklet_id,
                    "fromNumber": first["number"],
                    "toNumber": second["number"],
                    "switchTime": round(second["firstTimestamp"], 3),
                    "firstRunCount": first["count"],
                    "secondRunCount": second["count"],
                }
            )
            break
    return suspects


def aggregate_tracklet_evidence(
    tracklet_id: str,
    observations: Iterable[JerseyOcrObservation],
    *,
    config: JerseyFusionConfig | None = None,
) -> JerseyEvidenceSummary:
    """Sample and fuse OCR readings belonging to exactly one tracklet."""

    tracklet_identifier = identifier(tracklet_id, "tracklet_id")
    rows = tuple(observations)
    mismatches = sorted(
        {item.tracklet_id for item in rows if item.tracklet_id != tracklet_identifier}
    )
    if mismatches:
        raise ValueError(
            f"observations for {tracklet_identifier!r} include other tracklets: {mismatches!r}"
        )
    return _fuse(
        tracklet_identifier,
        "tracklet",
        rows,
        (tracklet_identifier,),
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
        canonical_id = identifier(canonical_id, "canonical person id")
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


__all__ = [
    "aggregate_canonical_people",
    "aggregate_tracklet_evidence",
    "aggregate_tracklets",
]
