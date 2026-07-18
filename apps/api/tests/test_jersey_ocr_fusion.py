from __future__ import annotations

import itertools
import random

import pytest

from app.identity_resolution_contract import IdentityTracklet
from app.identity_resolver import resolve_identities
from app.jersey_ocr_contract import (
    JerseyFusionConfig,
    JerseyOcrObservation,
    normalize_jersey_number,
)
from app.jersey_ocr_fusion import (
    aggregate_canonical_people,
    aggregate_tracklet_evidence,
    aggregate_tracklets,
)
from app.jersey_roster_candidates import (
    RosterPlayer,
    generate_roster_candidates,
)


def _observation(
    identifier: str,
    timestamp: float,
    number: str | int | None = 8,
    *,
    tracklet_id: str = "tracklet-a",
    confidence: float = 0.95,
    quality: float = 1.0,
    visibility: float = 1.0,
    frame_index: int | None = None,
    evidence_fingerprint: str | None = None,
) -> JerseyOcrObservation:
    return JerseyOcrObservation(
        id=identifier,
        tracklet_id=tracklet_id,
        timestamp_seconds=timestamp,
        raw_number=number,
        ocr_confidence=confidence,
        frame_quality=quality,
        back_visibility=visibility,
        frame_index=frame_index,
        evidence_fingerprint=evidence_fingerprint,
    )


def _reliable(
    tracklet_id: str,
    number: int,
    *,
    start: float = 0.0,
    sample_count: int = 2,
):
    return aggregate_tracklet_evidence(
        tracklet_id,
        [
            _observation(
                f"{tracklet_id}-{index}",
                start + index,
                number,
                tracklet_id=tracklet_id,
            )
            for index in range(sample_count)
        ],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (False, None),
        (True, None),
        (0, "0"),
        (8, "8"),
        ("08", "8"),
        (" # 08 ", "8"),
        ("99", "99"),
        ("", None),
        ("-1", None),
        (-1, None),
        (100, None),
        ("1O", None),
        ("O8", None),
        ("８", None),
        ("8.0", None),
    ],
)
def test_normalize_jersey_number_is_conservative(value, expected) -> None:
    assert normalize_jersey_number(value) == expected


def test_number_range_can_be_restricted_by_the_caller() -> None:
    assert normalize_jersey_number(26, minimum=1, maximum=26) == "26"
    assert normalize_jersey_number(0, minimum=1, maximum=26) is None
    assert normalize_jersey_number(27, minimum=1, maximum=26) is None


def test_identical_pixel_fingerprints_cannot_manufacture_repeated_ocr_support() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation(
                "crop-a",
                0.0,
                evidence_fingerprint="pixel-evidence-v1:same",
            ),
            _observation(
                "crop-b",
                1.0,
                evidence_fingerprint="pixel-evidence-v1:same",
            ),
        ],
    )

    assert result.status == "provisional"
    assert result.support_count == 1
    assert result.selected_sample_count == 1
    assert result.rejection_counts["duplicate-evidence-fingerprint"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ocr_confidence", -0.1),
        ("ocr_confidence", 1.1),
        ("ocr_confidence", float("nan")),
        ("frame_quality", -0.1),
        ("frame_quality", float("inf")),
        ("back_visibility", 1.1),
    ],
)
def test_observation_rejects_invalid_probability_fields(field: str, value: float) -> None:
    values = {
        "id": "obs",
        "tracklet_id": "tracklet",
        "timestamp_seconds": 0.0,
        "raw_number": 8,
        "ocr_confidence": 0.9,
        "frame_quality": 0.9,
        "back_visibility": 0.9,
    }
    values[field] = value
    with pytest.raises(ValueError):
        JerseyOcrObservation(**values)


@pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf")])
def test_observation_rejects_invalid_timestamp(timestamp: float) -> None:
    with pytest.raises(ValueError):
        _observation("obs", timestamp)


def test_empty_tracklet_has_nullable_number_and_no_evidence() -> None:
    result = aggregate_tracklet_evidence("tracklet-a", [])

    assert result.status == "no-evidence"
    assert result.jersey_number is None
    assert result.candidate_number is None
    assert result.confidence == 0.0
    assert result.identity_resolver_fields() == {
        "jersey_number": None,
        "jersey_confidence": 0.0,
        "jersey_sample_count": 0,
    }


def test_single_good_reading_is_provisional_and_never_published() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [_observation("only", 0.0, "08")],
    )

    assert result.status == "provisional"
    assert result.candidate_number == "8"
    assert result.jersey_number is None
    assert result.support_count == 1
    assert result.identity_resolver_fields()["jersey_number"] is None


def test_matching_spaced_readings_publish_reliable_number() -> None:
    result = _reliable("tracklet-a", 8)

    assert result.status == "reliable"
    assert result.jersey_number == "8"
    assert result.candidate_number == "8"
    assert result.support_count == 2
    assert result.confidence >= 0.8
    assert result.identity_resolver_fields() == {
        "jersey_number": "8",
        "jersey_confidence": result.confidence,
        "jersey_sample_count": 2,
    }


def test_best_frame_sampling_keeps_only_highest_quality_reading_per_window() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("weak-wrong", 0.10, 9, confidence=0.80),
            _observation("best", 0.20, 8, confidence=0.96),
            _observation("second-window", 0.60, 8, confidence=0.94),
        ],
    )

    assert [item.id for item in result.selected_observations] == [
        "best",
        "second-window",
    ]
    assert result.jersey_number == "8"
    assert result.rejection_counts["inferior-frame-same-window"] == 1


def test_best_frame_tie_break_is_deterministic_and_prefers_earliest_frame() -> None:
    rows = [
        _observation("later", 0.20, 9, frame_index=20),
        _observation("earlier", 0.10, 8, frame_index=10),
    ]

    for order in (rows, list(reversed(rows))):
        result = aggregate_tracklet_evidence("tracklet-a", order)
        assert [item.id for item in result.selected_observations] == ["earlier"]


def test_low_quality_missing_and_invalid_readings_are_rejected_with_diagnostics() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("missing", 0.0, None),
            _observation("ambiguous", 1.0, "O8"),
            _observation("low-ocr", 2.0, 8, confidence=0.4),
            _observation("bad-frame", 3.0, 8, quality=0.2),
            _observation("no-back", 4.0, 8, visibility=0.2),
            _observation("low-product", 5.0, 8, confidence=0.6, quality=0.6, visibility=0.6),
        ],
    )

    assert result.status == "no-evidence"
    assert result.rejection_counts == {
        "invalid-or-missing-number": 2,
        "ocr-confidence-low": 1,
        "frame-quality-low": 1,
        "back-visibility-low": 1,
        "effective-score-low": 1,
    }


def test_duplicate_observation_ids_cannot_inflate_support() -> None:
    duplicate = _observation("replayed-message", 0.0, 8)
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [duplicate, duplicate, _observation("independent", 1.0, 8)],
    )

    assert result.status == "provisional"
    assert result.jersey_number is None
    assert result.support_count == 1
    assert result.rejection_counts["duplicate-observation-id"] == 2


def test_balanced_conflicting_numbers_fail_closed() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("eight", 0.0, 8),
            _observation("nine", 1.0, 9),
        ],
    )

    assert result.status == "conflict"
    assert result.jersey_number is None
    assert result.candidate_number is None
    assert result.confidence == 0.0
    assert "competing-jersey-numbers" in result.conflict_reasons
    assert result.identity_resolver_fields()["jersey_sample_count"] == 0


def test_dominant_consistent_vote_can_survive_one_low_weight_outlier() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("eight-1", 0.0, 8),
            _observation("eight-2", 1.0, 8),
            _observation("eight-3", 2.0, 8),
            _observation("nine-low", 3.0, 9, confidence=0.55),
        ],
    )

    assert result.status == "reliable"
    assert result.jersey_number == "8"
    assert result.votes[0].support_count == 3
    assert result.votes[1].number == "9"


def test_two_to_one_high_confidence_majority_is_reliable() -> None:
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("eight-1", 0.0, 8, confidence=0.90),
            _observation("eight-2", 1.0, 8, confidence=0.90),
            _observation("three", 2.0, 3, confidence=0.90),
        ],
    )

    assert result.status == "reliable"
    assert result.jersey_number == "8"


def test_max_sample_cap_uses_best_frames_not_first_frames() -> None:
    config = JerseyFusionConfig(max_selected_frames=2)
    result = aggregate_tracklet_evidence(
        "tracklet-a",
        [
            _observation("early-low", 0.0, 9, confidence=0.60),
            _observation("best", 1.0, 8, confidence=0.99),
            _observation("second-best", 2.0, 8, confidence=0.98),
        ],
        config=config,
    )

    assert [item.id for item in result.selected_observations] == ["best", "second-best"]
    assert result.jersey_number == "8"
    assert result.rejection_counts["sample-cap"] == 1


def test_tracklet_aggregator_groups_interleaved_observations() -> None:
    rows = [
        _observation("a-1", 0.0, 8, tracklet_id="a"),
        _observation("b-1", 0.0, 9, tracklet_id="b"),
        _observation("a-2", 1.0, 8, tracklet_id="a"),
        _observation("b-2", 1.0, 9, tracklet_id="b"),
    ]

    result = aggregate_tracklets(rows)

    assert list(result) == ["a", "b"]
    assert result["a"].jersey_number == "8"
    assert result["b"].jersey_number == "9"


def test_single_tracklet_aggregator_rejects_mixed_input() -> None:
    with pytest.raises(ValueError, match="other tracklets"):
        aggregate_tracklet_evidence(
            "a",
            [_observation("b", 0.0, tracklet_id="b")],
        )


def test_canonical_aggregation_combines_consistent_tracklets() -> None:
    summaries = {
        "a": _reliable("a", 8, start=0.0),
        "b": _reliable("b", 8, start=10.0),
    }

    result = aggregate_canonical_people(summaries, {"a": "person-1", "b": "person-1"})[
        "person-1"
    ]

    assert result.scope == "canonical"
    assert result.tracklet_ids == ("a", "b")
    assert result.jersey_number == "8"
    assert result.support_count == 4


def test_canonical_aggregation_preserves_member_rejection_diagnostics() -> None:
    summary = aggregate_tracklet_evidence(
        "a",
        [
            _observation("valid-1", 0.0, 8, tracklet_id="a"),
            _observation("invalid", 0.5, "O8", tracklet_id="a"),
            _observation("valid-2", 1.0, 8, tracklet_id="a"),
        ],
    )

    canonical = aggregate_canonical_people(
        {"a": summary},
        {"a": "person"},
    )["person"]

    assert canonical.jersey_number == "8"
    assert canonical.rejection_counts["invalid-or-missing-number"] == 1


def test_reliable_tracklet_disagreement_is_a_hard_canonical_conflict() -> None:
    # The many observations for ``a`` would dominate raw weighted voting.  A
    # separately reliable ``b`` still makes the canonical result fail closed.
    summaries = {
        "a": _reliable("a", 8, sample_count=8),
        "b": _reliable("b", 9, start=20.0, sample_count=2),
    }

    result = aggregate_canonical_people(summaries, {"a": "person-1", "b": "person-1"})[
        "person-1"
    ]

    assert result.status == "conflict"
    assert result.jersey_number is None
    assert result.candidate_number is None
    assert result.confidence == 0.0
    assert "reliable-tracklet-jersey-conflict" in result.conflict_reasons


def test_canonical_aggregation_requires_explicit_complete_mapping() -> None:
    with pytest.raises(ValueError, match="missing canonical mapping"):
        aggregate_canonical_people({"a": _reliable("a", 8)}, {})


def test_canonical_aggregation_rejects_mismatched_mapping_key() -> None:
    with pytest.raises(ValueError, match="mapping key"):
        aggregate_canonical_people(
            {"wrong-key": _reliable("a", 8)},
            {"a": "person-1"},
        )


def test_summary_fields_are_compatible_with_identity_resolver_contract() -> None:
    evidence_a = _reliable("a", 8)
    evidence_b = _reliable("b", 8, start=3.0)
    tracklets = [
        IdentityTracklet(
            id="a",
            start_time=0.0,
            end_time=1.0,
            **evidence_a.identity_resolver_fields(),
        ),
        IdentityTracklet(
            id="b",
            start_time=3.0,
            end_time=4.0,
            **evidence_b.identity_resolver_fields(),
        ),
    ]

    result = resolve_identities(tracklets)

    assert len(result.groups) == 1
    assert result.groups[0].jersey_number == "8"
    assert "reliable-jersey-match" in result.accepted_edges[0].reasons


def test_unique_exact_roster_match_is_still_review_only() -> None:
    evidence = _reliable("a", 8)
    result = generate_roster_candidates(
        evidence,
        [RosterPlayer("player-8", "Alex Eight", 8, team_id="home")],
        team_id="home",
    )

    assert [item.external_player_id for item in result.candidates] == ["player-8"]
    assert result.candidates[0].requires_manual_confirmation is True
    assert result.requires_manual_confirmation is True
    assert result.to_payload()[0] == {
        "externalPlayerId": "player-8",
        "name": "Alex Eight",
        "number": "8",
        "teamId": "home",
        "position": None,
        "confidence": round(evidence.confidence, 6),
        "reasons": ["reliable-jersey-number-match", "team-match"],
        "requiresManualConfirmation": True,
    }


def test_roster_duplicate_numbers_are_all_preserved_and_team_match_ranks_first() -> None:
    evidence = _reliable("a", 8)
    result = generate_roster_candidates(
        evidence,
        [
            RosterPlayer("away-8", "Away Eight", "08", team_id="away"),
            RosterPlayer("unknown-8", "Unknown Eight", 8),
            RosterPlayer("home-8", "Home Eight", 8, team_id="home"),
            RosterPlayer("home-9", "Home Nine", 9, team_id="home"),
        ],
        team_id="home",
    )

    assert [item.external_player_id for item in result.candidates] == [
        "home-8",
        "unknown-8",
        "away-8",
    ]
    assert all(item.requires_manual_confirmation for item in result.candidates)


def test_roster_prior_cannot_generate_candidate_from_team_alone() -> None:
    evidence = aggregate_tracklet_evidence("a", [])
    result = generate_roster_candidates(
        evidence,
        [RosterPlayer("home-8", "Home Eight", 8, team_id="home")],
        team_id="home",
    )

    assert result.candidates == ()
    assert result.reason == "reliable-jersey-required"


def test_provisional_number_does_not_generate_roster_candidates() -> None:
    evidence = aggregate_tracklet_evidence("a", [_observation("one", 0.0, 8, tracklet_id="a")])
    result = generate_roster_candidates(
        evidence,
        [RosterPlayer("player-8", "Player Eight", 8)],
    )

    assert evidence.candidate_number == "8"
    assert result.candidates == ()


def test_roster_ids_must_be_unique() -> None:
    evidence = _reliable("a", 8)
    with pytest.raises(ValueError, match="must be unique"):
        generate_roster_candidates(
            evidence,
            [
                RosterPlayer("duplicate", "First", 8),
                RosterPlayer("duplicate", "Second", 8),
            ],
        )


def test_roster_limit_is_deterministic() -> None:
    evidence = _reliable("a", 8)
    result = generate_roster_candidates(
        evidence,
        [
            RosterPlayer("z", "Zulu", 8),
            RosterPlayer("a", "Alpha", 8),
            RosterPlayer("b", "Bravo", 8),
        ],
        limit=2,
    )
    assert [item.external_player_id for item in result.candidates] == ["a", "b"]


def test_property_fusion_is_invariant_under_input_permutation() -> None:
    rows = [
        _observation("a", 0.0, 8, confidence=0.92),
        _observation("b", 0.1, 9, confidence=0.70),
        _observation("c", 0.7, 8, confidence=0.95),
        _observation("d", 1.2, 8, confidence=0.93),
        _observation("e", 1.7, None, confidence=0.99),
    ]
    expected = aggregate_tracklet_evidence("tracklet-a", rows).to_payload()

    for permutation in itertools.permutations(rows):
        assert aggregate_tracklet_evidence(
            "tracklet-a", permutation
        ).to_payload() == expected


def test_property_dense_adjacent_frames_do_not_gain_independent_vote_weight() -> None:
    base = [
        _observation("correct-best", 0.10, 8, confidence=0.96),
        _observation("correct-2", 0.60, 8, confidence=0.95),
    ]
    dense_wrong = [
        _observation(f"wrong-{index}", 0.11 + index * 0.001, 9, confidence=0.90)
        for index in range(100)
    ]

    result = aggregate_tracklet_evidence("tracklet-a", [*base, *dense_wrong])

    assert result.jersey_number == "8"
    assert result.selected_sample_count == 2
    assert result.rejection_counts["inferior-frame-same-window"] == 100


def test_property_random_shuffle_does_not_change_tracklet_or_canonical_output() -> None:
    randomizer = random.Random(20260717)
    rows = [
        _observation(
            f"{tracklet}-{index}",
            float(index),
            8 if index != 3 else 9,
            tracklet_id=tracklet,
            confidence=0.95 if index != 3 else 0.55,
        )
        for tracklet in ("a", "b")
        for index in range(5)
    ]
    expected_tracklets = {
        key: value.to_payload() for key, value in aggregate_tracklets(rows).items()
    }
    expected_canonical = aggregate_canonical_people(
        aggregate_tracklets(rows),
        {"a": "person", "b": "person"},
    )["person"].to_payload()

    for _ in range(100):
        shuffled = rows[:]
        randomizer.shuffle(shuffled)
        summaries = aggregate_tracklets(shuffled)
        assert {key: value.to_payload() for key, value in summaries.items()} == expected_tracklets
        assert aggregate_canonical_people(
            summaries,
            {"a": "person", "b": "person"},
        )["person"].to_payload() == expected_canonical


def test_property_roster_candidate_generation_never_auto_binds() -> None:
    randomizer = random.Random(42)
    evidence = _reliable("a", 8)
    for size in range(1, 50):
        roster = [
            RosterPlayer(
                f"player-{size}-{index}",
                f"Player {index}",
                randomizer.randrange(1, 12),
                team_id=randomizer.choice(("home", "away", None)),
            )
            for index in range(size)
        ]
        result = generate_roster_candidates(evidence, roster, team_id="home")
        assert result.requires_manual_confirmation is True
        assert all(item.requires_manual_confirmation for item in result.candidates)
