from __future__ import annotations

import pytest

from app.closed_set_roster_resolution import resolve_closed_set_roster
from app.roster_identity_contract import (
    AttributeEvidence,
    CanonicalPersonEvidence,
    ParticipationEvidence,
    PersistedRosterPlayer,
    PlayerLikelihoodEvidence,
    RosterResolverConfig,
)
from app.roster_identity_temporal import TimeInterval, intervals_overlap, merge_intervals


def _attribute(
    value,
    confidence: float = 0.95,
    *,
    source: str = "test",
    support_count: int = 2,
    confirmed: bool = False,
) -> AttributeEvidence:
    return AttributeEvidence(
        value=value,
        confidence=confidence,
        source=source,
        support_count=support_count,
        confirmed=confirmed,
    )


def _person(identifier: str, **values) -> CanonicalPersonEvidence:
    return CanonicalPersonEvidence(
        canonical_person_id=identifier,
        visible_intervals=values.pop("visible_intervals", (TimeInterval(100, 110),)),
        **values,
    )


def _player(identifier: str, **values) -> PersistedRosterPlayer:
    return PersistedRosterPlayer(
        external_player_id=identifier,
        display_name=values.pop("display_name", identifier.title()),
        active_intervals=values.pop("active_intervals", (TimeInterval(0, 5400),)),
        **values,
    )


def _resolution(result, identifier: str):
    return next(item for item in result.people if item.canonical_person_id == identifier)


def _candidate(result, person_id: str, player_id: str):
    person = _resolution(result, person_id)
    return next(item for item in person.candidates if item.external_player_id == player_id)


def test_repeated_jersey_team_role_and_time_produce_review_suggestion_only() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-8",
                team=_attribute("team-home"),
                role=_attribute("midfielder"),
                jersey_number=_attribute("08", confidence=0.91, source="jersey-ocr"),
            )
        ],
        [
            _player(
                "player-8",
                team_id="team-home",
                jersey_number="8",
                role="Central Midfielder",
            ),
            _player(
                "player-9",
                team_id="team-home",
                jersey_number="9",
                role="Forward",
            ),
        ],
    )

    person = _resolution(result, "canonical-8")
    assert person.status == "suggested"
    assert person.suggested_external_player_id == "player-8"
    assert result.diagnostics["automaticBindingCount"] == 0
    candidate = _candidate(result, "canonical-8", "player-8")
    assert candidate.proposal_status == "selected"
    assert candidate.requires_manual_confirmation is True
    assert {item.code for item in candidate.evidence} >= {
        "team-match",
        "jersey-number-match",
        "role-match",
        "active-time-coverage",
    }


def test_team_role_and_availability_without_identity_signal_abstain() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-anonymous",
                team=_attribute("team-home"),
                role=_attribute("player"),
            )
        ],
        [
            _player("player-a", team_id="team-home", role="Defender"),
            _player("player-b", team_id="team-home", role="Forward"),
        ],
    )

    person = result.people[0]
    assert person.status == "abstain"
    assert person.suggested_external_player_id is None
    assert person.reasons == ("insufficient-identity-evidence",)
    assert all(item.identity_signal_score == 0.0 for item in person.candidates)


def test_single_jersey_crop_is_not_enough_to_force_closed_set_identity() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-single-crop",
                team=_attribute("home"),
                role=_attribute("player"),
                jersey_number=_attribute(
                    "8", confidence=0.78, support_count=1, source="jersey-ocr"
                ),
            )
        ],
        [_player("player-8", team_id="home", jersey_number="8", role="Defender")],
    )

    person = result.people[0]
    candidate = person.candidates[0]
    assert person.status == "abstain"
    assert candidate.identity_signal_score < RosterResolverConfig().min_identity_signal_score
    assert "insufficient-identity-evidence" in person.reasons


def test_reliable_jersey_and_team_mismatches_are_hard_gates() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-8",
                team=_attribute("home", confidence=0.9),
                jersey_number=_attribute("8", confidence=0.9, support_count=3),
            )
        ],
        [
            _player("wrong-number", team_id="home", jersey_number="9"),
            _player("wrong-team", team_id="away", jersey_number="8"),
        ],
    )

    assert _candidate(result, "canonical-8", "wrong-number").conflicts == (
        "jersey-number-mismatch-hard",
    )
    assert _candidate(result, "canonical-8", "wrong-team").conflicts == (
        "team-mismatch-hard",
    )
    assert result.people[0].status == "abstain"


def test_low_confidence_attribute_mismatch_is_penalty_not_hard_rejection() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-direct",
                team=_attribute("home", confidence=0.4),
                jersey_number=_attribute("7", confidence=0.4, support_count=1),
                player_likelihoods=(
                    PlayerLikelihoodEvidence(
                        external_player_id="player-8",
                        confidence=0.99,
                        source="face-gallery",
                        evidence_id="face-1",
                    ),
                ),
            )
        ],
        [_player("player-8", team_id="away", jersey_number="8")],
    )

    candidate = result.people[0].candidates[0]
    assert candidate.conflicts == ()
    assert "team-mismatch-soft" in candidate.reasons
    assert "jersey-number-mismatch-soft" in candidate.reasons
    assert result.people[0].status == "suggested"


def test_goalkeeper_role_and_inactive_window_are_constraints() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-gk",
                role=_attribute("goalkeeper", confidence=0.95),
                jersey_number=_attribute("1"),
                visible_intervals=(TimeInterval(3000, 3010),),
            )
        ],
        [
            _player(
                "outfield-1",
                jersey_number="1",
                role="Defender",
                active_intervals=(TimeInterval(0, 5400),),
            ),
            _player(
                "sub-gk",
                jersey_number="1",
                role="Goalkeeper",
                active_intervals=(TimeInterval(4000, 5400),),
            ),
        ],
    )

    assert "role-mismatch-hard" in _candidate(
        result, "canonical-gk", "outfield-1"
    ).conflicts
    assert "player-inactive-at-observation-time" in _candidate(
        result, "canonical-gk", "sub-gk"
    ).conflicts
    assert result.people[0].status == "abstain"


def test_player_specific_event_can_identify_without_jersey_number() -> None:
    event = ParticipationEvidence(
        event_id="event-goal-71", kind="goal", match_time_seconds=4260
    )
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-scorer",
                team=_attribute("home"),
                role=_attribute("forward"),
                participation=(event,),
                visible_intervals=(TimeInterval(4255, 4265),),
            )
        ],
        [
            _player(
                "scorer",
                team_id="home",
                role="Forward",
                participation=(event,),
            ),
            _player("other", team_id="home", role="Forward"),
        ],
    )

    person = result.people[0]
    assert person.status == "suggested"
    assert person.suggested_external_player_id == "scorer"
    assert "player-event-match" in _candidate(
        result, "canonical-scorer", "scorer"
    ).reasons


def test_global_assignment_is_one_to_one_and_uses_joint_evidence() -> None:
    people = [
        _person(
            "canonical-a",
            team=_attribute("home"),
            player_likelihoods=(
                PlayerLikelihoodEvidence("player-1", 0.92, "gallery", "a-1"),
                PlayerLikelihoodEvidence("player-2", 0.80, "gallery", "a-2"),
            ),
        ),
        _person(
            "canonical-b",
            team=_attribute("home"),
            player_likelihoods=(
                PlayerLikelihoodEvidence("player-1", 0.83, "gallery", "b-1"),
                PlayerLikelihoodEvidence("player-2", 0.96, "gallery", "b-2"),
            ),
        ),
    ]
    result = resolve_closed_set_roster(
        people,
        [
            _player("player-1", team_id="home"),
            _player("player-2", team_id="home"),
        ],
    )

    assert {
        item.canonical_person_id: item.suggested_external_player_id
        for item in result.people
    } == {"canonical-a": "player-1", "canonical-b": "player-2"}
    assert result.diagnostics["oneToOneSuggestions"] is True


def test_equal_closed_set_candidates_abstain_on_global_assignment_margin() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-8",
                team=_attribute("home"),
                jersey_number=_attribute("8"),
            )
        ],
        [
            _player("home-8-a", team_id="home", jersey_number="8"),
            _player("home-8-b", team_id="home", jersey_number="8"),
        ],
    )

    person = result.people[0]
    assert person.status == "abstain"
    assert "global-assignment-ambiguous" in person.reasons
    assert any(item.proposal_status == "ambiguous" for item in person.candidates)


def test_confirmed_binding_is_preserved_despite_number_team_role_and_time_conflicts() -> None:
    person = _person(
        "canonical-manual",
        team=_attribute("home", confirmed=True),
        role=_attribute("goalkeeper", confirmed=True),
        jersey_number=_attribute("1", confirmed=True),
        visible_intervals=(TimeInterval(100, 110),),
        confirmed_external_player_id="away-9",
    )
    result = resolve_closed_set_roster(
        [person],
        [
            _player(
                "away-9",
                team_id="away",
                jersey_number="9",
                role="Forward",
                active_intervals=(TimeInterval(200, 300),),
            )
        ],
    )

    resolution = result.people[0]
    assert resolution.status == "confirmed"
    assert resolution.confirmed_external_player_id == "away-9"
    assert resolution.suggested_external_player_id is None
    assert set(resolution.conflicts) == {
        "team-mismatch-hard",
        "jersey-number-mismatch-hard",
        "role-mismatch-hard",
        "player-inactive-at-observation-time",
    }
    assert resolution.candidates[0].proposal_status == "confirmed"
    assert resolution.candidates[0].identity_signal_score == 1.0


def test_missing_confirmed_player_is_retained_and_reported() -> None:
    result = resolve_closed_set_roster(
        [_person("canonical-manual", confirmed_external_player_id="missing-player")],
        [_player("other-player")],
    )

    resolution = result.people[0]
    assert resolution.status == "confirmed"
    assert resolution.confirmed_external_player_id == "missing-player"
    assert "confirmed-player-missing-from-persisted-roster" in resolution.conflicts
    missing = next(
        item
        for item in resolution.candidates
        if item.external_player_id == "missing-player"
    )
    assert missing.proposal_status == "confirmed"
    assert missing.eligible is False


def test_confirmed_player_is_reserved_from_all_automatic_suggestions() -> None:
    result = resolve_closed_set_roster(
        [
            _person("canonical-owner", confirmed_external_player_id="player-8"),
            _person(
                "canonical-other",
                jersey_number=_attribute("8"),
                player_likelihoods=(
                    PlayerLikelihoodEvidence(
                        "player-8", 0.99, "face-gallery", "other-face"
                    ),
                ),
            ),
        ],
        [_player("player-8", jersey_number="8")],
    )

    other = _resolution(result, "canonical-other")
    assert other.status == "abstain"
    candidate = _candidate(result, "canonical-other", "player-8")
    assert "player-reserved-by-confirmed-binding" in candidate.conflicts
    assert candidate.proposal_status == "blocked"


def test_duplicate_simultaneous_confirmed_bindings_remain_manual_but_raise_conflict() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-left",
                confirmed_external_player_id="player-8",
                visible_intervals=(TimeInterval(10, 20),),
            ),
            _person(
                "canonical-right",
                confirmed_external_player_id="player-8",
                visible_intervals=(TimeInterval(15, 25),),
            ),
        ],
        [_player("player-8")],
    )

    assert all(item.status == "confirmed" for item in result.people)
    assert {item.code for item in result.conflicts} == {
        "duplicate-confirmed-player-binding",
        "simultaneous-confirmed-player-duplicate",
    }
    assert all(
        "simultaneous-confirmed-player-duplicate" in item.conflicts
        for item in result.people
    )
    assert result.diagnostics["confirmedBindingConflictCount"] == 2


def test_non_overlapping_duplicate_confirmed_binding_still_violates_global_uniqueness() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "fragment-a",
                confirmed_external_player_id="player-8",
                visible_intervals=(TimeInterval(0, 10),),
            ),
            _person(
                "fragment-b",
                confirmed_external_player_id="player-8",
                visible_intervals=(TimeInterval(20, 30),),
            ),
        ],
        [_player("player-8")],
    )

    assert [item.code for item in result.conflicts] == [
        "duplicate-confirmed-player-binding"
    ]
    assert all(
        "duplicate-confirmed-player-binding" in item.conflicts
        for item in result.people
    )


def test_payload_contract_exposes_suggestion_but_never_auto_binding() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-8",
                jersey_number=_attribute("8"),
            )
        ],
        [_player("player-8", jersey_number="8")],
    )

    payload = result.to_payload()
    assert payload["schemaVersion"] == 1
    assert payload["people"][0]["suggestedExternalPlayerId"] == "player-8"
    assert payload["people"][0]["requiresManualConfirmation"] is True
    assert payload["people"][0]["candidates"][0][
        "requiresManualConfirmation"
    ] is True


def test_result_is_deterministic_for_reversed_inputs() -> None:
    people = [
        _person(
            "canonical-b",
            jersey_number=_attribute("9"),
        ),
        _person(
            "canonical-a",
            jersey_number=_attribute("8"),
        ),
    ]
    players = [
        _player("player-9", jersey_number="9"),
        _player("player-8", jersey_number="8"),
    ]

    forward = resolve_closed_set_roster(people, players).to_payload()
    reversed_result = resolve_closed_set_roster(
        reversed(people), reversed(players)
    ).to_payload()

    assert forward == reversed_result


def test_duplicate_ids_and_invalid_manual_state_fail_before_resolution() -> None:
    with pytest.raises(ValueError, match="external_player_id values must be unique"):
        resolve_closed_set_roster(
            [_person("canonical")],
            [_player("duplicate"), _player("duplicate")],
        )

    with pytest.raises(ValueError, match="confirmed player cannot also be manually excluded"):
        _person(
            "invalid",
            confirmed_external_player_id="player-8",
            excluded_external_player_ids=("player-8",),
        )


def test_half_open_active_intervals_do_not_overlap_only_at_substitution_boundary() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-boundary",
                jersey_number=_attribute("8"),
                visible_intervals=(TimeInterval(2700, 2710),),
            )
        ],
        [
            _player(
                "subbed-off",
                jersey_number="8",
                active_intervals=(TimeInterval(0, 2700),),
            )
        ],
        config=RosterResolverConfig(availability_tolerance_seconds=0.0),
    )

    candidate = result.people[0].candidates[0]
    assert "player-inactive-at-observation-time" in candidate.conflicts
    assert result.people[0].status == "abstain"


def test_point_overlap_is_symmetric_exact_and_preserves_half_open_endpoint() -> None:
    point_five = (TimeInterval(5, 5),)
    point_ten = (TimeInterval(10, 10),)
    span = (TimeInterval(0, 10),)

    assert intervals_overlap(point_five, span) is True
    assert intervals_overlap(span, point_five) is True
    assert intervals_overlap(point_five, point_five) is True
    assert intervals_overlap(point_five, point_ten) is False
    assert intervals_overlap(point_ten, span) is False
    assert intervals_overlap(span, point_ten) is False
    assert merge_intervals((*span, *point_ten)) == (
        TimeInterval(0, 10),
        TimeInterval(10, 10),
    )


def test_distinct_point_activity_does_not_create_candidate() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-point",
                jersey_number=_attribute("8"),
                visible_intervals=(TimeInterval(10, 10),),
            )
        ],
        [
            _player(
                "player-point",
                jersey_number="8",
                active_intervals=(TimeInterval(5, 5),),
            )
        ],
        config=RosterResolverConfig(availability_tolerance_seconds=0.0),
    )

    candidate = result.people[0].candidates[0]
    assert candidate.proposal_status == "blocked"
    assert "player-inactive-at-observation-time" in candidate.conflicts


def test_endpoint_point_is_included_in_full_activity_coverage_gate() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-endpoint",
                jersey_number=_attribute("8"),
                visible_intervals=(TimeInterval(0, 10), TimeInterval(10, 10)),
            )
        ],
        [
            _player(
                "player-8",
                jersey_number="8",
                active_intervals=(TimeInterval(0, 10),),
            )
        ],
        config=RosterResolverConfig(availability_tolerance_seconds=0.0),
    )

    candidate = result.people[0].candidates[0]
    assert "player-not-active-for-full-visible-interval" in candidate.conflicts
    assert candidate.proposal_status == "blocked"


def test_all_visible_intervals_must_be_covered_by_player_activity() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-two-ranges",
                jersey_number=_attribute("8"),
                visible_intervals=(TimeInterval(0, 10), TimeInterval(20, 30)),
            )
        ],
        [
            _player(
                "player-8",
                jersey_number="8",
                active_intervals=(TimeInterval(0, 10),),
            )
        ],
        config=RosterResolverConfig(availability_tolerance_seconds=0.0),
    )

    candidate = result.people[0].candidates[0]
    assert "player-not-active-for-full-visible-interval" in candidate.conflicts
    assert candidate.eligible is False
    assert result.people[0].status == "abstain"


def test_small_activity_clock_error_is_absorbed_by_documented_tolerance() -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-near-boundary",
                jersey_number=_attribute("8"),
                visible_intervals=(TimeInterval(99.5, 100.5),),
            )
        ],
        [
            _player(
                "player-8",
                jersey_number="8",
                active_intervals=(TimeInterval(0, 100),),
            )
        ],
        config=RosterResolverConfig(availability_tolerance_seconds=1.0),
    )

    candidate = result.people[0].candidates[0]
    assert candidate.eligible is True
    assert "active-time-coverage" in candidate.reasons
    assert result.people[0].status == "suggested"


def test_duplicate_namespaced_event_ids_are_rejected_deterministically() -> None:
    first = ParticipationEvidence("event-1", "goal", 100, source="provider")
    second = ParticipationEvidence("event-1", "goal", 101, source="PROVIDER")

    with pytest.raises(ValueError, match="duplicate namespaced event IDs"):
        _person("canonical", participation=(first, second))
    with pytest.raises(ValueError, match="duplicate namespaced event IDs"):
        _player("player", participation=(second, first))


@pytest.mark.parametrize(
    ("canonical_event", "roster_event", "expected_conflict", "expected_reason"),
    [
        (
            ParticipationEvidence("event-1", "goal", 100, source="video"),
            ParticipationEvidence("event-1", "goal", 100, source="api"),
            None,
            "event-source-mismatch",
        ),
        (
            ParticipationEvidence("event-1", "goal", 100, source="api"),
            ParticipationEvidence("event-1", "red-card", 100, source="api"),
            "event-kind-mismatch",
            None,
        ),
        (
            ParticipationEvidence("event-1", "goal", 100, source="api"),
            ParticipationEvidence("event-1", "goal", 120, source="api"),
            "event-time-mismatch",
            None,
        ),
    ],
)
def test_event_identity_requires_source_kind_and_time_compatibility(
    canonical_event,
    roster_event,
    expected_conflict,
    expected_reason,
) -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-event",
                visible_intervals=(TimeInterval(95, 105),),
                participation=(canonical_event,),
            )
        ],
        [
            _player(
                "player-event",
                active_intervals=(TimeInterval(0, 200),),
                participation=(roster_event,),
            )
        ],
    )

    candidate = result.people[0].candidates[0]
    assert "player-event-match" not in candidate.reasons
    if expected_conflict is not None:
        assert expected_conflict in candidate.conflicts
        assert candidate.proposal_status == "blocked"
    if expected_reason is not None:
        assert expected_reason in candidate.reasons
    assert result.people[0].status == "abstain"


def test_event_must_be_visible_and_inside_player_activity() -> None:
    outside_visibility = resolve_closed_set_roster(
        [
            _person(
                "canonical-off-event",
                visible_intervals=(TimeInterval(0, 10),),
                participation=(ParticipationEvidence("event-1", "goal", 100),),
            )
        ],
        [
            _player(
                "player",
                active_intervals=(TimeInterval(0, 200),),
                participation=(ParticipationEvidence("event-1", "goal", 100),),
            )
        ],
    )
    candidate = outside_visibility.people[0].candidates[0]
    assert "event-outside-canonical-visible-interval" in candidate.reasons
    assert "player-event-match" not in candidate.reasons

    outside_activity = resolve_closed_set_roster(
        [
            _person(
                "canonical-event",
                visible_intervals=(TimeInterval(99, 100),),
                participation=(ParticipationEvidence("event-1", "goal", 100),),
            )
        ],
        [
            _player(
                "player",
                active_intervals=(TimeInterval(90, 100.5),),
                participation=(ParticipationEvidence("event-1", "goal", 101),),
            )
        ],
        config=RosterResolverConfig(
            availability_tolerance_seconds=0.0,
            event_visibility_tolerance_seconds=1.0,
        ),
    )
    candidate = outside_activity.people[0].candidates[0]
    assert "event-outside-player-active-interval" in candidate.conflicts
    assert candidate.proposal_status == "blocked"


@pytest.mark.parametrize("role", ["referee", "coach", "other"])
@pytest.mark.parametrize("roster_role", [None, "Defender"])
def test_confirmed_non_player_role_blocks_player_roster_even_without_position(
    role: str, roster_role: str | None
) -> None:
    result = resolve_closed_set_roster(
        [
            _person(
                "canonical-non-player",
                role=_attribute(role, confirmed=True),
                player_likelihoods=(
                    PlayerLikelihoodEvidence("player", 1.0, "gallery", "face"),
                ),
            )
        ],
        [_player("player", role=roster_role)],
    )

    candidate = result.people[0].candidates[0]
    assert "confirmed-non-player-role" in candidate.conflicts
    assert candidate.proposal_status == "blocked"
    assert result.people[0].status == "abstain"


def test_zero_assignment_margin_still_abstains_on_exact_tie() -> None:
    result = resolve_closed_set_roster(
        [_person("canonical", jersey_number=_attribute("8"))],
        [
            _player("player-a", jersey_number="8"),
            _player("player-b", jersey_number="8"),
        ],
        config=RosterResolverConfig(assignment_margin=0.0),
    )

    assert result.people[0].status == "abstain"
    assert "global-assignment-ambiguous" in result.people[0].reasons


def test_candidate_serialization_distinguishes_blocked_confirmed_and_reviewable() -> None:
    blocked = resolve_closed_set_roster(
        [
            _person(
                "canonical-blocked",
                team=_attribute("home", confirmed=True),
                jersey_number=_attribute("8"),
            )
        ],
        [_player("away-8", team_id="away", jersey_number="8")],
    ).to_payload()["people"][0]["candidates"][0]
    assert blocked["proposalStatus"] == "blocked"
    assert blocked["eligible"] is False
    assert blocked["requiresManualConfirmation"] is False

    confirmed = resolve_closed_set_roster(
        [_person("canonical-confirmed", confirmed_external_player_id="player-8")],
        [_player("player-8")],
    ).to_payload()["people"][0]["candidates"][0]
    assert confirmed["proposalStatus"] == "confirmed"
    assert confirmed["requiresManualConfirmation"] is False

    selected = resolve_closed_set_roster(
        [_person("canonical-selected", jersey_number=_attribute("8"))],
        [_player("player-8", jersey_number="8")],
    ).to_payload()["people"][0]["candidates"][0]
    assert selected["proposalStatus"] == "selected"
    assert selected["requiresManualConfirmation"] is True
