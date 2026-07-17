from __future__ import annotations

from math import sqrt

import pytest

from app.identity_resolver import (
    IdentityResolverConfig,
    IdentityTracklet,
    resolve_global_identities,
    resolve_identities,
)


def _embedding_with_distance(distance: float) -> tuple[float, float]:
    cosine = 1.0 - distance
    return cosine, sqrt(max(0.0, 1.0 - cosine * cosine))


def _tracklet(
    identifier: str,
    start: float,
    end: float,
    *,
    embedding=(1.0, 0.0),
    **values,
) -> IdentityTracklet:
    reid_values = {} if embedding is None else {
        "mean_reid_embedding": embedding,
        # Most resolver tests model a tracklet, not a single crop.  Keep two
        # independent observations by default; single-crop behaviour is
        # covered explicitly below.
        "reid_embeddings": values.pop("reid_embeddings", (embedding, embedding)),
    }
    return IdentityTracklet(
        id=identifier,
        start_time=start,
        end_time=end,
        observation_count=values.pop("observation_count", 4),
        **reid_values,
        **values,
    )


def _reason(result, reason: str) -> bool:
    return any(reason in edge.reasons for edge in result.rejected_edges)


def test_tracklet_normalizes_mean_and_sample_embeddings() -> None:
    tracklet = IdentityTracklet(
        id="a",
        start_time=0.0,
        end_time=0.5,
        mean_reid_embedding=(3.0, 4.0),
        reid_embeddings=((0.0, 2.0), (0.0, 0.0)),
    )

    assert tracklet.mean_reid_embedding == pytest.approx((0.6, 0.8))
    assert tracklet.reid_embeddings == ((0.0, 1.0),)


def test_strong_reid_stitches_non_overlapping_tracklets() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                0.4,
                team_id="home",
                reid_embeddings=((1.0, 0.0), (0.999, 0.045)),
            ),
            _tracklet(
                "b",
                0.9,
                1.4,
                team_id="home",
                reid_embeddings=((1.0, 0.0), (0.998, 0.063)),
            ),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].tracklet_ids == ("a", "b")
    assert result.groups[0].status == "resolved"
    assert result.accepted_edges[0].reasons == ("strong-reid",)
    assert result.diagnostics["associationConfidenceP10"] is not None
    assert result.diagnostics["strongReidBidirectionalEdgeCount"] == 1
    assert result.diagnostics["groundTruthAvailable"] is False
    assert result.diagnostics["estimatedIdSwitchCount"] is None


def test_single_identical_crop_is_review_only_not_automatic_identity_proof() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a", 0.0, 0.4, team_id="home", reid_embeddings=((1.0, 0.0),)
            ),
            _tracklet(
                "b", 0.9, 1.4, team_id="home", reid_embeddings=((1.0, 0.0),)
            ),
        ]
    )

    assert len(result.groups) == 2
    assert result.accepted_edges == ()
    assert result.review_edges[0].reasons == ("review-reid",)
    assert result.review_edges[0].reid_strong_support_left == 1
    assert result.review_edges[0].reid_strong_support_right == 1


def test_review_reid_never_auto_stitches() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4),
            _tracklet("b", 0.9, 1.4, embedding=_embedding_with_distance(0.25)),
        ]
    )

    assert len(result.groups) == 2
    assert result.accepted_edges == ()
    assert result.review_edges[0].reasons == ("review-reid",)
    assert all(group.status == "provisional" for group in result.groups)


def test_one_accidental_sample_match_cannot_override_review_level_mean() -> None:
    left = _tracklet(
        "a",
        0.0,
        0.4,
        embedding=(1.0, 0.0),
        reid_embeddings=((1.0, 0.0), (0.0, 1.0), (0.7, 0.7)),
    )
    right = _tracklet(
        "b",
        0.9,
        1.4,
        embedding=_embedding_with_distance(0.25),
        reid_embeddings=((1.0, 0.0), (0.0, -1.0), (-0.7, -0.7)),
    )

    result = resolve_identities([left, right])

    assert result.accepted_edges == ()
    assert len(result.groups) == 2
    assert result.review_edges[0].reid_best_sample_distance == pytest.approx(0.0)
    assert result.review_edges[0].reid_robust_sample_distance > 0.1
    assert result.review_edges[0].reid_strong_support_left == 1
    assert result.review_edges[0].reid_strong_support_right == 1


def test_multiple_bidirectionally_supported_samples_can_strengthen_review_mean() -> None:
    left = _tracklet(
        "a",
        0.0,
        0.4,
        embedding=(1.0, 0.0),
        reid_embeddings=((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
    )
    right = _tracklet(
        "b",
        0.9,
        1.4,
        embedding=_embedding_with_distance(0.20),
        reid_embeddings=((1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
    )

    result = resolve_identities([left, right])

    assert len(result.groups) == 1
    assert result.accepted_edges[0].reid_robust_sample_distance == pytest.approx(0.0)
    assert result.accepted_edges[0].reid_strong_support_left == 2
    assert result.accepted_edges[0].reid_strong_support_right == 2


def test_proximity_without_identity_evidence_is_rejected() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, embedding=None, end_pitch=(0.0, 0.0)),
            _tracklet("b", 0.5, 0.9, embedding=None, start_pitch=(0.2, 0.0)),
        ]
    )

    assert len(result.groups) == 2
    assert _reason(result, "insufficient-identity-evidence")


def test_temporal_overlap_is_a_hard_constraint_even_with_identical_reid() -> None:
    result = resolve_identities(
        [_tracklet("a", 0.0, 1.0), _tracklet("b", 0.5, 1.5)]
    )

    assert len(result.groups) == 2
    assert _reason(result, "temporal-overlap")


def test_team_conflict_is_a_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, team_id="home", manual_team=True),
            _tracklet("b", 0.9, 1.4, team_id="away", manual_team=True),
        ]
    )

    assert len(result.groups) == 2
    assert _reason(result, "team-conflict")


def test_role_conflict_is_a_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, role="player", manual_role=True),
            _tracklet("b", 0.9, 1.4, role="referee", manual_role=True),
        ]
    )

    assert _reason(result, "role-conflict")


def test_automatic_team_flip_is_review_evidence_not_a_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, team_id="home"),
            _tracklet("b", 0.9, 1.4, team_id="away"),
        ]
    )

    assert result.accepted_edges == ()
    assert not _reason(result, "team-conflict")
    assert "automatic-team-disagreement" in result.review_edges[0].reasons


def test_automatic_role_flip_is_review_evidence_not_a_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, role="player"),
            _tracklet("b", 0.9, 1.4, role="goalkeeper"),
        ]
    )

    assert result.accepted_edges == ()
    assert not _reason(result, "role-conflict")
    assert "automatic-role-disagreement" in result.review_edges[0].reasons


def test_equal_external_player_id_is_sufficient_identity_evidence() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, embedding=None, external_player_id="roster-8"),
            _tracklet("b", 1.0, 1.4, embedding=None, external_player_id="roster-8"),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].id == "identity:external:roster-8"
    assert result.accepted_edges[0].reasons == ("external-player-match",)


def test_different_external_player_ids_are_a_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, external_player_id="roster-8"),
            _tracklet("b", 1.0, 1.4, external_player_id="roster-9"),
        ]
    )

    assert _reason(result, "external-player-conflict")


def test_unlabeled_bridge_cannot_transitively_join_different_external_players() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, external_player_id="roster-8"),
            _tracklet("bridge", 0.9, 1.3),
            _tracklet("c", 1.8, 2.2, external_player_id="roster-9"),
        ]
    )

    assert len(result.groups) == 2
    assert all(
        len(
            {
                external_id
                for tracklet_id, external_id in {
                    "a": "roster-8",
                    "bridge": None,
                    "c": "roster-9",
                }.items()
                if tracklet_id in group.tracklet_ids and external_id is not None
            }
        )
        <= 1
        for group in result.groups
    )
    demoted = next(
        edge
        for edge in result.review_edges
        if "transitive-external-player-conflict" in edge.reasons
    )
    assert (demoted.predecessor_id, demoted.successor_id) == ("bridge", "c")
    assert result.diagnostics["autoStitchCount"] == 1


def test_reliable_matching_jersey_stitches_without_reid() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                0.4,
                embedding=None,
                jersey_number="08",
                jersey_confidence=0.94,
                jersey_sample_count=3,
            ),
            _tracklet(
                "b",
                1.0,
                1.4,
                embedding=None,
                jersey_number=8,
                jersey_confidence=0.91,
                jersey_sample_count=2,
            ),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].jersey_number == "8"
    assert "reliable-jersey-match" in result.accepted_edges[0].reasons


def test_reliable_conflicting_jerseys_block_even_strong_reid() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, jersey_number=8, jersey_confidence=0.95, jersey_sample_count=3),
            _tracklet("b", 1.0, 1.4, jersey_number=9, jersey_confidence=0.95, jersey_sample_count=3),
        ]
    )

    assert _reason(result, "jersey-conflict")
    assert result.accepted_edges == ()


def test_unlabeled_bridge_cannot_transitively_join_reliable_jerseys() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                0.4,
                jersey_number=8,
                jersey_confidence=0.95,
                jersey_sample_count=3,
            ),
            _tracklet("bridge", 0.9, 1.3),
            _tracklet(
                "c",
                1.8,
                2.2,
                jersey_number=9,
                jersey_confidence=0.95,
                jersey_sample_count=3,
            ),
        ]
    )

    assert len(result.groups) == 2
    assert all(group.jersey_number in {"8", "9", None} for group in result.groups)
    assert not any(set(group.tracklet_ids) == {"a", "bridge", "c"} for group in result.groups)
    demoted = next(
        edge
        for edge in result.review_edges
        if "transitive-jersey-conflict" in edge.reasons
    )
    assert (demoted.predecessor_id, demoted.successor_id) == ("bridge", "c")
    assert result.diagnostics["autoStitchCount"] == 1


def test_low_confidence_jersey_disagreement_is_not_a_false_hard_constraint() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, jersey_number=8, jersey_confidence=0.45, jersey_sample_count=1),
            _tracklet("b", 1.0, 1.4, jersey_number=9, jersey_confidence=0.40, jersey_sample_count=1),
        ]
    )

    assert len(result.groups) == 1
    assert not _reason(result, "jersey-conflict")


def test_physically_impossible_transition_blocks_strong_reid() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, end_pitch=(0.0, 0.0)),
            _tracklet("b", 0.5, 0.9, start_pitch=(20.0, 0.0)),
        ]
    )

    assert _reason(result, "physically-impossible-transition")
    edge = result.rejected_edges[0]
    assert edge.pitch_distance_metres == 20.0
    assert edge.reachable_distance_metres == pytest.approx(3.2)


def test_position_uncertainty_relaxes_only_the_motion_gate() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                0.4,
                end_pitch=(0.0, 0.0),
                end_uncertainty_metres=10.0,
            ),
            _tracklet(
                "b",
                0.5,
                0.9,
                start_pitch=(20.0, 0.0),
                start_uncertainty_metres=10.0,
            ),
        ]
    )

    assert len(result.groups) == 1
    assert result.accepted_edges[0].reachable_distance_metres == pytest.approx(23.2)


def test_equal_successor_alternatives_remain_review_and_provisional() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4),
            _tracklet("b", 1.0, 1.4),
            _tracklet("c", 1.0, 1.4),
        ]
    )

    assert result.accepted_edges == ()
    assert len(result.groups) == 3
    assert sum("ambiguous-successor" in edge.reasons for edge in result.review_edges) == 2
    assert result.diagnostics["ambiguousEdgeCount"] == 2


def test_global_assignment_keeps_only_one_successor() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4),
            _tracklet("b", 1.0, 1.4),
            _tracklet(
                "c",
                1.0,
                1.4,
                embedding=_embedding_with_distance(0.16),
                reid_embeddings=((1.0, 0.0), (1.0, 0.0)),
            ),
        ]
    )

    assert [(edge.predecessor_id, edge.successor_id) for edge in result.accepted_edges] == [
        ("a", "b")
    ]
    not_selected = next(edge for edge in result.review_edges if edge.successor_id == "c")
    assert "not-selected-by-global-assignment" in not_selected.reasons


def test_global_assignment_builds_a_non_branching_chain() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4),
            _tracklet("b", 0.9, 1.3),
            _tracklet("c", 1.8, 2.2),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].tracklet_ids == ("a", "b", "c")
    assert {(edge.predecessor_id, edge.successor_id) for edge in result.accepted_edges} == {
        ("a", "b"),
        ("b", "c"),
    }


def test_manual_identity_is_a_must_link_even_for_overlapping_conflicting_fragments() -> None:
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                1.0,
                team_id="home",
                manual_identity_id="person-8",
                manual_confirmed=True,
                manual_team=True,
            ),
            _tracklet(
                "b",
                0.5,
                1.5,
                team_id="away",
                manual_identity_id="person-8",
                manual_confirmed=True,
                manual_team=True,
            ),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].id == "identity:manual:person-8"
    assert result.groups[0].source == "manual"
    assert result.accepted_edges[0].source == "manual"
    assert "manual-component-team-conflict" in result.accepted_edges[0].reasons
    assert result.diagnostics["manualConflictCount"] == 1


def test_manual_owner_without_positive_confirmation_preserves_group_but_not_resolution():
    result = resolve_identities(
        [
            _tracklet(
                "a",
                0.0,
                0.4,
                embedding=None,
                manual_identity_id="person-8",
                manual_confirmed=False,
            ),
            _tracklet(
                "b",
                1.0,
                1.4,
                embedding=None,
                manual_identity_id="person-8",
                manual_confirmed=False,
            ),
        ]
    )

    assert len(result.groups) == 1
    assert result.groups[0].id == "identity:manual:person-8"
    assert result.groups[0].status == "provisional"
    assert result.groups[0].source == "local-tracklet"
    assert result.groups[0].confidence == 0.0
    assert result.accepted_edges[0].source == "manual"


def test_different_manual_identity_ids_are_a_hard_cannot_link() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, manual_identity_id="person-a"),
            _tracklet("b", 1.0, 1.4, manual_identity_id="person-b"),
        ]
    )

    assert len(result.groups) == 2
    assert _reason(result, "manual-identity-conflict")


def test_excluded_tracklet_is_preserved_but_never_considered_for_auto_stitch() -> None:
    result = resolve_identities(
        [
            _tracklet("excluded", 0.0, 0.4, manual_excluded=True),
            _tracklet("visible", 1.0, 1.4),
        ]
    )

    assert set(result.tracklet_to_identity) == {"excluded", "visible"}
    excluded = next(group for group in result.groups if "excluded" in group.tracklet_ids)
    assert excluded.status == "excluded"
    assert result.accepted_edges == ()
    assert result.rejected_edges == ()


def test_resolution_is_deterministic_for_reversed_input_order() -> None:
    tracklets = [
        _tracklet("z", 1.0, 1.4),
        _tracklet("a", 0.0, 0.4),
    ]

    forward = resolve_identities(tracklets)
    reverse = resolve_global_identities(list(reversed(tracklets)))

    assert forward == reverse
    assert forward.groups[0].id == "identity:a"


def test_diagnostics_account_for_every_tracklet_and_observation() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, observation_count=7),
            _tracklet("b", 1.0, 1.4, observation_count=5),
        ]
    )

    assert result.diagnostics["trackletCount"] == 2
    assert result.diagnostics["observationCount"] == 12
    assert result.diagnostics["preservedObservationCount"] == 12
    assert result.diagnostics["identityObservationCoverage"] == 1.0
    assert result.diagnostics["allTrackletsPreserved"] is True
    assert sorted(
        tracklet_id for group in result.groups for tracklet_id in group.tracklet_ids
    ) == ["a", "b"]


def test_empty_input_has_a_complete_zero_result() -> None:
    result = resolve_identities([])

    assert result.groups == ()
    assert result.tracklet_to_identity == {}
    assert result.diagnostics["allTrackletsPreserved"] is True


def test_duplicate_tracklet_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        resolve_identities([_tracklet("a", 0.0, 0.4), _tracklet("a", 1.0, 1.4)])


def test_invalid_tracklet_contract_fails_early() -> None:
    with pytest.raises(ValueError, match="end_time"):
        IdentityTracklet("bad", 2.0, 1.0)
    with pytest.raises(ValueError, match="same dimension"):
        IdentityTracklet(
            "bad-reid",
            0.0,
            1.0,
            mean_reid_embedding=(1.0, 0.0),
            reid_embeddings=((1.0, 0.0, 0.0),),
        )


def test_custom_speed_config_is_honoured() -> None:
    result = resolve_identities(
        [
            _tracklet("a", 0.0, 0.4, end_pitch=(0.0, 0.0)),
            _tracklet("b", 1.4, 1.8, start_pitch=(9.0, 0.0)),
        ],
        IdentityResolverConfig(
            max_player_speed_metres_per_second=5.0,
            motion_slack_metres=1.0,
        ),
    )

    assert _reason(result, "physically-impossible-transition")
