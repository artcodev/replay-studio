from app.multi_angle_identity import fuse_aligned_identity_passes


def _person(identifier: str, **values) -> dict:
    return {
        "id": identifier,
        "canonicalPersonId": identifier,
        "teamId": "home",
        "role": "player",
        "jerseyNumber": None,
        "externalPlayerId": None,
        "observations": [],
        "evidence": [],
        **values,
    }


def _pass(people: list[dict], **alignment) -> dict:
    return {
        "sceneId": "angle-b",
        "canonicalPeople": people,
        "alignment": {
            "overlap": True,
            "confidence": 0.9,
            "method": "motion-dtw",
            "anchors": [{"referenceTime": 0, "passTime": 0}, {"referenceTime": 1, "passTime": 1}],
            **alignment,
        },
    }


def test_aligned_reliable_jersey_adds_namespaced_cross_angle_evidence() -> None:
    reference = [_person("reference-8", jerseyNumber="8")]
    source = _person(
        "source-8",
        jerseyNumber="8",
        sourceTrackletIds=["tracklet-1"],
        observations=[{"observationId": "frame-1:person", "frameIndex": 1}],
    )

    people, diagnostics = fuse_aligned_identity_passes(reference, [_pass([source])])

    assert diagnostics["matchedIdentityCount"] == 1
    assert people[0]["canonicalPersonId"] == "reference-8"
    cross_view = people[0]["multiAngleEvidence"][0]
    assert cross_view["signals"] == ["reliable-jersey-match"]
    assert cross_view["observations"][0]["observationId"].startswith("angle-")
    assert cross_view["observations"][0]["sourceSceneId"] == "angle-b"
    assert cross_view["sourceTrackletIds"][0] != "tracklet-1"


def test_unaligned_similar_kit_abstains() -> None:
    people, diagnostics = fuse_aligned_identity_passes(
        [_person("reference")],
        [_pass([_person("source")], confidence=0.2)],
    )

    assert "multiAngleEvidence" not in people[0]
    assert diagnostics["usableAlignedPassCount"] == 0
    assert diagnostics["ambiguousOrUnmatchedCount"] == 1


def test_conflicting_jersey_team_or_role_blocks_cross_angle_merge() -> None:
    reference = [_person("reference", jerseyNumber="8")]
    sources = [
        _person("wrong-number", jerseyNumber="9"),
        _person("wrong-team", jerseyNumber="8", teamId="away"),
        _person("wrong-role", jerseyNumber="8", role="referee"),
    ]

    people, diagnostics = fuse_aligned_identity_passes(reference, [_pass(sources)])

    assert "multiAngleEvidence" not in people[0]
    assert diagnostics["matchedIdentityCount"] == 0
    assert len(diagnostics["reviewCandidates"]) == 3


def test_bound_ocr_conflict_cannot_transfer_identity_by_jersey_to_unbound_angle() -> None:
    reference = [_person("reference-9", jerseyNumber="9")]
    source = _person(
        "source-bound-8",
        jerseyNumber="9",
        externalPlayerId="player-8",
        conflicts=[
            {
                "code": "manual-roster-jersey-conflict",
                "expectedNumber": "8",
                "observedNumber": "9",
            }
        ],
    )

    people, diagnostics = fuse_aligned_identity_passes(
        reference,
        [_pass([source])],
    )

    assert "multiAngleEvidence" not in people[0]
    assert diagnostics["matchedIdentityCount"] == 0
    assert diagnostics["ambiguousOrUnmatchedCount"] == 1


def test_bound_identity_requires_same_external_id_on_both_angles() -> None:
    people, diagnostics = fuse_aligned_identity_passes(
        [_person("reference-8", jerseyNumber="8")],
        [_pass([_person("source-8", jerseyNumber="8", externalPlayerId="player-8")])],
    )

    assert "multiAngleEvidence" not in people[0]
    assert diagnostics["matchedIdentityCount"] == 0


def test_duplicate_reference_number_is_ambiguous_and_fails_closed() -> None:
    reference = [
        _person("reference-a", jerseyNumber="8"),
        _person("reference-b", jerseyNumber="8"),
    ]

    people, diagnostics = fuse_aligned_identity_passes(
        reference,
        [_pass([_person("source", jerseyNumber="8")])],
    )

    assert all("multiAngleEvidence" not in person for person in people)
    assert diagnostics["matchedIdentityCount"] == 0
    assert diagnostics["ambiguousOrUnmatchedCount"] == 1


def test_manual_clock_alignment_can_enable_fusion() -> None:
    people, diagnostics = fuse_aligned_identity_passes(
        [_person("reference", externalPlayerId="roster-4")],
        [
            _pass(
                [_person("source", externalPlayerId="roster-4")],
                confidence=0.55,
                method="manual-clock-anchors",
            )
        ],
    )

    assert diagnostics["matchedIdentityCount"] == 1
    assert people[0]["multiAngleEvidence"][0]["signals"] == ["external-player-match"]
