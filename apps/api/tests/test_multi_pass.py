import app.multi_pass_alignment as multi_pass_alignment

from app.multi_pass_alignment import (
    classify_pass_relation,
    manual_clock_alignment,
    motion_dtw,
    temporal_alignment,
)
from app.multi_pass_fusion import (
    copy_reference_identity_state,
    fuse_aligned_pass_identities,
)
from app.multi_pass_metrics import consensus_summary, pass_quality
from app.project_http_contracts import ProjectCompositionRequest


def _scene(*, tracks=10, ball=12, frames=30, calibration=0.8, verdict="pass"):
    reconstruction = {
        "status": "ready",
        "frameCount": frames,
        "pitchCalibration": {"status": "ready", "confidence": calibration},
    }
    if verdict is not None:
        reconstruction.update(
            {
                "qualityVerdict": verdict,
                "quality": {
                    "verdict": verdict,
                    "gates": [
                        {"id": "coverage", "status": verdict, "required": True},
                        {"id": "physics", "status": verdict, "required": True},
                    ],
                },
                "calibration": {
                    "summary": {
                        "usableCoverage": 0.96 if verdict == "pass" else 0.05,
                    }
                },
            }
        )
    return {
        "payload": {
            "tracks": [{} for _ in range(tracks)],
            "ball": {"keyframes": [{} for _ in range(ball)]},
            "videoAsset": {
                "reconstruction": reconstruction
            },
        }
    }


def test_pass_quality_rewards_calibration_tracks_and_ball():
    strong = pass_quality(_scene())
    weak = pass_quality(_scene(tracks=2, ball=0, frames=8, calibration=0.0))

    assert strong > 0.7
    assert strong > weak


def test_pass_quality_does_not_reward_false_positive_counts_after_quality_reject():
    rejected_with_many_detections = pass_quality(
        _scene(tracks=22, ball=30, frames=60, calibration=0.99, verdict="reject")
    )
    accepted_with_less_evidence = pass_quality(
        _scene(tracks=8, ball=4, frames=24, calibration=0.8, verdict="pass")
    )

    assert rejected_with_many_detections <= 0.24
    assert accepted_with_less_evidence > rejected_with_many_detections


def test_pass_quality_requires_the_current_qa_contract():
    unverified = _scene(
        tracks=22,
        ball=30,
        frames=60,
        calibration=0.99,
        verdict=None,
    )

    assert pass_quality(unverified) == 0.0


def test_consensus_counts_independent_pass_evidence():
    summary = consensus_summary(
        [
            {"status": "ready", "qualityVerdict": "pass", "calibrationStatus": "ready", "ballSamples": 9, "trackCount": 12},
            {"status": "ready", "qualityVerdict": "review", "calibrationStatus": "fallback", "ballSamples": 0, "trackCount": 8},
            {"status": "failed", "calibrationStatus": "fallback", "ballSamples": 0, "trackCount": 0},
        ]
    )

    assert summary["passesAnalyzed"] == 2
    assert summary["metricPasses"] == 1
    assert summary["ballPasses"] == 1
    assert 0 < summary["evidenceScore"] < 1


def test_consensus_only_calls_quality_passes_metric():
    summary = consensus_summary(
        [
            {
                "status": "ready",
                "qualityVerdict": "reject",
                "calibrationStatus": "ready",
                "ballSamples": 20,
                "trackCount": 20,
            },
            {
                "status": "ready",
                "qualityVerdict": "pass",
                "calibrationStatus": "ready",
                "ballSamples": 4,
                "trackCount": 8,
            },
        ]
    )

    assert summary["metricPasses"] == 1


def test_motion_dtw_prefers_time_warped_replay_over_different_action():
    reference = [0.0, 0.2, 0.7, 1.0, 0.55, 0.1, 0.0]
    replay = [0.0, 0.1, 0.4, 0.8, 1.0, 0.7, 0.3, 0.05, 0.0]
    different = [1.0, 0.8, 0.3, 0.0, 0.0, 0.6, 1.0]

    replay_alignment = motion_dtw(reference, replay)
    different_alignment = motion_dtw(reference, different)

    assert replay_alignment["cost"] < different_alignment["cost"]
    assert replay_alignment["anchors"][0] == {"reference": 0.0, "pass": 0.0}


def test_relation_distinguishes_overlap_from_adjacent_continuation():
    reference = {"start": 4.0, "end": 10.5}

    assert classify_pass_relation(0.04, {"start": 10.5, "end": 17.0}, reference) == "replay-overlap"
    assert classify_pass_relation(0.12, {"start": 0.0, "end": 4.0}, reference) == "continuation-before"


def test_reference_identity_copy_preserves_canonical_people_and_evidence():
    reference = {
        "tracks": [
            {"id": "home-1", "canonicalPersonId": "canonical-home-1"},
            {"id": "away-1", "canonicalPersonId": "canonical-away-1"},
        ],
        "canonicalPeople": [
            {
                "canonicalPersonId": "canonical-home-1",
                "observations": [{"observationId": "obs-home-1"}],
                "evidence": [{"kind": "reid", "confidence": 0.92}],
            },
            {
                "canonicalPersonId": "canonical-away-1",
                "observations": [{"observationId": "obs-away-1"}],
                "evidence": [{"kind": "jersey-ocr", "value": "8"}],
            },
        ],
    }
    target = {}

    warnings = copy_reference_identity_state(target, reference)

    assert warnings == []
    assert target["canonicalPeople"] == reference["canonicalPeople"]
    assert target["canonicalPeople"] is not reference["canonicalPeople"]
    canonical_ids = {
        person["canonicalPersonId"] for person in target["canonicalPeople"]
    }
    assert {
        track["canonicalPersonId"] for track in target["tracks"]
    } <= canonical_ids
    assert target["canonicalPeople"][1]["evidence"][0]["value"] == "8"


def test_reference_identity_copy_detaches_orphan_track_reference():
    target = {}

    warnings = copy_reference_identity_state(
        target,
        {
            "tracks": [{"id": "legacy", "canonicalPersonId": "missing-person"}],
            "canonicalPeople": [],
        },
    )

    assert target["tracks"] == [{"id": "legacy"}]
    assert target["canonicalPeople"] == []
    assert "missing-person" in warnings[0]


def _identity_person(identifier: str, **values) -> dict:
    return {
        "canonicalPersonId": identifier,
        "teamId": "home",
        "role": "player",
        "jerseyNumber": None,
        "externalPlayerId": None,
        "observations": [],
        "evidence": [],
        **values,
    }


def _aligned_pass(scene_id: str, people: list[dict], *, confidence: float = 0.9) -> tuple[dict, dict]:
    alignment = {
        "relation": "replay-overlap",
        "method": "motion-dtw",
        "confidence": confidence,
        "motionCost": 0.03,
        "overlap": True,
        "anchors": [
            {"referenceTime": 0.0, "passTime": 0.0},
            {"referenceTime": 5.0, "passTime": 5.0},
        ],
    }
    return (
        {"id": scene_id, "payload": {"canonicalPeople": people}},
        {
            "sceneId": scene_id,
            "segmentId": f"segment-{scene_id}",
            "relation": "replay-overlap",
            "alignment": alignment,
        },
    )


def test_multi_pass_identity_fusion_enriches_reference_without_mixing_observations():
    target = {
        "canonicalPeople": [
            _identity_person(
                "reference-8",
                jerseyNumber="8",
                externalPlayerId="roster-8",
                observations=[{"observationId": "reference-observation"}],
            )
        ]
    }
    source = _identity_person(
        "source-8",
        jerseyNumber="8",
        externalPlayerId="roster-8",
        observations=[{"observationId": "foreign-observation", "frameIndex": 4}],
    )

    diagnostics = fuse_aligned_pass_identities(
        target,
        {"id": "reference-angle"},
        [_aligned_pass("reference-angle", []), _aligned_pass("source-angle", [source])],
    )

    person = target["canonicalPeople"][0]
    assert diagnostics["matchedIdentityCount"] == 1
    assert diagnostics["referenceSceneId"] == "reference-angle"
    assert person["observations"] == [{"observationId": "reference-observation"}]
    assert person["evidence"][-1]["signals"] == [
        "external-player-match",
        "reliable-jersey-match",
    ]
    cross_angle = person["multiAngleEvidence"][0]
    assert cross_angle["sourceSceneId"] == "source-angle"
    assert cross_angle["sourceCanonicalPersonId"] == "source-8"
    assert cross_angle["observations"][0]["sourceSceneId"] == "source-angle"
    assert cross_angle["observations"][0]["observationId"] != "foreign-observation"


def test_multi_pass_identity_fusion_abstains_for_weak_alignment():
    target = {"canonicalPeople": [_identity_person("reference-8", jerseyNumber="8")]}

    diagnostics = fuse_aligned_pass_identities(
        target,
        {"id": "reference-angle"},
        [
            _aligned_pass(
                "source-angle",
                [_identity_person("source-8", jerseyNumber="8")],
                confidence=0.2,
            )
        ],
    )

    assert "multiAngleEvidence" not in target["canonicalPeople"][0]
    assert diagnostics["matchedIdentityCount"] == 0
    assert diagnostics["skippedPasses"][0]["reason"] == "alignment-not-usable"


def test_multi_pass_identity_fusion_abstains_for_ambiguous_or_conflicting_people():
    target = {
        "canonicalPeople": [
            _identity_person("reference-a", jerseyNumber="8"),
            _identity_person("reference-b", jerseyNumber="8"),
        ]
    }
    sources = [
        _identity_person("ambiguous", jerseyNumber="8"),
        _identity_person("wrong-team", teamId="away", jerseyNumber="8"),
        _identity_person("wrong-number", jerseyNumber="9"),
    ]

    diagnostics = fuse_aligned_pass_identities(
        target,
        {"id": "reference-angle"},
        [_aligned_pass("source-angle", sources)],
    )

    assert all("multiAngleEvidence" not in person for person in target["canonicalPeople"])
    assert diagnostics["matchedIdentityCount"] == 0
    assert len(diagnostics["reviewCandidates"]) == 3


def test_manual_alignment_is_authoritative_over_motion_dtw(monkeypatch):
    reference = {"id": "reference-angle", "duration": 8.0}
    source = {"id": "source-angle", "duration": 6.0}
    reference_segment = {"id": "reference-segment", "start": 10.0, "end": 18.0}
    source_segment = {"id": "source-segment", "start": 30.0, "end": 36.0}
    saved = [
        {
            "sourceSceneId": "source-angle",
            "segmentId": "source-segment",
            "anchors": [
                {"referenceTime": 1.0, "passTime": 0.4},
                {"referenceTime": 7.0, "passTime": 5.5},
            ],
        }
    ]

    monkeypatch.setattr(
        multi_pass_alignment,
        "motion_signature",
        lambda _scene: (_ for _ in ()).throw(AssertionError("DTW must not run")),
    )
    alignment = temporal_alignment(
        reference,
        source,
        reference_segment,
        source_segment,
        saved,
    )

    assert alignment["method"] == "manual-clock-anchors"
    assert alignment["relation"] == "replay-overlap"
    assert alignment["confidence"] == 1.0
    assert alignment["anchors"] == saved[0]["anchors"]


def test_manual_alignment_rejects_non_monotonic_and_out_of_range_anchors():
    reference = {"id": "reference-angle", "duration": 8.0}
    source = {"id": "source-angle", "duration": 6.0}
    segment = {"id": "source-segment"}

    non_monotonic, non_monotonic_diagnostics = manual_clock_alignment(
        [
            {"sourceSceneId": "source-angle", "referenceTime": 1.0, "passTime": 4.0},
            {"sourceSceneId": "source-angle", "referenceTime": 7.0, "passTime": 2.0},
        ],
        reference,
        source,
        segment,
    )
    out_of_range, range_diagnostics = manual_clock_alignment(
        [
            {"segmentId": "source-segment", "referenceTime": 1.0, "passTime": 0.5},
            {"segmentId": "source-segment", "referenceTime": 9.0, "passTime": 5.0},
        ],
        reference,
        source,
        segment,
    )

    assert non_monotonic is None
    assert "anchors-not-strictly-monotonic" in non_monotonic_diagnostics["rejectionReasons"]
    assert out_of_range is None
    assert "reference-time-out-of-range" in range_diagnostics["rejectionReasons"]
    assert "at-least-two-valid-anchors-required" in range_diagnostics["rejectionReasons"]


def test_project_composition_request_accepts_http_and_python_field_names():
    anchors = [
        {
            "segmentId": "segment-b",
            "anchors": [
                {"referenceTime": 0.0, "passTime": 0.1},
                {"referenceTime": 4.0, "passTime": 3.8},
            ],
        }
    ]

    camel_case = ProjectCompositionRequest.model_validate(
        {
            "segment_ids": ["segment-a", "segment-b"],
            "manualAlignmentAnchors": anchors,
        }
    )
    snake_case = ProjectCompositionRequest.model_validate(
        {
            "segment_ids": ["segment-a", "segment-b"],
            "manual_alignment_anchors": anchors,
        }
    )

    assert camel_case.manual_alignment_anchors == anchors
    assert snake_case.manual_alignment_anchors == anchors
