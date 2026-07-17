from copy import deepcopy

import pytest

from app.quality_metrics import evaluate_reconstruction_quality


def _keyframe(t, x, z, *, source="direct", confidence=0.9, clamped=False):
    return {
        "t": t,
        "x": x,
        "z": z,
        "confidence": confidence,
        "projection": {"source": source, "clamped": clamped},
    }


def _evidence(
    index,
    *,
    status="accepted",
    source="direct",
    residual=2.0,
    residual_p95=None,
    inliers=8,
    visible_side="left",
    alignment_f1=0.4,
):
    return {
        "sourceFrameIndex": index,
        "sampleIndex": index,
        "sceneTime": round(index * 0.2, 3),
        "status": status,
        "source": "pnlcalib" if status == "accepted" else "none",
        "projectionSource": source,
        "keypointCount": 10,
        "inlierCount": inliers,
        "inlierRatio": inliers / 10,
        "reprojectionError": residual if status == "accepted" else None,
        "reprojectionP95": (
            residual_p95 if residual_p95 is not None else residual
        )
        if status == "accepted"
        else None,
        "visiblePitchSide": visible_side,
        "alignmentMetrics": {"f1": alignment_f1},
        "rejectionReasons": [] if status == "accepted" else ["no-valid-homography"],
    }


def _scene():
    evidence = [_evidence(index, residual=2.0 + (index % 3) * 0.2) for index in range(10)]
    player = [_keyframe(index * 0.2, index * 0.6, 3.0) for index in range(10)]
    ball = [_keyframe(index * 0.2, index * 2.0, 1.0) for index in range(10)]
    return {
        "id": "shot-qa",
        "title": "QA fixture",
        "version": 1,
        "duration": 2.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "tracks": [
                {
                    "id": "home-1",
                    "teamId": "home",
                    "keyframes": player,
                }
            ],
            "ball": {"keyframes": ball},
            "videoAsset": {
                "reconstruction": {
                    "status": "ready",
                    "frameCount": 10,
                    "coordinateSpace": "pitch-metric-per-frame-homography",
                    "pitchCalibration": {"status": "ready", "method": "pnlcalib-points-lines"},
                    "calibration": {"frameEvidence": evidence},
                    "calibrationFrames": evidence,
                    "diagnostics": {
                        "calibratedFrameCount": 10,
                        "calibrationFrameCoverage": 1.0,
                    },
                }
            },
        },
    }


def _gates(report):
    return {gate["id"]: gate for gate in report["gates"]}


def test_complete_evidence_and_physical_trajectories_pass():
    report = evaluate_reconstruction_quality(_scene())

    assert report["verdict"] == "pass"
    assert report["processingStatus"] == "ready"
    assert report["metrics"]["calibrationCoverage"]["value"] == 1.0
    assert report["metrics"]["directCalibrationCoverage"]["value"] == 1.0
    assert report["metrics"]["projectionFallbackRatio"]["value"] == 0.0
    assert report["metrics"]["trackContinuity"]["value"] == 1.0
    assert report["metrics"]["playerSpeedViolationRatio"]["value"] == 0.0
    assert all(gate["status"] == "pass" for gate in report["gates"] if gate["required"])
    assert report["identityValidation"]["groundTruthAvailable"] is False
    assert report["metrics"]["identityIdf1"]["value"] is None
    assert _gates(report)["identity-idf1"]["required"] is False


def test_labelled_identity_swap_rejects_even_when_motion_is_physical():
    scene = _scene()
    scene["payload"]["validationGroundTruth"] = {
        "identityAssignments": [
            {
                "frameIndex": frame_index,
                "groundTruthId": ground_truth,
                "predictedId": predicted,
            }
            for frame_index, mapping in enumerate(
                [
                    {"a": "left", "b": "right"},
                    {"a": "left", "b": "right"},
                    {"a": "right", "b": "left"},
                    {"a": "right", "b": "left"},
                ]
            )
            for ground_truth, predicted in mapping.items()
        ]
    }

    report = evaluate_reconstruction_quality(scene)

    assert report["metrics"]["playerSpeedViolationRatio"]["value"] == 0.0
    assert report["identityValidation"]["idf1"] == 0.5
    assert _gates(report)["identity-idf1"]["status"] == "reject"
    assert report["verdict"] == "reject"


def test_labelled_identity_frame_rate_is_required_for_overlap_seconds():
    scene = _scene()
    scene["payload"]["validationGroundTruth"] = {
        "identityAssignmentFrameRate": 10.0,
        "identityAssignments": [
            {"frameIndex": 10, "groundTruthId": "a", "predictedId": "same"},
            {"frameIndex": 10, "groundTruthId": "b", "predictedId": "same"},
            {"frameIndex": 11, "groundTruthId": "a", "predictedId": "a"},
            {"frameIndex": 11, "groundTruthId": "b", "predictedId": "b"},
        ],
    }

    report = evaluate_reconstruction_quality(scene)

    assert report["identityValidation"]["duplicateAssignmentFrameCount"] == 1
    assert report["identityValidation"]["duplicateOverlapSeconds"] == 0.1
    assert (
        report["identityValidation"]["duplicateOverlapTimebase"]
        == "frame-index+explicit-fps"
    )


def test_ball_temporal_resolver_metrics_and_pass_gates_are_exposed():
    scene = _scene()
    scene["payload"]["ball"]["diagnostics"] = {
        "frameCount": 20,
        "observedFrameCount": 14,
        "inferredFrameCount": 4,
        "occludedFrameCount": 2,
        "observedCoverage": 0.70,
        "publishedCoverage": 0.90,
        "gaps": {"gapCount": 2, "longestGapSeconds": 0.48},
        "pathCostMargin": 3.25,
    }

    report = evaluate_reconstruction_quality(scene)
    metrics = report["metrics"]
    gates = _gates(report)

    assert metrics["ballObservedCoverage"] == {
        "value": 0.7,
        "unit": "ratio",
        "source": "ball-temporal-resolver",
        "sampleCount": 20,
        "observedFrames": 14,
        "occludedFrames": 2,
    }
    assert metrics["ballPublishedCoverage"] == {
        "value": 0.9,
        "unit": "ratio",
        "source": "ball-temporal-resolver",
        "sampleCount": 20,
        "inferredFrames": 4,
    }
    assert metrics["ballLongestUnresolvedGap"]["value"] == 0.48
    assert metrics["ballLongestUnresolvedGap"]["sampleCount"] == 2
    assert metrics["ballPathCostMargin"]["value"] == 3.25
    assert metrics["ballPathCostMargin"]["sampleCount"] == 1
    assert gates["ball-observed-coverage"]["status"] == "pass"
    assert gates["ball-published-coverage"]["status"] == "pass"


def test_low_ball_coverage_is_diagnostic_and_does_not_override_required_verdict():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["ballDetection"] = {
        "tracking": {
            "frameCount": 20,
            "observedFrameCount": 8,
            "inferredFrameCount": 2,
            "occludedFrameCount": 10,
            "observedCoverage": 0.40,
            "publishedCoverage": 0.50,
            "gaps": {"gapCount": 3, "longestGapSeconds": 1.2},
            "pathCostMargin": None,
        }
    }

    report = evaluate_reconstruction_quality(scene)
    gates = _gates(report)

    assert report["metrics"]["ballObservedCoverage"]["source"] == (
        "ball-temporal-resolver"
    )
    assert gates["ball-observed-coverage"]["status"] == "review"
    assert gates["ball-published-coverage"]["status"] == "reject"
    assert gates["ball-observed-coverage"]["required"] is False
    assert gates["ball-published-coverage"]["required"] is False
    assert report["verdict"] == "pass"


def test_low_coverage_fallback_clamping_and_impossible_speed_reject():
    scene = _scene()
    evidence = [
        _evidence(
            index,
            status="accepted" if index in {0, 9} else "rejected",
            source="direct" if index in {0, 9} else "none",
        )
        for index in range(10)
    ]
    scene["payload"]["videoAsset"]["reconstruction"]["calibration"]["frameEvidence"] = evidence
    scene["payload"]["tracks"][0]["keyframes"] = [
        _keyframe(0.0, -52.5, -34.0, source="screen-approximate", clamped=True),
        _keyframe(0.2, 52.5, 34.0, source="screen-approximate", clamped=True),
    ]

    report = evaluate_reconstruction_quality(scene)
    gates = _gates(report)

    assert report["verdict"] == "reject"
    assert report["metrics"]["calibrationCoverage"]["value"] == 0.2
    assert report["metrics"]["projectionFallbackRatio"]["value"] > 0
    assert report["metrics"]["boundaryClampRatio"]["value"] > 0
    assert report["metrics"]["playerSpeedViolationRatio"]["value"] == 1.0
    assert gates["calibration-coverage"]["status"] == "reject"
    assert gates["calibration-gap"]["status"] == "reject"
    assert gates["player-speed"]["status"] == "reject"


def test_legacy_metric_run_without_provenance_cannot_pass():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction.pop("calibration")
    reconstruction.pop("calibrationFrames")
    reconstruction["diagnostics"] = {
        "calibratedFrameCount": 10,
        "calibrationFrameCoverage": 1.0,
        "calibrationReprojectionError": 2.5,
    }
    for track in scene["payload"]["tracks"]:
        for keyframe in track["keyframes"]:
            keyframe.pop("projection")
    for keyframe in scene["payload"]["ball"]["keyframes"]:
        keyframe.pop("projection")

    report = evaluate_reconstruction_quality(scene)
    gates = _gates(report)

    assert report["verdict"] == "review"
    assert gates["calibration-coverage"]["status"] == "pass"
    assert gates["calibration-gap"]["status"] == "unknown"
    assert gates["projection-fallback"]["status"] == "unknown"
    assert set(report["summary"]["unknownRequiredGates"]) >= {
        "calibration-gap",
        "projection-fallback",
    }


def test_bad_manual_anchor_alignment_rejects_even_without_frame_evidence():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction.pop("calibration")
    reconstruction.pop("calibrationFrames")
    reconstruction["pitchCalibration"] = {
        "status": "ready",
        "method": "manual-pitch-anchors",
        "alignmentError": 33.75,
    }
    reconstruction["diagnostics"] = {"calibrationFrameCoverage": 0.0}

    report = evaluate_reconstruction_quality(scene)
    gates = _gates(report)

    assert report["verdict"] == "reject"
    assert gates["calibration-coverage"]["status"] == "unknown"
    assert gates["reprojection-p50"]["status"] == "reject"
    assert gates["reprojection-p95"]["status"] == "reject"


def test_zero_confidence_interpolation_endpoints_do_not_create_speed_violations():
    scene = _scene()
    scene["payload"]["tracks"][0]["keyframes"] = [
        _keyframe(0.0, -50.0, 0.0, confidence=0.0),
        _keyframe(0.8, 1.0, 1.0),
        _keyframe(1.0, 1.5, 1.0),
        _keyframe(2.0, 50.0, 0.0, confidence=0.0),
    ]

    report = evaluate_reconstruction_quality(scene)

    assert report["metrics"]["playerSpeedViolationRatio"]["sampleCount"] == 1
    assert report["metrics"]["playerSpeedViolationRatio"]["value"] == 0.0


def test_inferred_presence_does_not_create_speed_violations_or_qa_evidence():
    scene = _scene()
    scene["payload"]["tracks"][0]["keyframes"] = [
        {**_keyframe(0.0, -50.0, 0.0, confidence=0.18), "observed": False},
        {**_keyframe(0.8, 1.0, 1.0), "observed": True},
        {**_keyframe(1.0, 1.5, 1.0), "observed": True},
        {**_keyframe(2.0, 50.0, 0.0, confidence=0.18), "observed": False},
    ]

    report = evaluate_reconstruction_quality(scene)

    assert report["metrics"]["playerSpeedViolationRatio"]["sampleCount"] == 1
    assert report["metrics"]["playerSpeedViolationRatio"]["value"] == 0.0


def test_discarded_impossible_jump_remains_visible_to_quality_gates():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["diagnostics"].update(
        {
            "preFilterSpeedSampleCount": 20,
            "preFilterSpeedViolationCount": 4,
            "preFilterMaximumSpeedMetresPerSecond": 86.0,
        }
    )

    report = evaluate_reconstruction_quality(scene)
    metric = report["metrics"]["playerSpeedViolationRatio"]

    assert metric["source"] == "trajectory-pre-filter"
    assert metric["value"] == 0.2
    assert metric["maxMetresPerSecond"] == 86.0
    assert _gates(report)["player-speed"]["status"] == "reject"


def test_calibration_gap_uses_sample_cadence_for_consecutive_missing_frames():
    scene = _scene()
    evidence = [
        _evidence(
            index,
            status="rejected" if index in {2, 3, 4} else "accepted",
            source="none" if index in {2, 3, 4} else "direct",
        )
        for index in range(10)
    ]

    report = evaluate_reconstruction_quality(scene, evidence)

    assert report["metrics"]["maxCalibrationGap"]["value"] == pytest.approx(0.6)
    assert _gates(report)["calibration-gap"]["status"] == "pass"


def test_reprojection_p95_uses_per_frame_tail_not_frame_median():
    scene = _scene()
    evidence = [
        _evidence(index, residual=2.0, residual_p95=18.0 if index == 9 else 3.0)
        for index in range(10)
    ]

    report = evaluate_reconstruction_quality(scene, evidence)

    assert report["metrics"]["calibrationResidualP50"]["value"] == 2.0
    assert report["metrics"]["calibrationResidualP95"]["value"] > 10.0
    assert _gates(report)["reprojection-p95"]["status"] == "review"


def test_visible_side_flips_are_an_explicit_reject_gate():
    scene = _scene()
    evidence = [
        _evidence(index, visible_side="left" if index % 2 else "right")
        for index in range(10)
    ]

    report = evaluate_reconstruction_quality(scene, evidence)

    metric = report["metrics"]["visiblePitchSideAgreement"]
    assert metric["value"] == 0.5
    assert metric["sideVotes"] == {"left": 5, "right": 5}
    assert _gates(report)["orientation-stability"]["status"] == "reject"
    assert report["verdict"] == "reject"


def test_high_temporal_recovery_uncertainty_is_a_required_reject_gate():
    scene = _scene()
    evidence = [_evidence(0)]
    evidence.extend(
        {
            **_evidence(index, source="temporal-forward"),
            "solutionStatus": "temporal-accepted",
            "uncertainty": {"p95Metres": 6.2},
            "temporal": {"anchorFrameIndices": [0]},
        }
        for index in range(1, 10)
    )

    report = evaluate_reconstruction_quality(scene, evidence)
    gate = _gates(report)["temporal-uncertainty"]

    assert report["metrics"]["temporalCalibrationCoverage"]["value"] == 0.9
    assert report["metrics"]["temporalCalibrationUncertaintyP95"]["value"] == 6.2
    assert gate["required"] is True
    assert gate["status"] == "reject"
    assert report["verdict"] == "reject"


def test_report_does_not_mutate_scene():
    scene = _scene()
    before = deepcopy(scene)

    evaluate_reconstruction_quality(scene)

    assert scene == before
