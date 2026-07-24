from __future__ import annotations

from app.artifact_store import FilesystemArtifactStore
from app.reconstruction_calibration_stage import (
    _frame_gate,
    calibration_stage_status,
    publish_calibration_stage,
)
from app.reconstruction_detection_contract import CalibrationPhaseResult


def _calibration_result(frame_evidence: list[dict]) -> CalibrationPhaseResult:
    return CalibrationPhaseResult(
        calibration=None,
        quality={
            "verdict": "review",
            "summary": {"visiblePitchSide": "left", "sideAgreement": 0.8},
        },
        coordinate_mode="screen-relative",
        metric_calibration=False,
        frame_evidence=frame_evidence,
        accepted_frame_calibrations={},
        accepted_automatic_direct_by_sample={},
        accepted_manual_direct_by_sample={},
        resolved_calibrations_by_sample={},
        manual_override_by_sample={},
        representative_manual_sample=None,
        rejected_frame_count=0,
        temporal_recovered_frame_count=0,
        metric_person_sample_count=0,
        metric_ball_sample_count=0,
        warnings=["one frame stayed unresolved"],
    )


def _resolved(sample_index: int) -> dict:
    return {
        "sampleIndex": sample_index,
        "sourceFrameIndex": sample_index + 1,
        "sceneTime": sample_index * 0.04,
        "solutionStatus": "direct-accepted",
        "projectionSource": "direct",
        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
        "frameWidth": 960,
        "frameHeight": 540,
    }


def _unresolved(sample_index: int, status: str = "unresolved") -> dict:
    return {
        "sampleIndex": sample_index,
        "sourceFrameIndex": sample_index + 1,
        "sceneTime": sample_index * 0.04,
        "solutionStatus": status,
        "projectionSource": "none",
        "rejectionReasons": ["insufficient-line-support"],
    }


def _scene_with_prior_reconstruction() -> dict:
    return {
        "id": "scene-1",
        "title": "Scene 1",
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                # A manual ball trajectory is authoritative user data; a
                # calibrate run must never disturb it.
                "reconstruction": {
                    "inputFingerprint": "sha256:abc",
                    "artifactManifest": {
                        "schemaVersion": 1,
                        "artifacts": {
                            "identityTimeline": {
                                "kind": "identity-timeline",
                                "schemaVersion": 1,
                                "digest": "sha256:prior-identity",
                            }
                        },
                    },
                },
            },
            "ball": {"mode": "manual", "keyframes": [{"t": 0.0, "x": 1, "y": 2}]},
            "tracks": [{"id": "home-01"}],
        },
    }


def test_gate_partitions_resolved_and_unresolved_frames():
    result = _calibration_result(
        [_resolved(0), _unresolved(1), _resolved(2), _unresolved(3, "temporal-rejected")]
    )
    gate = _frame_gate(result.frame_evidence)
    assert gate["totalFrames"] == 4
    assert gate["resolvedFrames"] == 2
    assert gate["unresolvedFrames"] == 2
    assert gate["resolvedRatio"] == 0.5
    assert [item["sampleIndex"] for item in gate["unresolvedSamples"]] == [1, 3]
    # Every sampled frame is browsable, resolved ones included.
    assert [item["sampleIndex"] for item in gate["frames"]] == [0, 1, 2, 3]
    assert [item["resolved"] for item in gate["frames"]] == [True, False, True, False]
    # Only resolved frames carry the homography for the inspection overlay.
    assert gate["frames"][0]["imageToPitch"] is not None
    assert gate["frames"][0]["frameWidth"] == 960
    assert gate["frames"][1]["imageToPitch"] is None
    assert calibration_stage_status(gate) == "review"


def test_gate_is_ready_only_when_every_frame_resolves():
    assert calibration_stage_status(_frame_gate([_resolved(0), _resolved(1)])) == "ready"


def test_publish_calibration_stage_invalidates_prior_derived_identity(tmp_path):
    scene = _scene_with_prior_reconstruction()
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    result = _calibration_result([_resolved(0), _unresolved(1)])

    published = publish_calibration_stage(scene, result, store=store)

    reconstruction = published["payload"]["videoAsset"]["reconstruction"]
    # Gate: one of two frames unresolved → review, stamped with the run inputs.
    assert reconstruction["stage"] == "calibration"
    review = reconstruction["calibrationReview"]
    assert review["status"] == "review"
    assert review["resolvedFrames"] == 1
    assert review["unresolvedFrames"] == 1
    assert review["inputFingerprint"] == "sha256:abc"
    # The job itself completed: the gate lives only in calibrationReview.
    assert reconstruction["status"] == "ready"
    assert reconstruction["processingStatus"] == "completed"
    # Calibration is offloaded exactly like a full run (no inline frameEvidence).
    assert "frameEvidence" not in reconstruction["calibration"]
    assert reconstruction["calibration"]["frameEvidenceCount"] == 2
    manifest = reconstruction["artifactManifest"]["artifacts"]
    assert "calibrationFrames" in manifest
    provenance = reconstruction["calibrationProvenance"]
    assert provenance["runId"] is None
    assert provenance["dataFingerprint"].startswith("sha256:")
    assert provenance["artifact"] == manifest["calibrationFrames"]
    assert provenance["totalFrames"] == 2
    assert provenance["resolvedFrames"] == 1
    # A prior reconstruction cannot be displayed under a newly produced
    # calibration. Operator-owned manual ball input survives, derived identity
    # artifacts and rendered tracks do not.
    assert set(manifest) == {"calibrationFrames"}
    assert reconstruction["resultState"] == "calibration-only"
    ball = published["payload"]["ball"]
    manual_keyframes = [{"t": 0.0, "x": 1, "y": 2}]
    assert ball["mode"] == "manual"
    assert ball["keyframes"] == manual_keyframes
    assert ball["manualKeyframes"] == manual_keyframes
    assert ball["automaticKeyframes"] == []
    assert (
        ball["automaticDiagnostics"]["status"]
        == "invalidated-by-calibration"
    )
    assert published["payload"]["tracks"] == []
    assert published["payload"]["canonicalPeople"] == []


def test_publish_calibration_stage_keeps_manual_orientation_fingerprint_stable(tmp_path):
    # A manually chosen pitch side must survive the calibrate publish unchanged,
    # or the run's input fingerprint would shift and the terminal compare-and-swap
    # would reject the (correct) result as stale.
    from app.scene_document import reconstruction_input_fingerprint

    scene = _scene_with_prior_reconstruction()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["pitchOrientation"] = {
        "visiblePitchSide": "left",
        "visiblePitchSideSource": "manual-calibration",
        "attackingGoal": "right",
        "attackingGoalSource": "manual",
    }
    reconstruction["pitchCalibrationOverrides"] = [
        {"id": "manual-frame-1", "sourceFrameIndex": 1, "imageToPitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    ]
    before = reconstruction_input_fingerprint(scene)

    result = _calibration_result([_resolved(0), _resolved(1)])
    result.quality["summary"]["visiblePitchSide"] = "right"  # detector disagrees with the operator
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publish_calibration_stage(scene, result, store=store)

    orientation = scene["payload"]["videoAsset"]["reconstruction"]["pitchOrientation"]
    assert orientation["visiblePitchSide"] == "left"
    assert orientation["visiblePitchSideSource"] == "manual-calibration"
    assert reconstruction_input_fingerprint(scene) == before


def test_publish_calibration_stage_ready_when_all_frames_resolve(tmp_path):
    scene = _scene_with_prior_reconstruction()
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    result = _calibration_result([_resolved(0), _resolved(1)])

    published = publish_calibration_stage(scene, result, store=store)

    reconstruction = published["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["calibrationReview"]["status"] == "ready"
    assert reconstruction["calibrationReview"]["unresolvedFrames"] == 0
    assert reconstruction["qualityVerdict"] == "pass"


def test_publish_calibration_stage_removes_prior_automatic_ball(tmp_path):
    scene = _scene_with_prior_reconstruction()
    scene["payload"]["ball"] = {
        "mode": "automatic",
        "keyframes": [{"t": 0.0, "x": 4.0, "z": 2.0}],
        "automaticKeyframes": [{"t": 0.0, "x": 4.0, "z": 2.0}],
        "manualKeyframes": [{"t": 0.5, "x": 1.0, "z": 1.0}],
    }

    publish_calibration_stage(
        scene,
        _calibration_result([_resolved(0)]),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
    )

    ball = scene["payload"]["ball"]
    assert ball["mode"] == "automatic"
    assert ball["keyframes"] == []
    assert ball["automaticKeyframes"] == []
    # Inactive operator-authored input is independent from reconstruction.
    assert ball["manualKeyframes"] == [
        {"t": 0.5, "x": 1.0, "z": 1.0}
    ]
