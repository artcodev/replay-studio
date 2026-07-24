from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from app.reconstruction_calibration_snapshot import (
    calibration_artifact_input,
    calibration_data_fingerprint,
    load_persisted_calibration_snapshot,
)
from app.reconstruction_errors import ReconstructionError


def _reference(digest: str = "a" * 64) -> dict:
    return {
        "id": f"sha256:{digest}",
        "kind": "reconstruction.calibration-frames",
        "schemaVersion": 1,
        "uri": f"artifact://sha256/{digest}",
        "sha256": digest,
        "byteSize": 123,
        "contentType": "application/json",
    }


def _scene() -> dict:
    evidence = {
        "sampleIndex": 0,
        "sourceFrameIndex": 7,
        "sceneTime": 0.0,
        "frameWidth": 960,
        "frameHeight": 540,
        "status": "accepted",
        "solutionStatus": "direct-accepted",
        "projectionSource": "direct",
        "source": "pnlcalib-points-lines",
        "confidence": 0.94,
        "imageToPitch": [
            [0.1, 0.0, -48.0],
            [0.0, 0.1, -27.0],
            [0.0, 0.0, 1.0],
        ],
        "positionUncertaintyMetres": 0.7,
        "cameraMotion": {
            "status": "first-frame",
            "currentToReference": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    }
    scene = {
        "id": "snapshot-scene",
        "duration": 1.0,
        "payload": {
            "pitch": {"length": 105.0, "width": 68.0},
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "calibrationInputFingerprint": "sha256:inputs",
                    "coordinateSpace": "pitch-metric",
                    "pitchOrientation": {
                        "visiblePitchSide": "left",
                        "visiblePitchSideSource": "calibration",
                        "visiblePitchSideAgreement": 1.0,
                    },
                    "pitchCalibration": {
                        **evidence,
                        "status": "ready",
                        "method": "pnlcalib-points-lines",
                        "rectangle": "penalty-left",
                    },
                    "calibration": {
                        "schemaVersion": 2,
                        "summary": {
                            "usableCoverage": 1.0,
                            "directCoverage": 1.0,
                            "maxGapSeconds": 0.0,
                            "reprojectionP95": 2.0,
                            "sideAgreement": 1.0,
                        },
                        "manualFrameAnchors": [],
                        "frameEvidence": [evidence],
                    },
                    "artifactManifest": {
                        "schemaVersion": 1,
                        "artifacts": {
                            "calibrationFrames": _reference(),
                        },
                    },
                },
            },
        },
    }
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    data_fingerprint = calibration_data_fingerprint(reconstruction)
    reconstruction["calibrationProvenance"] = {
        "schemaVersion": 1,
        "runId": "calibration-run-1",
        "producedAt": "2026-07-21T12:00:00+00:00",
        "calibrationInputFingerprint": "sha256:inputs",
        "dataFingerprint": data_fingerprint,
        "artifact": _reference(),
        "totalFrames": 1,
        "resolvedFrames": 1,
        "unresolvedFrames": 0,
    }
    reconstruction["calibrationArtifactInput"] = calibration_artifact_input(
        reconstruction
    )
    return scene


def test_snapshot_loads_exact_pinned_calibration_without_solving():
    scene = _scene()
    snapshot = load_persisted_calibration_snapshot(
        scene,
        [(Path("/tmp/frame_00007.jpg"), 0.0)],
    )

    assert list(snapshot.result.resolved_calibrations_by_sample) == [0]
    assert snapshot.result.resolved_calibrations_by_sample[0].method == (
        "pnlcalib-points-lines"
    )
    assert snapshot.temporal.anchor_by_sample == {0: 7}
    assert snapshot.temporal.uncertainty_by_sample == {0: 0.7}
    assert snapshot.provenance["producerRunId"] == "calibration-run-1"
    assert snapshot.provenance["artifactSha256"] == "a" * 64
    assert snapshot.provenance["projectionSourceCounts"] == {"direct": 1}


def test_snapshot_rejects_calibration_data_changed_after_queue():
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["calibration"][
        "frameEvidence"
    ][0]["imageToPitch"][0][0] = 0.2

    with pytest.raises(ReconstructionError, match="fingerprint validation"):
        load_persisted_calibration_snapshot(
            scene,
            [(Path("/tmp/frame_00007.jpg"), 0.0)],
        )


def test_snapshot_rejects_evidence_for_a_different_sampled_frame():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["calibration"]["frameEvidence"][0]["sourceFrameIndex"] = 8
    new_fingerprint = calibration_data_fingerprint(reconstruction)
    reconstruction["calibrationArtifactInput"][
        "dataFingerprint"
    ] = new_fingerprint

    with pytest.raises(ReconstructionError, match="different source frame"):
        load_persisted_calibration_snapshot(
            scene,
            [(Path("/tmp/frame_00007.jpg"), 0.0)],
        )


def test_calibration_artifact_input_requires_completed_provenance():
    scene = _scene()
    reconstruction = deepcopy(
        scene["payload"]["videoAsset"]["reconstruction"]
    )
    reconstruction.pop("calibrationProvenance")

    with pytest.raises(ReconstructionError, match="provenance is missing"):
        calibration_artifact_input(reconstruction)


def test_calibration_artifact_input_rejects_data_changed_after_publication():
    scene = _scene()
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["calibration"]["frameEvidence"][0]["confidence"] = 0.4

    with pytest.raises(ReconstructionError, match="published fingerprint"):
        calibration_artifact_input(reconstruction)
