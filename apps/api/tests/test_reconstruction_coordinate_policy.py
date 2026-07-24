from __future__ import annotations

import pytest

from app.reconstruction_calibration_fingerprint import calibration_input_fingerprint
from app.reconstruction_coordinate_policy import (
    EXPLICIT_IMAGE_FALLBACK,
    METRIC_REQUIRED,
    resolve_full_run_coordinate_authorization,
    validate_runtime_calibration_coverage,
)
from app.reconstruction_errors import ReconstructionError


FINGERPRINT = "sha256:current"


def test_calibration_fingerprint_ignores_identity_but_tracks_geometry_inputs():
    scene = {
        "payload": {
            "pitch": {"length": 105.0, "width": 68.0},
            "videoAsset": {
                "id": "asset-1",
                "sourceStart": 1.0,
                "sourceEnd": 5.0,
                "analysisFps": 25.0,
                "reconstruction": {"frameAnnotations": []},
            },
        }
    }
    baseline = calibration_input_fingerprint(scene)
    scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] = [
        {"action": "confirm", "id": "identity-only"}
    ]
    assert calibration_input_fingerprint(scene) == baseline

    scene["payload"]["videoAsset"]["reconstruction"][
        "pitchCalibrationOverrides"
    ] = [{"sceneTime": 2.0, "anchors": []}]
    assert calibration_input_fingerprint(scene) != baseline


def test_calibration_fingerprint_tracks_direct_sampling_policy():
    scene = {
        "payload": {
            "pitch": {"length": 105.0, "width": 68.0},
            "videoAsset": {
                "id": "asset-1",
                "fps": 30.0,
                "analysisFps": 30.0,
                "reconstruction": {
                    "samplingFrameRate": 30.0,
                    "directCalibrationMaxGapSeconds": 0.0,
                },
            },
        }
    }
    every_frame = calibration_input_fingerprint(scene)

    scene["payload"]["videoAsset"]["reconstruction"][
        "directCalibrationMaxGapSeconds"
    ] = 1.0

    assert calibration_input_fingerprint(scene) != every_frame


def _review(status: str) -> dict:
    return {
        "stage": "calibration",
        "calibrationReview": {
            "status": status,
            "calibrationInputFingerprint": FINGERPRINT,
            "unresolvedSamples": [
                {"sampleIndex": 1, "resolved": False},
                {"sampleIndex": 3, "resolved": False},
            ],
            "confirmedAt": "2026-07-21T00:00:00Z",
        },
    }


def test_full_run_requires_a_current_calibration_gate():
    with pytest.raises(ReconstructionError, match="Run pitch calibration"):
        resolve_full_run_coordinate_authorization(
            {},
            calibration_input_fingerprint=FINGERPRINT,
        )


def test_ready_calibration_requires_metric_coordinates():
    policy, consent = resolve_full_run_coordinate_authorization(
        _review("ready"),
        calibration_input_fingerprint=FINGERPRINT,
    )

    assert policy == METRIC_REQUIRED
    assert consent is None


def test_pending_review_refuses_silent_fallback():
    with pytest.raises(ReconstructionError, match="explicitly authorize"):
        resolve_full_run_coordinate_authorization(
            _review("review"),
            calibration_input_fingerprint=FINGERPRINT,
        )


def test_confirmed_review_authorizes_only_its_unresolved_samples():
    policy, consent = resolve_full_run_coordinate_authorization(
        _review("confirmed"),
        calibration_input_fingerprint=FINGERPRINT,
    )

    assert policy == EXPLICIT_IMAGE_FALLBACK
    assert consent == {
        "policy": EXPLICIT_IMAGE_FALLBACK,
        "calibrationInputFingerprint": FINGERPRINT,
        "sampleIndices": [1, 3],
        "confirmedAt": "2026-07-21T00:00:00Z",
    }


def test_runtime_fails_when_metric_coverage_has_an_unapproved_gap():
    with pytest.raises(ReconstructionError, match="fix calibration"):
        validate_runtime_calibration_coverage(
            policy=METRIC_REQUIRED,
            consent=None,
            calibration_input_fingerprint=FINGERPRINT,
            sampled_frame_count=4,
            resolved_sample_indices=[0, 2, 3],
        )


def test_runtime_accepts_only_the_explicitly_confirmed_fallback_frames():
    consent = {
        "policy": EXPLICIT_IMAGE_FALLBACK,
        "calibrationInputFingerprint": FINGERPRINT,
        "sampleIndices": [1],
    }

    diagnostics = validate_runtime_calibration_coverage(
        policy=EXPLICIT_IMAGE_FALLBACK,
        consent=consent,
        calibration_input_fingerprint=FINGERPRINT,
        sampled_frame_count=4,
        resolved_sample_indices=[0, 2, 3],
    )

    assert diagnostics["fallbackSampleIndices"] == [1]

    with pytest.raises(ReconstructionError, match="unapproved unresolved frames"):
        validate_runtime_calibration_coverage(
            policy=EXPLICIT_IMAGE_FALLBACK,
            consent=consent,
            calibration_input_fingerprint=FINGERPRINT,
            sampled_frame_count=4,
            resolved_sample_indices=[0, 3],
        )
