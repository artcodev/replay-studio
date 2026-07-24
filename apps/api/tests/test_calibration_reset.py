from __future__ import annotations

import pytest

from app.reconstruction_errors import ReconstructionError
from app.scene_calibration_reset_command import reset_scene_calibration


def _scene() -> dict:
    return {
        "id": "scene-1",
        "payload": {
            "videoAsset": {
                "reconstruction": {
                    "status": "ready",
                    "stage": "calibration",
                    "pitchCalibrationOverrides": [{"id": "manual-frame-1"}],
                    "calibrationReview": {"status": "ready"},
                    "calibrationInputFingerprint": "sha256:inputs",
                    "calibrationProvenance": {"dataFingerprint": "sha256:data"},
                    "calibrationArtifactInput": {"schemaVersion": 1},
                    "calibrationWarnings": ["warning"],
                    "trackingCoordinatePolicy": "metric-required",
                    "pitchCalibration": {"status": "ready"},
                    "calibration": {"schemaVersion": 2},
                    "pitchOrientation": {
                        "visiblePitchSide": "left",
                        "attackingGoal": "right",
                    },
                    "artifactManifest": {
                        "schemaVersion": 1,
                        "artifacts": {
                            "calibrationFrames": {"digest": "calibration"},
                            "identityTimeline": {"digest": "identity"},
                        },
                    },
                }
            }
        },
    }


def test_reset_revokes_all_calibration_authority(monkeypatch):
    monkeypatch.setattr(
        "app.scene_calibration_reset_command.scenes.put", lambda value: value
    )
    reset = reset_scene_calibration(_scene())
    reconstruction = reset["payload"]["videoAsset"]["reconstruction"]

    for field in (
        "pitchCalibrationOverrides",
        "calibrationReview",
        "calibrationInputFingerprint",
        "calibrationProvenance",
        "calibrationArtifactInput",
        "calibrationWarnings",
        "trackingCoordinatePolicy",
        "pitchCalibration",
        "calibration",
        "stage",
    ):
        assert field not in reconstruction
    assert reconstruction["artifactManifest"]["artifacts"] == {
        "identityTimeline": {"digest": "identity"}
    }
    assert reconstruction["pitchOrientation"] == {
        "visiblePitchSide": "left",
        "attackingGoal": "right",
    }


def test_reset_refuses_while_calibration_is_running():
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["status"] = "processing"

    with pytest.raises(ReconstructionError, match="Wait for calibration"):
        reset_scene_calibration(scene)
