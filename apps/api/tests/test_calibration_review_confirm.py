from __future__ import annotations

import pytest

from app.reconstruction_errors import ReconstructionError
from app.scene_calibration_review_command import confirm_calibration_review
from app.reconstruction_calibration_fingerprint import calibration_input_fingerprint


def _scene_in_review(status: str = "review") -> dict:
    scene = {
        "id": "scene-1",
        "title": "Scene 1",
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "analysisFps": 25.0,
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "reconstruction": {
                    "status": "ready",
                    "stage": "calibration",
                    "model": "yolo26m.pt",
                    "calibrationReview": {
                        "status": status,
                        "unresolvedFrames": 3,
                        "unresolvedSamples": [
                            {"sampleIndex": 2, "resolved": False},
                            {"sampleIndex": 4, "resolved": False},
                            {"sampleIndex": 7, "resolved": False},
                        ],
                    },
                },
            }
        },
    }
    # Stamp the gate with the inputs it was computed for so it is confirmable.
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["calibrationReview"]["calibrationInputFingerprint"] = (
        calibration_input_fingerprint(scene)
    )
    return scene


@pytest.fixture(autouse=True)
def _persist_returns_scene(monkeypatch):
    # The command persists through the scene repository; the gate logic is what
    # is under test, so persistence is the identity.
    monkeypatch.setattr("app.scene_repository.scenes.put", lambda value: value)


def test_confirm_flips_a_pending_review_to_confirmed():
    scene = _scene_in_review()

    confirmed = confirm_calibration_review(scene)

    review = confirmed["payload"]["videoAsset"]["reconstruction"]["calibrationReview"]
    assert review["status"] == "confirmed"
    assert "confirmedAt" in review
    # The unresolved count is preserved: confirming accepts the gap, it does not
    # pretend the frames resolved.
    assert review["unresolvedFrames"] == 3
    assert review["fallbackPolicy"] == "explicit-image-fallback"
    assert review["fallbackSampleIndices"] == [2, 4, 7]


def test_confirm_refuses_when_there_is_no_calibration_stage():
    scene = _scene_in_review()
    scene["payload"]["videoAsset"]["reconstruction"]["stage"] = "reconstruction"

    with pytest.raises(ReconstructionError, match="no calibration review"):
        confirm_calibration_review(scene)


def test_confirm_explains_when_no_published_review_exists():
    scene = _scene_in_review()
    scene["payload"]["videoAsset"]["reconstruction"].pop("calibrationReview")

    with pytest.raises(ReconstructionError, match="No published calibration review"):
        confirm_calibration_review(scene)


def test_confirm_refuses_when_inputs_changed_since_calibration():
    scene = _scene_in_review()
    # A manual anchor (or any fingerprinted edit) during review invalidates the
    # gate; the stamped fingerprint no longer matches the current inputs.
    scene["payload"]["videoAsset"]["reconstruction"]["calibrationReview"][
        "calibrationInputFingerprint"
    ] = "sha256:stale"

    with pytest.raises(ReconstructionError, match="inputs changed"):
        confirm_calibration_review(scene)


def test_confirm_refuses_while_calibration_is_still_running():
    scene = _scene_in_review()
    scene["payload"]["videoAsset"]["reconstruction"]["status"] = "processing"

    with pytest.raises(ReconstructionError, match="Wait for calibration"):
        confirm_calibration_review(scene)


def test_confirm_is_idempotent_on_an_already_confirmed_gate():
    scene = _scene_in_review(status="confirmed")

    confirmed = confirm_calibration_review(scene)

    assert (
        confirmed["payload"]["videoAsset"]["reconstruction"]["calibrationReview"][
            "status"
        ]
        == "confirmed"
    )
