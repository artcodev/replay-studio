from __future__ import annotations

import pytest

import app.track_trajectory_corrections as corrections_module
from app.reconstruction_errors import ReconstructionError
from app.reconstruction_series import reconstruction_series_window
from app.track_trajectory_corrections import (
    apply_track_trajectory_correction,
    set_track_trajectory,
    track_trajectory_corrections,
)


@pytest.fixture(autouse=True)
def stored_writes(monkeypatch):
    written: list[dict] = []
    monkeypatch.setattr(
        corrections_module.scenes,
        "put",
        lambda scene: (written.append(scene), scene)[1],
    )
    return written


def _scene(*, canonical: str | None = "person-7", status: str | None = None) -> dict:
    track = {"id": "auto-home-02", "label": "Home track 02"}
    if canonical:
        track["canonicalPersonId"] = canonical
    return {
        "id": "scene-1",
        "duration": 10.0,
        "payload": {
            "tracks": [track],
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {"status": status} if status else {},
            },
        },
    }


def test_correction_is_anchored_to_the_canonical_identity_not_the_render_track():
    scene = _scene()
    set_track_trajectory(scene, "auto-home-02", [{"t": 1.0, "x": 10.0, "z": 2.0}])

    stored = scene["payload"]["trackTrajectoryCorrections"]
    assert len(stored) == 1
    # Anchored to the identity that survives a rebuild, never to auto-home-02.
    assert stored[0]["canonicalPersonId"] == "person-7"
    assert stored[0]["keyframes"] == [{"t": 1.0, "x": 10.0, "z": 2.0}]


def test_correction_survives_a_rebuild_that_renumbers_render_tracks():
    scene = _scene()
    set_track_trajectory(scene, "auto-home-02", [{"t": 1.0, "x": 10.0, "z": 2.0}])
    saved = scene["payload"]["trackTrajectoryCorrections"]

    # A later run renders the same person under a different track id.
    rebuilt = _scene()
    rebuilt["payload"]["tracks"] = [
        {"id": "auto-home-05", "canonicalPersonId": "person-7"}
    ]
    rebuilt["payload"]["trackTrajectoryCorrections"] = saved

    assert track_trajectory_corrections(rebuilt) == {
        "person-7": [{"t": 1.0, "x": 10.0, "z": 2.0}]
    }


def test_a_track_without_canonical_identity_is_refused_rather_than_silently_lost():
    with pytest.raises(ReconstructionError, match="no canonical identity"):
        set_track_trajectory(
            _scene(canonical=None), "auto-home-02", [{"t": 1.0, "x": 1.0, "z": 1.0}]
        )


def test_manual_points_override_model_keyframes_and_are_marked_manual():
    merged = apply_track_trajectory_correction(
        [
            {"t": 0.0, "x": 0.0, "z": 0.0, "observed": True},
            {"t": 1.0, "x": 5.0, "z": 5.0, "observed": False},
            {"t": 2.0, "x": 9.0, "z": 9.0, "observed": True},
        ],
        [{"t": 1.0, "x": -3.0, "z": 4.0}, {"t": 1.5, "x": -2.0, "z": 4.5}],
    )

    times = [item["t"] for item in merged]
    assert times == [0.0, 1.0, 1.5, 2.0]
    corrected = merged[1]
    assert (corrected["x"], corrected["z"]) == (-3.0, 4.0)
    # A user-authored position must never be reported as a model observation.
    assert corrected["positionSource"] == "manual"
    assert corrected["observed"] is True
    assert merged[0]["x"] == 0.0 and merged[3]["x"] == 9.0


def test_series_window_publishes_the_correction_over_the_artifact(monkeypatch):
    scene = _scene()
    scene["payload"]["trackTrajectoryCorrections"] = [
        {"canonicalPersonId": "person-7", "keyframes": [{"t": 1.0, "x": -3.0, "z": 4.0}]}
    ]
    monkeypatch.setattr(
        "app.reconstruction_series.load_dense_reconstruction_artifacts",
        lambda *_args, **_kwargs: {
            "identityTimeline": {
                "tracks": [
                    {
                        "id": "auto-home-02",
                        "keyframes": [
                            {"t": 1.0, "x": 5.0, "z": 5.0},
                            {"t": 2.0, "x": 9.0, "z": 9.0},
                        ],
                        "observations": [],
                    }
                ],
                "canonicalPeople": [],
            }
        },
    )

    window = reconstruction_series_window(scene, start=0.0, end=5.0)

    keyframes = window["tracks"][0]["keyframes"]
    corrected = next(item for item in keyframes if item["t"] == 1.0)
    assert (corrected["x"], corrected["z"]) == (-3.0, 4.0)
    assert corrected["positionSource"] == "manual"


def test_invalid_or_running_edits_fail_closed():
    with pytest.raises(ReconstructionError, match="Wait for reconstruction"):
        set_track_trajectory(
            _scene(status="processing"), "auto-home-02", [{"t": 1.0, "x": 1.0, "z": 1.0}]
        )
    with pytest.raises(ReconstructionError, match="outside the scene"):
        set_track_trajectory(
            _scene(), "auto-home-02", [{"t": 99.0, "x": 1.0, "z": 1.0}]
        )
    with pytest.raises(ReconstructionError, match="finite t, x and z"):
        set_track_trajectory(
            _scene(), "auto-home-02", [{"t": 1.0, "x": float("nan"), "z": 1.0}]
        )


def test_clearing_all_points_removes_the_correction_entry():
    scene = _scene()
    set_track_trajectory(scene, "auto-home-02", [{"t": 1.0, "x": 1.0, "z": 1.0}])
    set_track_trajectory(scene, "auto-home-02", [])
    assert scene["payload"]["trackTrajectoryCorrections"] == []
