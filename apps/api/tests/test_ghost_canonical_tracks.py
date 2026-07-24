"""A canonical person the roster publisher drops remains a provisional actor.

An early-then-gone player falls below the publication length gate, so it never
reaches the 3D layer and survives only as an "Unassigned person" identity row.
`publish_provisional_canonical_tracks` projects each such canonical track with
the normal trajectory pipeline without misrepresenting observed positions as
ghosts.
"""

import numpy as np

from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_reid_evidence import capture_detection_observations
from app.reconstruction_scene_track_publisher import (
    publish_provisional_canonical_tracks,
)
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation


def _scene() -> dict:
    return {
        "id": "scene-ghost",
        "duration": 1.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "videoAsset": {"sourceStart": 0.0},
        },
    }


def _track(track_id: int, canonical_id: str, count: int, *, start_frame: int) -> TrackState:
    track = TrackState(id=track_id)
    track.canonical_person_id = canonical_id
    for index in range(count):
        frame_index = start_frame + index
        detection = Detection(
            x=110.0 + index * 6,
            y=200.0,
            width=20.0,
            height=40.0,
            confidence=0.8,
            feature=np.zeros(12, dtype=np.float32),
            pitch_x=index * 0.5,
            pitch_z=0.0,
            projection_source="direct",
            calibration_frame_index=frame_index,
            position_uncertainty_metres=0.8,
        )
        capture_detection_observations([detection], frame_index)
        append_track_observation(track, detection, frame_index=index, time=index * 0.1)
    return track


def test_dropped_canonical_person_is_published_as_provisional_identity():
    published = _track(1, "canonical-published", 6, start_frame=100)
    dropped = _track(2, "canonical-dropped", 3, start_frame=300)

    provisional = publish_provisional_canonical_tracks(
        [published, dropped],
        [{"id": "auto-home-01", "canonicalPersonId": "canonical-published"}],
        {1: "home"},
        {"home": "#ffffff"},
        (960, 540),
        _scene(),
        coordinate_mode="metric",
    )

    assert len(provisional) == 1
    track = provisional[0]
    assert track["id"] == "provisional-canonical-dropped"
    assert track["provisional"] is True
    assert "ghost" not in track
    assert track["source"] == "provisional"
    assert track["number"] == 0
    # Not clustered into a team → excluded from any roster, rendered neutral.
    assert track["teamId"] == "unknown"
    assert track["canonicalPersonId"] == "canonical-dropped"
    assert track["keyframes"], "a provisional person must carry observed keyframes"


def test_provisional_track_carries_the_team_hint_when_clustered():
    dropped = _track(4, "canonical-hinted", 3, start_frame=500)

    provisional = publish_provisional_canonical_tracks(
        [dropped],
        [],
        {4: "away"},
        {"away": "#2b6cff"},
        (960, 540),
        _scene(),
        coordinate_mode="metric",
    )

    assert provisional[0]["teamId"] == "away"
    assert provisional[0]["color"] == "#2b6cff"


def test_excluded_canonical_person_is_never_published():
    excluded = _track(3, "canonical-excluded", 3, start_frame=700)
    excluded.identity_status = "excluded"

    provisional = publish_provisional_canonical_tracks(
        [excluded],
        [],
        {},
        {},
        (960, 540),
        _scene(),
        coordinate_mode="metric",
    )

    assert provisional == []


def test_no_provisional_tracks_are_published_without_a_metric_pitch():
    dropped = _track(5, "canonical-screen", 3, start_frame=900)

    provisional = publish_provisional_canonical_tracks(
        [dropped],
        [],
        {},
        {},
        (960, 540),
        _scene(),
        coordinate_mode="unavailable",
    )

    assert provisional == []
