import numpy as np

from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_person_tracking import track_people as _track_people


def _detection(
    x: float,
    pitch_x: float,
    feature_index: int,
    *,
    external_player_id: str | None = None,
    confidence: float = 0.8,
) -> Detection:
    feature = np.zeros(12, dtype=np.float32)
    feature[feature_index] = 1.0
    return Detection(
        x=x,
        y=280.0,
        width=18.0,
        height=42.0,
        confidence=confidence,
        feature=feature,
        pitch_x=pitch_x,
        pitch_z=2.0,
        projection_source="direct",
        calibration_frame_index=1,
        position_uncertainty_metres=0.4,
        external_player_id=external_player_id,
    )


def test_global_assignment_keeps_identity_when_players_cross_on_screen():
    frames = [
        (
            [
                _detection(100.0, -10.0, 0),
                _detection(200.0, 10.0, 1),
            ],
            0.0,
        ),
        (
            [
                _detection(190.0, -9.0, 0),
                _detection(110.0, 9.0, 1),
            ],
            0.1,
        ),
    ]

    tracks = _track_people(frames)

    assert len(tracks) == 2
    left_identity = next(track for track in tracks if track.points[0]["px"] == 100.0)
    right_identity = next(track for track in tracks if track.points[0]["px"] == 200.0)
    assert left_identity.points[-1]["px"] == 190.0
    assert left_identity.points[-1]["pitchX"] == -9.0
    assert right_identity.points[-1]["px"] == 110.0
    assert right_identity.points[-1]["pitchX"] == 9.0
    assert left_identity.points[-1]["associationCost"] < 1.05


def test_tracker_does_not_bridge_an_unobserved_gap_longer_than_650ms():
    tracks = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(104.0, -9.5, 0)], 0.8),
        ]
    )

    assert len(tracks) == 2
    assert [len(track.points) for track in tracks] == [1, 1]


def test_manual_roster_identity_is_a_hard_association_constraint():
    first = _detection(100.0, -10.0, 0, external_player_id="player-a")
    first.annotation_id = "anchor-a"
    first.annotation_kind = "home-player"
    first.roster_binding_state = "bound"
    first.roster_binding_annotation_ids = {"anchor-a"}
    second = _detection(103.0, -9.8, 0, external_player_id="player-b")
    second.annotation_id = "anchor-b"
    second.annotation_kind = "home-player"
    second.roster_binding_state = "bound"
    second.roster_binding_annotation_ids = {"anchor-b"}

    tracks = _track_people([([first], 0.0), ([second], 0.1)])

    assert len(tracks) == 2
    assert tracks[0].manual_external_player_id == "player-a"
    assert tracks[1].manual_external_player_id == "player-b"


def test_low_confidence_detection_continues_track_but_cannot_create_ghost():
    continued = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(103.0, -9.7, 0, confidence=0.06)], 0.1),
        ]
    )
    ghost_only = _track_people(
        [
            ([_detection(500.0, 20.0, 2, confidence=0.06)], 0.0),
            ([_detection(502.0, 20.2, 2, confidence=0.06)], 0.1),
        ]
    )

    assert len(continued) == 1
    assert len(continued[0].points) == 2
    assert ghost_only == []
