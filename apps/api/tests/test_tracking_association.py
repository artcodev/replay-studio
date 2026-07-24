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
    assert left_identity.points[-1]["associationCost"] < 0.80


def test_metric_tracking_does_not_reject_a_player_only_for_screen_jump():
    # A camera pan or an abrupt direction change can move the player hundreds
    # of pixels while the calibrated ground displacement remains plausible.
    tracks = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(420.0, -9.2, 0)], 0.1),
        ]
    )

    assert len(tracks) == 1
    assert len(tracks[0].points) == 2
    association = tracks[0].points[-1]["associationDiagnostics"]
    assert association["coordinateMode"] == "metric"
    assert association["pixelDistance"] > association["pixelGate"]


def test_explicit_image_fallback_keeps_the_hard_pixel_gate():
    first = _detection(100.0, -10.0, 0)
    second = _detection(420.0, -9.2, 0)
    for item in (first, second):
        item.pitch_x = None
        item.pitch_z = None

    tracks = _track_people(
        [([first], 0.0), ([second], 0.1)],
        coordinate_policy="explicit-image-fallback",
        image_fallback_sample_indices=[0, 1],
    )

    assert len(tracks) == 2


def test_tracking_diagnostics_explain_unprojected_and_matched_detections():
    missing = _detection(100.0, -10.0, 0)
    missing.pitch_x = None
    missing.pitch_z = None
    missing.metric_projection_reason = "outside-pitch-length"
    diagnostics = {}

    tracks = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(105.0, -9.7, 0), missing], 0.1),
        ],
        diagnostics=diagnostics,
    )

    assert len(tracks) == 1
    assert diagnostics["model"] == "metric-first-ambiguity-guarded-v2"
    assert diagnostics["outcomeCounts"]["matched-existing-track"] == 1
    assert diagnostics["rejectionReasonCounts"]["outside-pitch-length"] == 1
    rejected = diagnostics["frames"][1]["detections"][1]
    assert rejected["status"] == "untracked"
    assert rejected["reason"] == "outside-pitch-length"


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


def test_low_confidence_detection_continues_track_but_cannot_create_track():
    continued = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(103.0, -9.7, 0, confidence=0.06)], 0.1),
        ]
    )
    low_confidence_only = _track_people(
        [
            ([_detection(500.0, 20.0, 2, confidence=0.06)], 0.0),
            ([_detection(502.0, 20.2, 2, confidence=0.06)], 0.1),
        ]
    )

    assert len(continued) == 1
    assert len(continued[0].points) == 2
    assert low_confidence_only == []


def test_metric_required_does_not_create_a_track_without_metric_position():
    detection = _detection(100.0, -10.0, 0)
    detection.pitch_x = None
    detection.pitch_z = None

    assert _track_people([([detection], 0.0)]) == []


def test_high_cost_appearance_mismatch_starts_a_new_tracklet():
    """A plausible metric jump must not overwrite contradictory identity."""

    tracks = _track_people(
        [
            ([_detection(100.0, -10.0, 0)], 0.0),
            ([_detection(350.0, -6.0, 1)], 0.1),
        ]
    )

    assert len(tracks) == 2
    assert [len(track.points) for track in tracks] == [1, 1]


def test_image_fallback_requires_an_explicit_tracker_policy():
    detection = _detection(100.0, -10.0, 0)
    detection.pitch_x = None
    detection.pitch_z = None

    tracks = _track_people(
        [([detection], 0.0)],
        coordinate_policy="explicit-image-fallback",
        image_fallback_sample_indices=[0],
    )

    assert len(tracks) == 1
    assert "pitchX" not in tracks[0].points[0]


def test_image_fallback_consent_is_scoped_to_specific_samples():
    missing = _detection(100.0, -10.0, 0)
    missing.pitch_x = None
    missing.pitch_z = None
    metric = _detection(104.0, -9.7, 0)

    tracks = _track_people(
        [([missing], 0.0), ([metric], 0.1)],
        coordinate_policy="explicit-image-fallback",
        image_fallback_sample_indices=[0],
    )

    assert len(tracks) == 1
    assert len(tracks[0].points) == 2

    unapproved = _detection(100.0, -10.0, 0)
    unapproved.pitch_x = None
    unapproved.pitch_z = None
    assert _track_people(
        [([unapproved], 0.0)],
        coordinate_policy="explicit-image-fallback",
        image_fallback_sample_indices=[1],
    ) == []
