from __future__ import annotations

from app.reconstruction_track_state import TrackState
from app.reconstruction_track_trajectory import project_track_trajectory


def _track(spans: list[list[tuple[float, float]]], cadence: float = 0.1) -> TrackState:
    """Build a metric track from spans of (x, z); spans are contiguous in time."""

    track = TrackState(id=1)
    time = 0.0
    for span in spans:
        for x, z in span:
            track.points.append(
                {
                    "t": round(time, 3),
                    "px": 0.0,
                    "py": 0.0,
                    "confidence": 0.9,
                    "frameIndex": int(round(time * 10)),
                    "pitchX": x,
                    "pitchZ": z,
                    "projectionSource": "direct",
                    "calibrationFrameIndex": 1,
                    "positionUncertaintyMetres": 0.5,
                }
            )
            time += cadence
    return track


def _walk(start_x: float, count: int, step: float = 0.4) -> list[tuple[float, float]]:
    return [(start_x + index * step, 0.0) for index in range(count)]


def test_uncertainty_prevents_a_false_splice_for_measurement_noise():
    # A 1.6 m step over 0.1 s (16 m/s) is projection noise, not a new person:
    # the early, clearly visible span must survive.
    early = _walk(30.0, 5)
    late = _walk(early[-1][0] + 1.6, 6)
    trajectory = project_track_trajectory(
        _track([early, late]), (960, 540), {"length": 105, "width": 68}, "metric"
    )

    quality = trajectory.quality
    assert quality["retainedObservationCount"] == 11
    assert quality["discardedObservationCount"] == 0
    assert quality["fragmentCount"] == 1
    assert quality["retainedFragmentCount"] == 1
    assert quality["softSpliceBridgedCount"] == 0
    assert quality["identitySpliceCount"] == 0
    assert quality["retentionPolicy"] == "soft-splice-chains-v1"


def test_identity_splice_still_quarantines_the_other_person():
    # A 6.7 m teleport over 0.1 s (67 m/s) is an identity switch: only the
    # longest chain survives, the foreign fragment stays discarded.
    foreign = _walk(30.0, 5)
    own = _walk(foreign[-1][0] + 6.7, 8)
    trajectory = project_track_trajectory(
        _track([foreign, own]), (960, 540), {"length": 105, "width": 68}, "metric"
    )

    quality = trajectory.quality
    assert quality["retainedObservationCount"] == 8
    assert quality["discardedObservationCount"] == 5
    assert quality["retainedFragmentCount"] == 1
    assert quality["discardedFragmentCount"] == 1
    assert quality["identitySpliceCount"] == 1
    assert quality["discardedRanges"] == [
        {
            "chainIndex": 0,
            "startTime": 0.0,
            "endTime": 0.4,
            "startFrameIndex": 0,
            "endFrameIndex": 4,
            "observationCount": 5,
            "reason": "identity-grade-speed-boundary",
        }
    ]


def test_home02_shape_recovers_the_early_span_behind_a_soft_splice():
    # The real shot-02 pattern: foreign head | hard splice | own early span |
    # soft splice | own accepted tail. The own early span must now be
    # retained together with the tail.
    foreign_head = _walk(30.0, 5)
    own_early = _walk(foreign_head[-1][0] + 6.7, 8)
    own_tail = _walk(own_early[-1][0] + 1.6, 9)
    trajectory = project_track_trajectory(
        _track([foreign_head, own_early, own_tail]),
        (960, 540),
        {"length": 105, "width": 68},
        "metric",
    )

    quality = trajectory.quality
    assert quality["retainedObservationCount"] == 17
    assert quality["discardedObservationCount"] == 5
    assert quality["retainedFragmentCount"] == 1
    assert quality["softSpliceBridgedCount"] == 0
    assert quality["identitySpliceCount"] == 1
    retained_times = [
        keyframe["t"] for keyframe in trajectory.observed_keyframes
    ]
    # Observed coverage now starts right after the identity splice, not at
    # the tail: the "clearly visible early position" is published.
    assert retained_times[0] == 0.5
