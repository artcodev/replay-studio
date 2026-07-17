from app.model_comparison import _comparison_summary, _pair_frame_observations


def _observation(x: float, *, inside: bool = True) -> dict:
    return {
        "x": x,
        "y": 100.0,
        "height": 30.0,
        "insidePitch": inside,
    }


def _run(*, stable: int, accepted: int, outside: int, frames: int = 5) -> dict:
    return {
        "frameCount": frames,
        "stableTrackCount": stable,
        "acceptedTrackCount": accepted,
        "outsidePitchDetections": outside,
    }


def test_pair_frame_observations_separates_model_only_detections():
    shared, baseline_only, candidate_only = _pair_frame_observations(
        [_observation(10), _observation(100)],
        [_observation(12), _observation(200)],
    )

    assert shared == 1
    assert [item["x"] for item in baseline_only] == [100]
    assert [item["x"] for item in candidate_only] == [200]


def test_comparison_recommends_candidate_only_for_meaningful_in_pitch_gain():
    summary = _comparison_summary(
        _run(stable=8, accepted=7, outside=2),
        _run(stable=9, accepted=8, outside=2),
        [[_observation(10)]],
        [[_observation(10), _observation(100), _observation(140), _observation(180)]],
    )

    assert summary["sharedDetections"] == 1
    assert summary["candidateOnlyInPitchDetections"] == 3
    assert summary["inPitchObservationGain"] == 3
    assert summary["verdict"] == "candidate"


def test_comparison_keeps_outside_detections_out_of_recovery_gain():
    summary = _comparison_summary(
        _run(stable=8, accepted=7, outside=0),
        _run(stable=9, accepted=8, outside=3),
        [[_observation(10)]],
        [[_observation(10), _observation(100, inside=False), _observation(140, inside=False), _observation(180, inside=False)]],
    )

    assert summary["candidateOnlyDetections"] == 3
    assert summary["candidateOnlyInPitchDetections"] == 0
    assert summary["verdict"] == "review"


def test_comparison_requires_review_when_candidate_trades_in_pitch_coverage_for_less_noise():
    summary = _comparison_summary(
        _run(stable=24, accepted=19, outside=59, frames=21),
        _run(stable=22, accepted=19, outside=45, frames=21),
        [[_observation(10), _observation(40), _observation(70), _observation(100)]],
        [[_observation(10)]],
    )

    assert summary["inPitchObservationGain"] == -3
    assert summary["outsidePitchDetectionDelta"] == -14
    assert summary["verdict"] == "review"
