from app.reconstruction_latent_presence import materialize_continuous_presence


PITCH = {"length": 105.0, "width": 68.0}


def _observed(time: float, x: float = 1.0) -> dict:
    return {
        "t": time,
        "x": x,
        "z": 2.0,
        "confidence": 0.9,
        "observed": True,
        "presenceState": "observed",
    }


def test_sub_frame_segment_offset_does_not_create_a_false_ghost_at_zero():
    keyframes, presence = materialize_continuous_presence(
        [_observed(0.025), _observed(0.058, 1.2)],
        1.0,
        PITCH,
        1,
    )

    assert keyframes[0]["t"] == 0.025
    assert keyframes[0]["observed"] is True
    assert {item.get("presenceState") for item in keyframes} == {"observed"}
    assert presence["observedStart"] == 0.025


def test_unobserved_prefix_is_not_invented():
    keyframes, presence = materialize_continuous_presence(
        [_observed(0.5), _observed(0.533, 1.2)],
        1.0,
        PITCH,
        1,
    )

    assert keyframes[0]["t"] == 0.5
    assert keyframes[0]["observed"] is True
    assert presence["policy"] == "observed-window-with-latent-gaps"
