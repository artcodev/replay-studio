import numpy as np
import pytest

from app.reconstruction_errors import ReconstructionError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation


def test_roster_conflict_is_rejected_before_any_track_evidence_mutates() -> None:
    track = TrackState(
        id=4,
        roster_binding_state="bound",
        roster_binding_annotation_ids={"binding-a"},
        manual_external_player_id="player-a",
    )
    detection = Detection(
        x=120.0,
        y=180.0,
        width=28.0,
        height=72.0,
        confidence=0.95,
        feature=np.ones(12, dtype=np.float32),
        annotation_id="binding-b",
        external_player_id="player-b",
        roster_binding_state="bound",
        roster_binding_annotation_ids={"binding-b"},
    )

    with pytest.raises(
        ReconstructionError,
        match="Conflicting dedicated roster corrections reached one raw track",
    ):
        append_track_observation(track, detection, frame_index=3, time=0.3)

    assert track.points == []
    assert track.feature_sum is None
    assert track.feature_count == 0
    assert track.last_frame == 0
    assert track.annotation_ids == set()
    assert track.roster_binding_annotation_ids == {"binding-a"}
