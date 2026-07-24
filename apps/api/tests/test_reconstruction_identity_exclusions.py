from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pitch_anchor_calibration import calibration_from_anchors
from app.pitch_calibration_contract import PitchCalibration
from app.pitch_calibration_orientation import canonicalize_penalty_side
from app.pitch_geometry import projected_pitch_markings
from app.reconstruction_pitch_side_command import set_scene_pitch_side
from app.reconstruction_errors import IdentityCorrectionError, ReconstructionError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete as delete_frame_person_annotation,
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_progress import ReconstructionProgress
from app.reconstruction_calibration_detection import (
    best_pitch_calibration as _best_pitch_calibration,
)
from app.reconstruction_metric_projection import (
    calibration_person_support as _calibration_person_support,
)
from app.reconstruction_identity_correction_service import (
    apply_track_identity_corrections as _apply_track_identity_corrections,
)
from app.reconstruction_identity_scene_corrections import (
    apply_scene_track_identity_corrections as _apply_scene_track_identity_corrections,
)
from app.reconstruction_identity_splitting import (
    apply_canonical_split_corrections as _apply_canonical_split_corrections,
)
from app.reconstruction_identity_read_model import (
    interpolate_scene_keyframes as _interpolate_scene_keyframes,
    saved_pitch_calibration as _saved_pitch_calibration,
)
from app.reconstruction_identity_validation import (
    validate_identity_corrections as _validate_identity_corrections,
)
from app.reconstruction_canonical_identity_resolution import (
    resolve_canonical_track_states as _resolve_canonical_track_states,
)
from app.reconstruction_team_classification import (
    cluster_color as _cluster_color,
    include_goalkeeper_candidates as _include_goalkeeper_candidates,
)
from app.person_appearance import is_pitch_person as _is_pitch_person
from app.reconstruction_person_annotations import (
    apply_person_annotations as _apply_person_annotations,
    frame_annotations as _frame_annotations,
)
from app.scene_document import reconstruction_input_fingerprint


def test_identity_scoped_exclusion_removes_only_matching_raw_track():
    def raw_track(track_id: int, x: float) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
        )

    previous_keyframes = [
        {"t": index * 0.2, "x": index * 0.1, "z": 2.0, "confidence": 0.9, "observed": True}
        for index in range(4)
    ]
    scene = {
        "payload": {
            "tracks": [{"id": "auto-home-01", "keyframes": previous_keyframes}],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    corrected = _apply_track_identity_corrections(
        [raw_track(1, 0.0), raw_track(2, 20.0)],
        scene,
    )

    assert [track.id for track in corrected] == [2]


def test_identity_exclusion_prefers_exact_annotation_anchor_after_calibration_shift():
    def raw_track(track_id: int, x: float, annotation_id: str | None = None) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
            annotation_ids={annotation_id} if annotation_id else set(),
        )

    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {"t": index * 0.2, "x": index * 0.1, "z": 2.0, "observed": True}
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    corrected = _apply_track_identity_corrections(
        [raw_track(1, 20.0, "phantom"), raw_track(2, 0.0)],
        scene,
    )

    assert [track.id for track in corrected] == [2]


def test_identity_exclusion_rejects_ambiguous_geometry_without_exact_anchor():
    def raw_track(track_id: int, x: float) -> TrackState:
        return TrackState(
            id=track_id,
            points=[
                {
                    "t": index * 0.2,
                    "px": 100.0 + index,
                    "py": 200.0,
                    "pitchX": x + index * 0.1,
                    "pitchZ": 2.0,
                    "confidence": 0.9,
                    "positionUncertaintyMetres": 0.5,
                }
                for index in range(4)
            ],
            feature_sum=np.zeros(12, dtype=np.float32),
            feature_count=1,
            last_frame=3,
            last_height=40.0,
        )

    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {
                            "t": index * 0.2,
                            "x": index * 0.1,
                            "z": 2.0,
                            "observed": True,
                            "positionUncertaintyMetres": 0.5,
                        }
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    with pytest.raises(IdentityCorrectionError, match="ambiguous") as caught:
        _apply_track_identity_corrections(
            [raw_track(1, 0.2), raw_track(2, 0.3)],
            scene,
        )

    diagnostic = caught.value.diagnostic
    assert diagnostic["correctionId"] == "phantom"
    assert diagnostic["action"] == "exclude"
    assert diagnostic["status"] == "ambiguous"
    assert diagnostic["reason"] == "nearby-trajectories"
    assert [item["rawTrackId"] for item in diagnostic["candidates"]] == [1, 2]
    assert all("medianDistanceMetres" in item for item in diagnostic["candidates"])


def test_identity_exclusion_reports_unresolved_when_remap_evidence_is_missing():
    raw = TrackState(
        id=7,
        points=[
            {
                "t": index * 0.2,
                "px": 100.0 + index,
                "py": 200.0,
                "confidence": 0.9,
            }
            for index in range(4)
        ],
        feature_sum=np.zeros(12, dtype=np.float32),
        feature_count=1,
        last_frame=3,
        last_height=40.0,
    )
    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {
                            "t": index * 0.2,
                            "x": index * 0.1,
                            "z": 2.0,
                            "observed": True,
                        }
                        for index in range(4)
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "phantom",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "identity",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    with pytest.raises(IdentityCorrectionError, match="could not resolve") as caught:
        _apply_track_identity_corrections([raw], scene)

    diagnostic = caught.value.diagnostic
    assert diagnostic["correctionId"] == "phantom"
    assert diagnostic["status"] == "unresolved"
    assert diagnostic["reason"] == "insufficient-observation-overlap"
    assert diagnostic["candidates"] == [{"rawTrackId": 7}]


def test_observation_scoped_exclusion_does_not_remove_complete_raw_track():
    track = TrackState(
        id=1,
        points=[
            {"t": 0.0, "px": 100.0, "py": 200.0, "pitchX": 0.0, "pitchZ": 2.0, "confidence": 0.9}
        ],
        feature_sum=np.zeros(12, dtype=np.float32),
        feature_count=1,
        last_frame=0,
        last_height=40.0,
    )
    scene = {
        "payload": {
            "tracks": [
                {
                    "id": "auto-home-01",
                    "keyframes": [
                        {"t": 0.0, "x": 0.0, "z": 2.0, "confidence": 0.9, "observed": True}
                    ],
                }
            ],
            "videoAsset": {
                "reconstruction": {
                    "frameAnnotations": [
                        {
                            "id": "one-frame",
                            "kind": "ignore",
                            "action": "exclude",
                            "scope": "observation",
                            "sourceTrackId": "auto-home-01",
                        }
                    ]
                }
            },
        }
    }

    assert _apply_track_identity_corrections([track], scene) == [track]
