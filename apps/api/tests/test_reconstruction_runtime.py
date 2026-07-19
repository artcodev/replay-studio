from copy import deepcopy

import cv2
import numpy as np
import pytest

from app.pitch_anchor_calibration import calibration_from_anchors
from app.pitch_calibration_contract import PitchCalibration
from app.pitch_calibration_orientation import canonicalize_penalty_side
from app.pitch_geometry import projected_pitch_markings
from app.reconstruction_calibration_apply import apply_scene_pitch_calibration
from app.reconstruction_pitch_side_command import set_scene_pitch_side
from app.reconstruction_errors import IdentityCorrectionError, ReconstructionError
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete as delete_frame_person_annotation,
    draft_frame_person_annotation_upsert as upsert_frame_person_annotation,
)
from app.reconstruction_queue import queue_reconstruction
from app.reconstruction_queue_draft import (
    ReconstructionQueueInputs,
    prepare_reconstruction_queue_draft,
)
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


def _track(track_id: int, x: float, samples: int = 10) -> TrackState:
    feature = np.zeros(12, dtype=np.float32)
    return TrackState(
        id=track_id,
        points=[{"t": index * 0.2, "px": x, "py": 220.0, "confidence": 0.8} for index in range(samples)],
        feature_sum=feature,
        feature_count=1,
        last_frame=samples - 1,
        last_height=40.0,
    )


def test_referee_voted_track_is_mapped_to_officials_instead_of_dropped():
    from app.reconstruction_team_classification import (
        include_reid_official_candidates,
    )

    referee = _track(5, 480)
    referee.role = "referee"
    already_teamed = _track(1, 650)
    already_teamed.role = "referee"
    manual = _track(6, 500)
    manual.role = "referee"
    manual.manual_kind = "confirm-referee"
    unvoted = _track(7, 520)
    mapping = {1: "home"}

    officials = include_reid_official_candidates(
        [referee, already_teamed, manual, unvoted],
        mapping,
    )

    # Only the unteamed, unmanaged referee-voted track joins officials.
    assert mapping[5] == "officials"
    assert mapping[1] == "home"
    assert 6 not in mapping and 7 not in mapping
    assert officials == [referee.local_tracklet_id]


def test_goalkeeper_candidate_survives_third_kit_cluster():
    tracks = [
        _track(1, 650),
        _track(2, 700),
        _track(3, 760),
        _track(4, 620),
        _track(5, 900),
        _track(6, 480),
    ]
    mapping = {1: "home", 2: "away", 3: "away", 4: "home"}

    result = _include_goalkeeper_candidates(tracks, mapping, 960)

    assert result[5] == "away"
    assert tracks[4].role == "goalkeeper"
    assert 6 not in result


def test_frame_analysis_interpolates_track_at_requested_time():
    keyframes = [
        {"t": 0.0, "x": 10.0, "z": -4.0, "confidence": 0.8},
        {"t": 1.0, "x": 20.0, "z": 6.0, "confidence": 0.6},
    ]

    position = _interpolate_scene_keyframes(keyframes, 0.25)

    assert position is not None
    assert position["x"] == 12.5
    assert position["z"] == -1.5
    assert position["confidence"] == 0.75


def _queue_draft(
    scene: dict,
    *,
    model: str = "yolo26m.pt",
    ball_backend: str = "generic-ultralytics",
    ball_input: dict | None = None,
    frame_count: int = 0,
    run_id: str = "test-run",
    ball_detection_profile: str = "automatic",
) -> dict:
    return prepare_reconstruction_queue_draft(
        scene,
        ReconstructionQueueInputs(
            model=model,
            ball_backend=ball_backend,
            ball_detection_input=ball_input
            or {"schemaVersion": 1, "backend": ball_backend},
            frame_count=frame_count,
            run_id=run_id,
            match_snapshot_ref=None,
            ball_detection_profile=ball_detection_profile,
        ),
    )


def _profile_scene(ball_mode: str = "manual") -> dict:
    return {
        "id": "shot-profile",
        "duration": 4.0,
        "payload": {
            "videoAsset": {"id": "asset-1", "analysisFps": 10},
            "ball": {
                "mode": ball_mode,
                "manualKeyframes": [{"t": 0.0, "x": 1.0, "z": 2.0}],
                "automaticKeyframes": [],
                "keyframes": [],
            },
        },
    }


def test_skip_ball_profile_is_pinned_and_changes_the_input_fingerprint():
    automatic = _queue_draft(_profile_scene(), run_id="run-a")
    skipped = _queue_draft(
        _profile_scene(),
        run_id="run-a",
        ball_detection_profile="skip-manual-authoritative",
    )

    reconstruction = skipped["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["ballDetectionProfile"] == "skip-manual-authoritative"
    assert (
        automatic["payload"]["videoAsset"]["reconstruction"]["ballDetectionProfile"]
        == "automatic"
    )
    assert (
        reconstruction["inputFingerprint"]
        != automatic["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    )


def test_default_profile_keeps_pre_profile_fingerprints_stable():
    # Scenes queued before profiles existed must not become stale: the
    # automatic default is omitted from the digest entirely.
    scene = _profile_scene()
    baseline = _queue_draft(scene, run_id="run-a")
    reconstruction = baseline["payload"]["videoAsset"]["reconstruction"]
    with_field = reconstruction_input_fingerprint(baseline)
    reconstruction.pop("ballDetectionProfile")
    without_field = reconstruction_input_fingerprint(baseline)

    assert with_field == without_field


def test_skip_ball_profile_requires_the_manual_trajectory_to_be_authoritative():
    with pytest.raises(ReconstructionError, match="manual ball"):
        _queue_draft(
            _profile_scene(ball_mode="automatic"),
            ball_detection_profile="skip-manual-authoritative",
        )
    with pytest.raises(ReconstructionError, match="Unknown ball detection profile"):
        _queue_draft(_profile_scene(), ball_detection_profile="turbo")


def test_jersey_ocr_profile_is_pinned_validated_and_fenced(monkeypatch):
    draft = prepare_reconstruction_queue_draft(
        _profile_scene(),
        ReconstructionQueueInputs(
            model="yolo26m.pt",
            ball_backend="generic-ultralytics",
            ball_detection_input={"schemaVersion": 1},
            frame_count=0,
            run_id="run-a",
            match_snapshot_ref=None,
            jersey_ocr_profile="off",
        ),
    )
    reconstruction = draft["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["jerseyOcrProfile"] == "off"

    baseline = _queue_draft(_profile_scene(), run_id="run-a")
    assert (
        reconstruction["inputFingerprint"]
        != baseline["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"]
    )
    # The automatic default stays out of the digest: pre-profile scenes keep
    # their fingerprints.
    baseline_reconstruction = baseline["payload"]["videoAsset"]["reconstruction"]
    with_field = reconstruction_input_fingerprint(baseline)
    baseline_reconstruction.pop("jerseyOcrProfile")
    assert reconstruction_input_fingerprint(baseline) == with_field

    with pytest.raises(ReconstructionError, match="Unknown jersey OCR profile"):
        prepare_reconstruction_queue_draft(
            _profile_scene(),
            ReconstructionQueueInputs(
                model="yolo26m.pt",
                ball_backend="generic-ultralytics",
                ball_detection_input={"schemaVersion": 1},
                frame_count=0,
                run_id="run-a",
                match_snapshot_ref=None,
                jersey_ocr_profile="disabled",
            ),
        )


def test_progress_write_throttle_coalesces_quiet_same_phase_ticks():
    from app.reconstruction_progress import ProgressWriteThrottle

    clock = {"now": 0.0}
    throttle = ProgressWriteThrottle(1.0, clock=lambda: clock["now"])

    assert throttle.should_write({"phase": "detection"}) is True
    clock["now"] = 0.2
    assert throttle.should_write({"phase": "detection"}) is False
    clock["now"] = 0.4
    # A phase transition always writes, even inside the interval.
    assert throttle.should_write({"phase": "tracking"}) is True
    clock["now"] = 0.5
    assert throttle.should_write({"phase": "tracking"}) is False
    clock["now"] = 1.5
    assert throttle.should_write({"phase": "tracking"}) is True
    clock["now"] = 1.6
    # Terminal ticks always write.
    assert throttle.should_write({"phase": "complete"}) is True
    clock["now"] = 1.7
    assert throttle.should_write({"phase": "failed"}) is True


def test_identity_phase_skips_jersey_ocr_when_profile_is_off(monkeypatch):
    from types import SimpleNamespace

    from app.reconstruction_identity_phase import track_and_resolve_identity_phase

    monkeypatch.setattr(
        "app.reconstruction_identity_phase.run_jersey_ocr_for_tracklets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("jersey OCR must not run under the off profile")
        ),
    )
    scene = {
        "id": "identity-skip",
        "duration": 1.0,
        "payload": {
            "videoAsset": {"id": "asset-1", "reconstruction": {}},
            "teams": [],
            "tracks": [],
            "canonicalPeople": [],
            "ball": {"keyframes": []},
        },
    }
    progress = SimpleNamespace(update=lambda *args, **kwargs: None)

    result = track_and_resolve_identity_phase(
        scene,
        [],
        [],
        (1920, 1080),
        "unavailable",
        {},
        None,
        progress,
        None,
        {},
        [],
        jersey_ocr_profile="off",
    )

    assert result.jersey_ocr_diagnostics["status"] == "skipped-by-profile"
    assert result.jersey_ocr_diagnostics["skippedByProfile"] is True
    assert any(
        "skipped by the analysis profile" in warning for warning in result.warnings
    )


def test_queue_draft_preserves_last_good_result_and_records_previous():
    scene = {
        "id": "shot-1",
        "duration": 4.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "status": "ready",
                    "completedAt": "yesterday",
                    "pitchCalibration": {"status": "ready"},
                    "pitchCalibrationOverrides": [
                        {
                            "method": "manual-pitch-anchors",
                            "imageToPitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        }
                    ],
                },
            },
            "tracks": [{"id": "old-track"}],
            "ball": {"keyframes": [{"t": 1.0}]},
        },
    }
    queued = _queue_draft(scene, frame_count=2)

    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    assert queued is not scene
    assert scene["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    # Artifact publication is intentionally absent from the pure draft.
    assert queued["payload"]["tracks"] == [{"id": "old-track"}]
    assert queued["payload"]["ball"] == {"keyframes": [{"t": 1.0}]}
    assert reconstruction["status"] == "queued"
    assert reconstruction["frameCount"] == 2
    assert reconstruction["progress"]["phase"] == "preparing"
    assert reconstruction["progress"]["overallPercent"] == 0
    assert [item["status"] for item in reconstruction["progress"]["phases"]] == [
        "current",
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert reconstruction["previousResult"] == {
        "completedAt": "yesterday",
        "trackCount": 1,
        "ballSamples": 1,
        "calibrationStatus": "ready",
    }
    assert reconstruction["pitchCalibrationOverrides"][0]["method"] == "manual-pitch-anchors"


def test_reconstruction_progress_exposes_completed_current_and_pending_phases():
    scene = {
        "id": "progress-scene",
        "payload": {"videoAsset": {"reconstruction": {"status": "processing"}}},
    }
    payload = ReconstructionProgress(scene).update(
        "calibration",
        2,
        "Calibrating the pitch",
        "PnLCalib · 4/8 frames.",
        4,
        62,
        completed=4,
        total=8,
    )

    assert payload["phasePercent"] == 50
    assert payload["overallPercent"] == 33
    assert payload["completed"] == 4
    assert payload["total"] == 8
    assert [item["status"] for item in payload["phases"]] == [
        "completed",
        "current",
        "pending",
        "pending",
        "pending",
        "pending",
    ]


def test_queue_draft_uses_requested_model():
    scene = {
        "id": "shot-model",
        "duration": 2.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {"status": "ready", "model": "yolo26n.pt"},
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    queued = _queue_draft(scene, model="yolo26m.pt", frame_count=1)

    assert queued["payload"]["videoAsset"]["reconstruction"]["model"] == "yolo26m.pt"


def test_queue_draft_records_requested_ball_backend_and_input():
    scene = {
        "id": "shot-ball-backend",
        "duration": 2.0,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {
                    "status": "ready",
                    "ballBackend": "generic-ultralytics",
                    "ballDetectionInput": {
                        "backend": "generic-ultralytics",
                        "analysisFrameRate": 10.0,
                    },
                },
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    requested_input = {
        "schemaVersion": 1,
        "backend": "wasb-service",
        "checkpoint": {"name": "wasb-soccer-best.pth.tar", "size": 1234},
        "analysisFrameRate": 25.0,
        "failurePolicy": "fallback",
    }
    queued = _queue_draft(
        scene,
        model="yolo26m.pt",
        ball_backend="wasb-service",
        ball_input=requested_input,
        frame_count=1,
    )
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]

    assert reconstruction["ballBackend"] == "wasb-service"
    assert reconstruction["ballDetectionInput"] == requested_input
    assert reconstruction["inputFingerprint"] == reconstruction_input_fingerprint(queued)


def test_queue_command_publishes_artifacts_before_atomic_enqueue(monkeypatch):
    scene = {
        "id": "queue-command",
        "duration": 2.0,
        "revision": 4,
        "payload": {
            "videoAsset": {
                "id": "asset-queue-command",
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "ready", "model": "yolo26m.pt"},
            },
            "tracks": [],
            "ball": {"keyframes": []},
        },
    }
    expected_fingerprint = reconstruction_input_fingerprint(scene)
    events: list[str] = []

    monkeypatch.setattr("app.reconstruction_queue.frame_paths", lambda _: [])
    monkeypatch.setattr(
        "app.reconstruction_queue.ball_detection_input",
        lambda backend: {"schemaVersion": 1, "backend": backend},
    )
    monkeypatch.setattr(
        "app.reconstruction_queue.hydrate_scene_reconstruction",
        lambda _scene: events.append("hydrate"),
    )

    def publish(working_scene: dict) -> None:
        events.append("publish")
        reconstruction = working_scene["payload"]["videoAsset"]["reconstruction"]
        reconstruction["artifactManifest"] = {"schemaVersion": 1, "artifacts": {}}

    monkeypatch.setattr(
        "app.reconstruction_queue.publish_dense_reconstruction_artifacts",
        publish,
    )

    class Runs:
        @staticmethod
        def enqueue_reconstruction(
            queued_scene: dict,
            *,
            expected_input_fingerprint: str,
        ) -> dict:
            events.append("enqueue")
            assert expected_input_fingerprint == expected_fingerprint
            reconstruction = queued_scene["payload"]["videoAsset"][
                "reconstruction"
            ]
            assert reconstruction["artifactManifest"] == {
                "schemaVersion": 1,
                "artifacts": {},
            }
            return queued_scene

    monkeypatch.setattr("app.reconstruction_queue.reconstruction_runs", Runs())

    queued = queue_reconstruction(scene, match_snapshot=None)

    assert events == ["hydrate", "publish", "enqueue"]
    assert scene["payload"]["videoAsset"]["reconstruction"]["status"] == "ready"
    assert queued["payload"]["videoAsset"]["reconstruction"]["status"] == "queued"


def test_ball_backend_and_detector_config_are_reconstruction_fingerprint_inputs():
    scene = {
        "id": "shot-ball-fingerprint",
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "sourceStart": 1.0,
                "sourceEnd": 3.0,
                "reconstruction": {
                    "model": "yolo26m.pt",
                    "ballBackend": "dedicated-ultralytics",
                    "ballDetectionInput": {
                        "schemaVersion": 1,
                        "backend": "dedicated-ultralytics",
                        "checkpoint": {"name": "football-ball-detection.pt", "size": 1234},
                        "analysisFrameRate": 25.0,
                        "inferenceBatchSize": 8,
                    },
                    # Worker output is deliberately not an immutable input.
                    "ballDetection": {"candidateCount": 10},
                },
            }
        },
    }
    baseline = reconstruction_input_fingerprint(scene)

    changed_backend = deepcopy(scene)
    changed_backend["payload"]["videoAsset"]["reconstruction"][
        "ballBackend"
    ] = "generic-ultralytics"
    assert reconstruction_input_fingerprint(changed_backend) != baseline

    changed_config = deepcopy(scene)
    changed_config["payload"]["videoAsset"]["reconstruction"][
        "ballDetectionInput"
    ]["inferenceBatchSize"] = 4
    assert reconstruction_input_fingerprint(changed_config) != baseline

    changed_runtime_output = deepcopy(scene)
    changed_runtime_output["payload"]["videoAsset"]["reconstruction"][
        "ballDetection"
    ]["candidateCount"] = 999
    assert reconstruction_input_fingerprint(changed_runtime_output) == baseline
