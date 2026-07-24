from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.pitch_calibration_contract import CalibrationAlignmentMetrics, PitchCalibration
from app.reconstruction import (
    _calibrate_scene_in_memory,
    _reconstruct_scene_in_memory,
)
from app.reconstruction_calibration_edit_command import (
    build_manual_calibration_override,
)
from app.reconstruction_calibration_snapshot import calibration_artifact_input
from app.reconstruction_calibration_overrides import (
    manual_pitch_calibration_overrides as _manual_pitch_calibration_overrides,
)
from app.reconstruction_calibration_proposal import propose_scene_pitch_calibration
from app.reconstruction_calibration_resolution import (
    resolve_temporal_frame_calibrations as _resolve_temporal_frame_calibrations,
)
from app.reconstruction_errors import ReconstructionError
from app.artifact_store import FilesystemArtifactStore
from app.reconstruction_artifact_hydration import hydrate_scene_reconstruction


def _scene() -> dict:
    return {
        "id": "shot-calibrate-frame",
        "title": "Calibration shot",
        "duration": 2.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "teams": [],
            "tracks": [],
            "ball": {"keyframes": []},
            "videoAsset": {
                "id": "asset-1",
                "fps": 10.0,
                "analysisFps": 10.0,
                "sourceStart": 10.0,
                "selectedSegmentId": "segment-1",
                "reconstruction": {"status": "ready", "qualityVerdict": "review"},
            },
        },
    }


def _alignment() -> CalibrationAlignmentMetrics:
    return CalibrationAlignmentMetrics(
        precision=0.82,
        recall=0.71,
        f1=0.76,
        residual_p50=2.0,
        residual_p95=5.5,
        model_sample_count=120,
        observed_sample_count=100,
        tolerance_pixels=6.0,
    )


def _keypoint_calibration(source_frame_index: int = 123) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.asarray(
            [[0.1, 0.0, -48.0], [0.0, 0.12, -30.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        confidence=0.94,
        supported_lines=10,
        mean_line_score=0.86,
        rectangle="field-keypoints-right",
        method="pnlcalib-points-lines",
        keypoint_count=8,
        detected_keypoint_count=8,
        inlier_count=7,
        inlier_ratio=0.875,
        reprojection_error=1.5,
        reprojection_p95=3.0,
        frame_index=source_frame_index,
        raw_keypoints=(
            {
                "id": "kp-1",
                "image": {"x": 480.0, "y": 300.0},
                "pitch": {"x": 0.0, "z": 0.0},
                "inlier": True,
            },
        ),
        raw_lines=(
            {
                "id": 17,
                "name": "Side line top",
                "start": {"x": 100.0, "y": 120.0},
                "end": {"x": 800.0, "y": 140.0},
                "confidence": 0.91,
                "groundPlane": True,
            },
        ),
    )


def _patch_frame(monkeypatch, image: np.ndarray, frame_index: int = 123) -> Path:
    path = Path(f"/tmp/frame_{frame_index:05d}.jpg")
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.frame_paths",
        lambda _: [(path, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.sampled_frame_context",
        lambda *_: (0, 0.0, image),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_draft.calibration_alignment_metrics",
        lambda *_: _alignment(),
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_calibration_quality.calibration_alignment_metrics",
        lambda *_: _alignment(),
    )
    return path


def test_auto_calibration_is_ephemeral_and_never_bumps_the_scene_revision(
    monkeypatch,
):
    scene = _scene()
    before = deepcopy(scene)
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    calls = []

    def automatic(frames, on_progress=None, *, worker_timeout=None):
        calls.append((frames, worker_timeout))
        return {123: _keypoint_calibration()}, []

    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        automatic,
    )
    monkeypatch.setattr(
        "app.scene_repository.scenes.put",
        lambda value: (_ for _ in ()).throw(
            AssertionError("a calibration preview must not persist the scene")
        ),
    )

    draft = propose_scene_pitch_calibration(scene, 0.04)

    assert calls[0][0][0][0].name == "frame_00123.jpg"
    assert calls[0][1] == 60.0
    assert draft["source"] == "frame-evidence"
    assert draft["status"] == "accepted"
    assert draft["solutionStatus"] == "direct-accepted"
    assert draft["sourceFrameIndex"] == 123
    assert draft["backend"] == "pnlcalib-points-lines"
    assert draft["rejectionReasons"] == []
    assert draft["evidence"]["keypoints"][0]["residualVector"] is not None
    assert draft["rawLines"][0]["name"] == "Side line top"
    assert draft["rawLines"][0]["residualStatus"] == "scored"
    assert draft["evidence"]["rawLines"] == draft["rawLines"]
    assert draft["evidence"]["markings"] == draft["markings"]
    # Preview → Cancel → Save regression: the proposal leaves the scene
    # document untouched, so a later save cannot 409 on a hidden revision.
    assert scene == before


def test_auto_calibration_without_pnl_solution_opens_explicit_manual_seed(monkeypatch):
    scene = _scene()
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        "app.reconstruction_pnlcalib_retry.recalibrate_frames_with_worker",
        lambda *_args, **_kwargs: {},
    )
    draft = propose_scene_pitch_calibration(scene, 0.0)

    assert draft["source"] == "manual-seed"
    assert draft["backend"] is None
    assert len(draft["attempts"]) == 3
    assert all(item["status"] == "missing" for item in draft["attempts"])
    assert any("No automatic fallback was used" in warning for warning in draft["warnings"])


def test_rejected_pnl_candidate_gets_two_fresh_attempts_without_fallback(monkeypatch):
    scene = _scene()
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    pnl = PitchCalibration(
        **{
            **_keypoint_calibration().__dict__,
            "confidence": 0.70,
            "method": "pnlcalib-points-lines",
        }
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({123: pnl}, []),
    )
    monkeypatch.setattr(
        "app.reconstruction_pnlcalib_retry.recalibrate_frames_with_worker",
        lambda *_args, **_kwargs: ({123: pnl}),
    )

    draft = propose_scene_pitch_calibration(scene, 0.0)

    assert draft["status"] == "rejected"
    assert draft["backend"] == "pnlcalib-points-lines"
    assert [item["backend"] for item in draft["attempts"]] == [
        "pnlcalib-points-lines",
        "pnlcalib-points-lines",
        "pnlcalib-points-lines",
    ]


def test_rejected_pnl_candidate_can_pass_on_a_fresh_retry(monkeypatch):
    scene = _scene()
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    rejected = PitchCalibration(
        **{**_keypoint_calibration().__dict__, "confidence": 0.70}
    )
    accepted = _keypoint_calibration()
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({123: rejected}, []),
    )
    monkeypatch.setattr(
        "app.reconstruction_pnlcalib_retry.recalibrate_frames_with_worker",
        lambda *_args, **_kwargs: ({123: accepted}),
    )

    draft = propose_scene_pitch_calibration(scene, 0.0)

    assert draft["status"] == "accepted"
    assert len(draft["attempts"]) == 2
    assert draft["attempts"][1]["selected"] is True
    assert any("attempt 2/3" in warning for warning in draft["warnings"])


def test_apply_calibration_upserts_authoritative_manual_frame_collection(monkeypatch):
    scene = _scene()
    old = {
        "id": "manual-frame-101",
        "method": "manual-pitch-anchors",
        "sceneTime": 0.0,
        "sampleIndex": 0,
        "sourceFrameIndex": 101,
        "imageToPitch": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["pitchCalibrationOverrides"] = [old]
    anchors = [
        {
            "id": str(index),
            "label": str(index),
            "image": {"x": float(index), "y": float(index)},
            "pitch": {"x": float(index), "z": float(index)},
        }
        for index in range(4)
    ]
    draft = {
        "sceneTime": 0.2,
        "frameIndex": 2,
        "confidence": 0.91,
        "alignmentError": 3.2,
        "alignmentMetrics": _alignment().as_dict(),
        "horizon": None,
        "quality": "good",
        "preset": "center-circle",
        "anchors": anchors,
        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
    }
    frames = [
        (Path("/tmp/frame_00101.jpg"), 0.0),
        (Path("/tmp/frame_00103.jpg"), 0.2),
        (Path("/tmp/frame_00105.jpg"), 0.4),
    ]
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.preview_scene_pitch_calibration",
        lambda *_: draft,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.frame_paths", lambda _: frames
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.calibration_frame_context",
        lambda *_: (1, 0.2, np.zeros((540, 960, 3), dtype=np.uint8), np.eye(3)),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.scenes.put",
        lambda value: value,
    )
    build_manual_calibration_override(scene, 0.2, "center-circle", anchors)
    first_new = reconstruction["pitchCalibrationOverrides"][-1]

    assert [item["sourceFrameIndex"] for item in reconstruction["pitchCalibrationOverrides"]] == [
        101,
        103,
    ]
    assert first_new["sourceFrameIndex"] == 103
    assert first_new["coordinateSpace"] == "stabilized-reference-image"
    assert len(_manual_pitch_calibration_overrides(reconstruction)) == 2

    draft["sceneTime"] = 0.4
    draft["frameIndex"] = 3
    build_manual_calibration_override(scene, 0.4, "center-circle", anchors)

    assert [item["sourceFrameIndex"] for item in reconstruction["pitchCalibrationOverrides"]] == [
        101,
        103,
        105,
    ]

    draft["confidence"] = 0.96
    build_manual_calibration_override(scene, 0.4, "center-circle", anchors)

    assert len(reconstruction["pitchCalibrationOverrides"]) == 3
    assert reconstruction["pitchCalibrationOverrides"][-1]["confidence"] == 0.96


def test_build_manual_correction_does_not_queue_or_change_stage(monkeypatch):
    anchors = [
        {
            "id": str(index),
            "label": str(index),
            "image": {"x": float(index), "y": float(index)},
            "pitch": {"x": float(index), "z": float(index)},
        }
        for index in range(4)
    ]
    draft = {
        "sceneTime": 0.2,
        "frameIndex": 2,
        "confidence": 0.91,
        "alignmentError": 3.2,
        "alignmentMetrics": _alignment().as_dict(),
        "horizon": None,
        "quality": "good",
        "preset": "center-circle",
        "anchors": anchors,
        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
    }
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.preview_scene_pitch_calibration",
        lambda *_: draft,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.frame_paths",
        lambda _: [(Path("/tmp/frame_00103.jpg"), 0.2)],
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.calibration_frame_context",
        lambda *_: (1, 0.2, np.zeros((540, 960, 3), dtype=np.uint8), np.eye(3)),
    )
    gated = _scene()
    gated["payload"]["videoAsset"]["reconstruction"]["stage"] = "calibration"
    original_run = gated["payload"]["videoAsset"]["reconstruction"].get("runId")
    build_manual_calibration_override(gated, 0.2, "center-circle", anchors)

    reconstruction = gated["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["stage"] == "calibration"
    assert reconstruction.get("runId") == original_run
    assert len(reconstruction["pitchCalibrationOverrides"]) == 1


def test_poor_manual_alignment_requires_explicit_operator_consent(monkeypatch):
    anchors = [
        {
            "id": str(index),
            "label": str(index),
            "image": {"x": float(index), "y": float(index)},
            "pitch": {"x": float(index), "z": float(index)},
        }
        for index in range(4)
    ]
    draft = {
        "sceneTime": 0.2,
        "frameIndex": 1,
        "confidence": 0.55,
        "alignmentError": 12.0,
        "alignmentMetrics": {
            **_alignment().as_dict(),
            "recall": 0.15,
            "f1": 0.25,
        },
        "horizon": None,
        "quality": "poor",
        "preset": "center-circle",
        "anchors": anchors,
        "imageToPitch": [
            [0.1, 0.0, -48.0],
            [0.0, 0.1, -27.0],
            [0.0, 0.0, 1.0],
        ],
    }
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.preview_scene_pitch_calibration",
        lambda *_: draft,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_edit_command.frame_paths",
        lambda _: [(Path("/tmp/frame_00103.jpg"), 0.2)],
    )
    scene = _scene()

    with pytest.raises(ReconstructionError, match="explicitly save it"):
        build_manual_calibration_override(
            scene,
            0.2,
            "center-circle",
            anchors,
            camera_transform=np.eye(3),
        )

    override, _ = build_manual_calibration_override(
        scene,
        0.2,
        "center-circle",
        anchors,
        camera_transform=np.eye(3),
        accept_quality_warning=True,
    )

    assert override["status"] == "review"
    assert override["validationStatus"] == "poor"
    assert override["qualityWarningAccepted"] is True


def test_full_reconstruction_never_runs_automatic_calibration(monkeypatch):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["pitchCalibrationOverrides"] = [{
        "method": "manual-pitch-anchors",
        "sceneTime": 0.0,
        "sourceFrameIndex": 123,
        "confidence": 0.95,
        "supportedLines": 4,
        "preset": "center-circle",
        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
    }]
    frame = Path("/tmp/frame_00123.jpg")
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_scene_track_publisher.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    automatic_calls = []

    def automatic(frames, on_progress=None, *, worker_timeout=None):
        automatic_calls.append(frames)
        return {}, []

    monkeypatch.setattr(
        "app.reconstruction_sampled_calibration.automatic_frame_calibrations",
        automatic,
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.load_model",
        lambda *_: pytest.fail("detector must not load without calibration input"),
    )
    with pytest.raises(ReconstructionError, match="Calibration output is missing"):
        _reconstruct_scene_in_memory(scene)

    assert automatic_calls == []


def test_full_process_entry_rejects_a_calibration_job_before_work_starts(
    monkeypatch,
):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["mode"] = "calibrate"
    monkeypatch.setattr(
        "app.reconstruction._calibrate_only_phase",
        lambda *_args, **_kwargs: pytest.fail(
            "the full process must not reach calibration"
        ),
    )

    with pytest.raises(
        ReconstructionError,
        match="full process cannot execute a calibrate job",
    ):
        _reconstruct_scene_in_memory(scene)


def test_skip_profile_run_never_calls_the_ball_detector_and_keeps_manual_ball(
    monkeypatch, tmp_path
):
    scene = _scene()
    manual_keyframes = [{"t": 0.5, "x": 1.0, "z": 2.0}]
    previous_automatic = [{"t": 0.1, "x": 9.0, "z": 9.0}]
    scene["payload"]["ball"] = {
        "mode": "manual",
        "manualKeyframes": deepcopy(manual_keyframes),
        "automaticKeyframes": deepcopy(previous_automatic),
        "keyframes": deepcopy(manual_keyframes),
    }
    # The queue command pins every fingerprinted input before computing the
    # fence; the in-memory pipeline must then re-write identical values only.
    scene["payload"]["videoAsset"]["reconstruction"].update(
        {
            "ballDetectionProfile": "skip-manual-authoritative",
            "jerseyOcrProfile": "automatic",
            "model": "yolo26m.pt",
            "ballBackend": "generic-ultralytics",
            "ballDetectionInput": {
                "schemaVersion": 1,
                "backend": "generic-ultralytics",
                "analysisFrameRate": 25.0,
                "failurePolicy": "fallback",
            },
        }
    )
    frame = Path("/tmp/frame_00123.jpg")
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_scene_track_publisher.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_calibration.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({123: _keypoint_calibration()}, []),
    )
    monkeypatch.setattr("app.reconstruction_calibration_only_phase.cv2.imread", lambda *_: image)
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.load_model",
        lambda *_: object(),
    )
    person_prediction = SimpleNamespace(
        image_bgr=image,
        names={},
        diagnostics={},
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.build_person_detection_provider",
        lambda *_: SimpleNamespace(
            predict=lambda _path: person_prediction,
            info=lambda: {},
        ),
    )
    monkeypatch.setattr(
        "app.person_base_detection_cache.parse_person_prediction",
        lambda *_: ([], []),
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_calibration_quality.calibration_alignment_metrics",
        lambda *_: _alignment(),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.calibration_alignment_metrics_from_mask",
        lambda *_: _alignment(),
    )

    class ForbiddenBallDetector:
        backend_name = "dedicated-ultralytics"

        def detect(self, *_args, **_kwargs):
            raise AssertionError("ball detector must not run under the skip profile")

    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.configured_ball_detectors",
        lambda *_args, **_kwargs: (ForbiddenBallDetector(), None),
    )

    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    from app.reconstruction_run_log import ReconstructionRunLog
    from app.scene_document import reconstruction_input_fingerprint

    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["mode"] = "calibrate"
    reconstruction["runId"] = "calibration-run-a"
    _calibrate_scene_in_memory(scene, artifact_store=artifact_store)
    hydrate_scene_reconstruction(scene, store=artifact_store)
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["calibrationArtifactInput"] = calibration_artifact_input(
        reconstruction
    )
    reconstruction["mode"] = "full"
    reconstruction["stage"] = "reconstruction"
    reconstruction["trackingCoordinatePolicy"] = "metric-required"
    monkeypatch.setattr(
        "app.reconstruction_sampled_calibration.automatic_frame_calibrations",
        lambda *_args, **_kwargs: pytest.fail(
            "full reconstruction must not call calibration inference"
        ),
    )
    fingerprint_before = reconstruction_input_fingerprint(scene)
    run_log = ReconstructionRunLog(
        tmp_path / "analysis-runs", scene_id="shot-calibrate-frame", run_id="run-a"
    )
    rebuilt = _reconstruct_scene_in_memory(
        scene, artifact_store=artifact_store, run_log=run_log
    )
    run_log.close("ready")
    # Terminal publication accepts a result only while the fingerprint stays
    # byte-identical: a pipeline mutation of any fingerprinted input would
    # silently make every finished run unpublishable.
    assert reconstruction_input_fingerprint(rebuilt) == fingerprint_before
    import json as _json

    journal = [
        _json.loads(line)
        for line in run_log.path.read_text(encoding="utf-8").splitlines()
    ]
    finished_phases = [
        item["phase"] for item in journal if item["event"] == "phase-finished"
    ]
    assert finished_phases == [
        "detection",
        "identity",
        "ball",
        "publish",
    ]
    run_inputs = next(item for item in journal if item["event"] == "run-inputs")
    assert run_inputs["reconstructionMode"] == "full"
    assert run_inputs["calibrationAuthority"] == "consumer-only"
    assert run_inputs["calibrationComputation"] == "forbidden"
    assert run_inputs["calibrationArtifactInput"]["producerRunId"] == (
        reconstruction["calibrationProvenance"]["runId"]
    )
    used = next(
        item for item in journal if item["event"] == "calibration-input-used"
    )
    assert used["dataFingerprint"] == reconstruction[
        "calibrationProvenance"
    ]["dataFingerprint"]
    assert used["artifactSha256"]
    assert used["usedFor"] == [
        "person-metric-projection",
        "tracking-association",
        "ball-world-projection",
        "3d-trajectory-publication",
    ]
    assert used["sampleUsage"][0]["sourceFrameIndex"] == 123
    assert used["sampleUsage"][0]["imageToPitchFingerprint"].startswith(
        "sha256:"
    )
    assert used["sampleUsage"][0]["cameraTransformFingerprint"].startswith(
        "sha256:"
    )
    assert any(
        item["event"] == "calibration-tracking-impact" for item in journal
    )
    assert any(item["event"] == "calibration-ball-impact" for item in journal)
    assert any(item["event"] == "progress" for item in journal)

    reconstruction = rebuilt["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "ready"
    assert reconstruction["ballDetectionProfile"] == "skip-manual-authoritative"
    assert reconstruction["ballDetection"]["status"] == "skipped"
    assert reconstruction["diagnostics"]["ballTracking"]["skippedByProfile"] is True
    assert reconstruction["diagnostics"]["calibrationUsage"][
        "dataFingerprint"
    ] == used["dataFingerprint"]
    hydrate_scene_reconstruction(
        rebuilt,
        names=("ballTrajectory",),
        store=artifact_store,
    )
    ball = rebuilt["payload"]["ball"]
    assert ball["mode"] == "manual"
    assert [
        {key: item[key] for key in ("t", "x", "z")} for item in ball["keyframes"]
    ] == manual_keyframes
    # Publishing a newer calibration invalidates the previous automatic
    # trajectory. The skip profile preserves the manual source of truth, but
    # must not resurrect automatic samples computed from an older camera.
    assert ball["automaticKeyframes"] == []
    assert any(
        "skipped by the analysis profile" in warning
        for warning in reconstruction["warnings"]
    )


def test_calibration_process_uses_operator_override_on_the_same_sample(
    monkeypatch, tmp_path
):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["pitchCalibrationOverrides"] = [{
        "id": "manual-frame-123",
        "method": "manual-pitch-anchors",
        "sceneTime": 0.0,
        "sampleIndex": 0,
        "sourceFrameIndex": 123,
        # This is an explicit operator correction for this exact frame.
        "confidence": 0.25,
        "supportedLines": 4,
        "preset": "center-circle",
        "alignmentError": 2.0,
        "imageToPitch": [[0.1, 0.0, -48.0], [0.0, 0.1, -27.0], [0.0, 0.0, 1.0]],
    }]
    frame = Path("/tmp/frame_00123.jpg")
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    automatic = _keypoint_calibration()
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_scene_track_publisher.frame_paths",
        lambda _: [(frame, 0.0)],
    )
    monkeypatch.setattr(
        "app.reconstruction_sampled_calibration.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({123: automatic}, []),
    )
    monkeypatch.setattr("app.reconstruction_calibration_only_phase.cv2.imread", lambda *_: image)
    monkeypatch.setattr(
        "app.reconstruction_frame_calibration_quality.calibration_alignment_metrics",
        lambda *_: _alignment(),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.calibration_alignment_metrics_from_mask",
        lambda *_: _alignment(),
    )
    captured_direct = {}
    real_resolver = _resolve_temporal_frame_calibrations

    def capture_resolver(
        frames,
        frame_sizes,
        direct_calibrations,
        motion_edges,
        frame_evidence,
        person_frames,
        pitch,
        **kwargs,
    ):
        captured_direct.update(direct_calibrations)
        return real_resolver(
            frames,
            frame_sizes,
            direct_calibrations,
            motion_edges,
            frame_evidence,
            person_frames,
            pitch,
            **kwargs,
        )

    monkeypatch.setattr(
        "app.reconstruction_temporal_calibration_phase.resolve_temporal_frame_calibrations",
        capture_resolver,
    )

    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    scene["payload"]["videoAsset"]["reconstruction"]["mode"] = "calibrate"
    rebuilt = _calibrate_scene_in_memory(
        scene,
        artifact_store=artifact_store,
    )

    assert captured_direct[0].method == "manual-pitch-anchors"
    reconstruction = rebuilt["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["artifactManifest"]["schemaVersion"] == 1
    assert "calibrationFrames" not in reconstruction
    assert "identityResolver" not in (reconstruction.get("diagnostics") or {})
    hydrate_scene_reconstruction(
        rebuilt,
        names=("calibrationFrames",),
        store=artifact_store,
    )
    evidence = reconstruction["calibration"]["frameEvidence"][0]
    assert evidence["status"] == "accepted"
    assert evidence["projectionSource"] == "manual-direct"
    assert evidence["backend"] == "manual-pitch-anchors"
    assert evidence["automaticObservation"]["status"] == "accepted"
    # The automatic candidate stays inspectable, but the saved operator edit
    # is the only selected observation for this frame.
    assert evidence["manualObservation"]["status"] == "accepted"
    assert [item["kind"] for item in evidence["observations"]] == [
        "manual",
        "automatic",
    ]
    assert "directAttempts" not in evidence
    assert "confidence-below-metric-threshold" not in evidence["manualObservation"][
        "rejectionReasons"
    ]
    assert evidence["observationChoice"] == {
        "selectedKind": "manual",
        "reason": "operator-frame-override",
        "automaticCandidateStatus": "accepted",
    }
    assert reconstruction["calibration"]["manualFrameAnchors"] == [
        {
            "id": "manual-frame-123",
            "sampleIndex": 0,
            "sourceFrameIndex": 123,
            "sceneTime": 0.0,
            "status": "accepted",
        }
    ]
