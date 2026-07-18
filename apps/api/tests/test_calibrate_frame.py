from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.reconstruction_calibration_detection as calibration_detection
from app.pitch_calibration_contract import CalibrationAlignmentMetrics, PitchCalibration
from app.reconstruction import _reconstruct_scene_in_memory
from app.reconstruction_calibration_apply import apply_scene_pitch_calibration
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
        "app.reconstruction_calibration_proposal.calibration_frame_context",
        lambda *_: (0, 0.0, image, np.eye(3, dtype=np.float64)),
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


def test_auto_calibration_runs_on_exact_frame_and_persists_evidence_without_queue(
    monkeypatch,
):
    scene = _scene()
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
        "app.reconstruction_calibration_proposal.calibrate_pitch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("line fallback must not run after an accepted keypoint fit")
        ),
    )
    persisted = []
    monkeypatch.setattr(
        "app.reconstruction_calibration_preview.scenes.put",
        lambda value: persisted.append(value) or value,
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
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "ready"
    assert reconstruction["qualityVerdict"] == "review"
    assert reconstruction["calibration"]["lastFramePreview"]["sourceFrameIndex"] == 123
    assert persisted


def test_auto_calibration_line_fallback_is_downscaled_bounded_and_lifted(monkeypatch):
    scene = _scene()
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({}, []),
    )
    call = {}

    def line_fallback(input_image, **kwargs):
        call.update({"shape": input_image.shape, **kwargs})
        kwargs["diagnostics"].update(
            {
                "budgetExhausted": True,
                "deadlineExceeded": False,
                "candidateLimitReached": True,
            }
        )
        return PitchCalibration(
            image_to_pitch=np.eye(3, dtype=np.float64),
            confidence=0.9,
            supported_lines=4,
            mean_line_score=0.8,
            rectangle="penalty-area-right",
            matched_curves=1,
            method="pitch-lines-ransac",
        )

    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.calibrate_pitch", line_fallback
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_preview.scenes.put",
        lambda value: value,
    )

    draft = propose_scene_pitch_calibration(scene, 0.0)

    assert call["shape"][1] == 640
    assert call["max_quad_candidates"] == 240
    assert call["deadline"] is not None
    assert draft["backend"] == "pitch-lines-ransac"
    assert draft["status"] == "accepted"
    assert np.isclose(abs(draft["imageToPitch"][0][0]), 2 / 3)
    assert draft["evidence"]["backendDiagnostics"]["budgetExhausted"] is True
    assert any("candidate search limit" in warning for warning in draft["warnings"])
    assert not any("five-second deadline" in warning for warning in draft["warnings"])


def test_rejected_line_fallback_does_not_mask_semantic_keypoints(monkeypatch):
    scene = _scene()
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    _patch_frame(monkeypatch, image)
    semantic = PitchCalibration(
        **{
            **_keypoint_calibration().__dict__,
            "confidence": 0.70,
            "method": "roboflow-field-keypoints",
        }
    )
    line = PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.74,
        supported_lines=4,
        mean_line_score=0.9,
        rectangle="penalty-area-right",
        matched_curves=1,
        method="pitch-lines-ransac",
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.automatic_frame_calibrations",
        lambda *_args, **_kwargs: ({123: semantic}, []),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_proposal.calibrate_pitch",
        lambda *_args, **_kwargs: line,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_preview.scenes.put",
        lambda value: value,
    )

    draft = propose_scene_pitch_calibration(scene, 0.0)

    assert draft["status"] == "rejected"
    assert draft["backend"] == "roboflow-field-keypoints"
    assert [item["backend"] for item in draft["attempts"]] == [
        "roboflow-field-keypoints",
        "pitch-lines-ransac",
    ]


def test_local_keypoint_inference_uses_checkpoint_native_size(monkeypatch, tmp_path):
    model_path = tmp_path / "pitch.pt"
    model_path.write_bytes(b"checkpoint")
    frame_path = tmp_path / "frame_00123.jpg"
    frame_path.write_bytes(b"frame")
    calls = []

    class FakePoseModel:
        overrides = {"imgsz": 640}

        def predict(self, sources, **kwargs):
            calls.append((sources, kwargs))
            return [object()]

    settings = SimpleNamespace(
        pitch_keypoint_model=str(model_path),
        pitch_keypoint_image_size=None,
        reconstruction_device="cpu",
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_detection.get_settings", lambda: settings
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_detection.load_model", lambda *_: FakePoseModel()
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_detection.calibration_from_pose_result",
        lambda _prediction, source_index: _keypoint_calibration(source_index),
    )

    result = calibration_detection.local_frame_calibrations([(frame_path, 0.0)])

    assert result[123].frame_index == 123
    assert calls[0][1]["imgsz"] == 640


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
        "app.reconstruction_calibration_apply.preview_scene_pitch_calibration",
        lambda *_: draft,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.frame_paths", lambda _: frames
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.calibration_frame_context",
        lambda *_: (1, 0.2, np.zeros((540, 960, 3), dtype=np.uint8), np.eye(3)),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.scenes.put",
        lambda value: value,
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_apply.queue_reconstruction",
        lambda value, **_kwargs: value,
    )

    apply_scene_pitch_calibration(scene, 0.2, "center-circle", anchors)
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
    apply_scene_pitch_calibration(scene, 0.4, "center-circle", anchors)

    assert [item["sourceFrameIndex"] for item in reconstruction["pitchCalibrationOverrides"]] == [
        101,
        103,
        105,
    ]

    draft["confidence"] = 0.96
    apply_scene_pitch_calibration(scene, 0.4, "center-circle", anchors)

    assert len(reconstruction["pitchCalibrationOverrides"]) == 3
    assert reconstruction["pitchCalibrationOverrides"][-1]["confidence"] == 0.96


def test_rebuild_still_runs_automatic_calibration_when_manual_anchors_exist(monkeypatch):
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
        lambda *_: (_ for _ in ()).throw(ReconstructionError("stop after calibration")),
    )
    with pytest.raises(ReconstructionError, match="stop after calibration"):
        _reconstruct_scene_in_memory(scene)

    assert automatic_calls == [[(frame, 0.0)]]


def test_rebuild_uses_accepted_auto_when_same_sample_manual_is_rejected(
    monkeypatch, tmp_path
):
    scene = _scene()
    scene["payload"]["videoAsset"]["reconstruction"]["pitchCalibrationOverrides"] = [{
        "id": "manual-frame-123",
        "method": "manual-pitch-anchors",
        "sceneTime": 0.0,
        "sampleIndex": 0,
        "sourceFrameIndex": 123,
        # Deliberately below the metric gate. This observation must remain
        # auditable, but must not hide a healthy automatic observation.
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
    monkeypatch.setattr(
        "app.reconstruction_sampled_detection_preparation.load_model",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        "app.ultralytics_person_inference.predict_frame",
        lambda *_: SimpleNamespace(orig_img=image),
    )
    monkeypatch.setattr(
        "app.ultralytics_person_inference.parse_person_detections",
        lambda *_: ([], []),
    )
    monkeypatch.setattr(
        "app.reconstruction_frame_calibration_quality.calibration_alignment_metrics",
        lambda *_: _alignment(),
    )
    monkeypatch.setattr(
        "app.reconstruction_calibration_resolution.calibration_alignment_metrics",
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
    rebuilt = _reconstruct_scene_in_memory(
        scene,
        artifact_store=artifact_store,
    )

    assert captured_direct[0].method == "pnlcalib-points-lines"
    reconstruction = rebuilt["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["artifactManifest"]["schemaVersion"] == 1
    assert "calibrationFrames" not in reconstruction
    assert "identityResolver" not in reconstruction["diagnostics"]
    hydrate_scene_reconstruction(
        rebuilt,
        names=("calibrationFrames",),
        store=artifact_store,
    )
    evidence = reconstruction["calibration"]["frameEvidence"][0]
    assert evidence["status"] == "accepted"
    assert evidence["projectionSource"] == "direct"
    assert evidence["backend"] == "pnlcalib-points-lines"
    assert evidence["automaticObservation"]["status"] == "accepted"
    assert evidence["manualObservation"]["status"] == "rejected"
    assert [item["kind"] for item in evidence["observations"]] == [
        "manual",
        "automatic",
    ]
    assert "directAttempts" not in evidence
    assert "confidence-below-metric-threshold" in evidence["manualObservation"][
        "rejectionReasons"
    ]
    assert reconstruction["calibration"]["manualFrameAnchors"] == [
        {
            "id": "manual-frame-123",
            "sampleIndex": 0,
            "sourceFrameIndex": 123,
            "sceneTime": 0.0,
            "status": "rejected",
        }
    ]
