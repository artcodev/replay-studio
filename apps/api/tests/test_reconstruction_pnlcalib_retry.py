from pathlib import Path

import numpy as np

import app.reconstruction_pnlcalib_retry as retry_module
from app.pitch_calibration_contract import PitchCalibration


def _calibration(confidence: float) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=confidence,
        supported_lines=8,
        mean_line_score=0.8,
        rectangle="field-keypoints-right",
        method="pnlcalib-points-lines",
        keypoint_count=8,
        detected_keypoint_count=8,
        inlier_count=8,
        inlier_ratio=1.0,
    )


def _evidence(*_args, **kwargs) -> dict:
    calibration = _args[4]
    if calibration is None:
        return {
            "status": "missing",
            "backend": None,
            "confidence": None,
            "reprojectionError": None,
            "reprojectionP95": None,
            "groundErrorP50Metres": None,
            "groundErrorP95Metres": None,
            "visiblePitchSide": None,
            "rejectionReasons": ["no-automatic-calibration-candidate"],
            "backendDiagnostics": None,
            "qualityGates": [],
        }
    accepted = calibration.confidence >= 0.9
    return {
        "status": "accepted" if accepted else "rejected",
        "backend": calibration.method,
        "confidence": calibration.confidence,
        "reprojectionError": 2.0,
        "reprojectionP95": 3.0,
        "groundErrorP50Metres": 0.2,
        "groundErrorP95Metres": 0.4,
        "visiblePitchSide": "right",
        "rejectionReasons": [] if accepted else ["confidence-below-metric-threshold"],
        "backendDiagnostics": {},
        "qualityGates": [],
    }


def test_rejected_frame_gets_at_most_two_fresh_single_frame_attempts(monkeypatch):
    fresh = iter((_calibration(0.8), _calibration(0.95)))
    calls = []
    monkeypatch.setattr(retry_module, "frame_calibration_evidence", _evidence)

    def recalibrate(frames, **_kwargs):
        calls.append(frames)
        return {7: next(fresh)}

    monkeypatch.setattr(retry_module, "recalibrate_frames_with_worker", recalibrate)
    resolution = retry_module.resolve_pnlcalib_frame_attempts(
        {"payload": {"pitch": {"length": 105, "width": 68}}},
        sample_index=0,
        source_frame_index=7,
        scene_time=0.0,
        frame_path=Path("/tmp/frame_00007.jpg"),
        image=np.zeros((32, 48, 3), dtype=np.uint8),
        initial_calibration=_calibration(0.7),
        additional_attempts=9,
    )

    assert len(calls) == 2
    assert all(call[0][0] == 7 and len(call) == 1 for call in calls)
    assert resolution.accepted_attempt == 3
    assert resolution.calibration is not None
    assert resolution.calibration.confidence == 0.95
    assert [item["requestKind"] for item in resolution.attempts] == [
        "initial-cache-aware",
        "forced-refresh-single-frame",
        "forced-refresh-single-frame",
    ]
    assert resolution.evidence["pnlcalibAttempts"]["attemptCount"] == 3


def test_accepted_initial_frame_does_not_run_a_retry(monkeypatch):
    monkeypatch.setattr(retry_module, "frame_calibration_evidence", _evidence)
    monkeypatch.setattr(
        retry_module,
        "recalibrate_frames_with_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accepted direct calibration must not be retried")
        ),
    )

    resolution = retry_module.resolve_pnlcalib_frame_attempts(
        {"payload": {"pitch": {"length": 105, "width": 68}}},
        sample_index=0,
        source_frame_index=7,
        scene_time=0.0,
        frame_path=Path("/tmp/frame_00007.jpg"),
        image=np.zeros((32, 48, 3), dtype=np.uint8),
        initial_calibration=_calibration(0.95),
    )

    assert resolution.accepted_attempt == 1
    assert len(resolution.attempts) == 1


def test_rejected_frames_retry_once_per_batch_round(monkeypatch):
    monkeypatch.setattr(retry_module, "frame_calibration_evidence", _evidence)
    monkeypatch.setattr(
        retry_module.cv2,
        "imread",
        lambda *_args: np.zeros((32, 48, 3), dtype=np.uint8),
    )
    calls: list[list[tuple[int, Path]]] = []

    def recalibrate(frames, **_kwargs):
        calls.append(frames)
        if len(calls) == 1:
            return {7: _calibration(0.95), 8: _calibration(0.8)}
        return {8: _calibration(0.96)}

    monkeypatch.setattr(retry_module, "recalibrate_frames_with_worker", recalibrate)
    batches = []
    requests = [
        retry_module.PnlCalibBatchRequest(
            sample_index=index,
            source_frame_index=source_index,
            scene_time=float(index),
            frame_path=Path(f"/tmp/frame_{source_index:05d}.jpg"),
            initial_calibration=_calibration(0.7),
        )
        for index, source_index in enumerate((7, 8))
    ]

    resolutions = retry_module.resolve_pnlcalib_batch_attempts(
        {"payload": {"pitch": {"length": 105, "width": 68}}},
        requests,
        on_retry_batch=lambda *values: batches.append(values),
    )

    assert [[source for source, _ in call] for call in calls] == [[7, 8], [8]]
    assert resolutions[0].accepted_attempt == 2
    assert resolutions[1].accepted_attempt == 3
    assert len(resolutions[0].attempts) == 2
    assert len(resolutions[1].attempts) == 3
    assert [batch[2] for batch in batches] == [2, 1]
