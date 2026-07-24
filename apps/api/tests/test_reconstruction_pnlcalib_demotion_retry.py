from pathlib import Path

import numpy as np

import app.reconstruction_pnlcalib_demotion_retry as retry_module
from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_sampled_frame_contract import SampledCalibrationAnalysis


def _calibration(frame_index: int) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.95,
        supported_lines=8,
        mean_line_score=0.8,
        rectangle="field-keypoints-right",
        method="pnlcalib-points-lines",
        keypoint_count=8,
        detected_keypoint_count=8,
        inlier_count=8,
        inlier_ratio=1.0,
        frame_index=frame_index,
    )


def test_shot_level_p95_outlier_gets_a_fresh_retry_before_demotion(monkeypatch):
    frames = [
        (Path(f"/tmp/frame_{index:05d}.jpg"), float(index))
        for index in range(3)
    ]
    calibrations = {index: _calibration(index) for index in range(3)}
    evidence = [
        {
            "sourceFrameIndex": index,
            "sampleIndex": index,
            "status": "accepted",
            "solutionStatus": "direct-accepted",
            "projectionSource": "direct",
            "alignmentMetrics": {"residualP95": residual},
            "rejectionReasons": [],
            "pnlcalibAttempts": {
                "attemptCount": 1,
                "maximumAttempts": 3,
                "acceptedAttempt": 1,
                "attempts": [{"attempt": 1, "selected": True}],
            },
        }
        for index, residual in enumerate((2.0, 2.0, 20.0))
    ]
    analysis = SampledCalibrationAnalysis(
        frame_size=(960, 540),
        frame_sizes={index: (960, 540) for index in range(3)},
        camera_motion_edges={},
        camera_transforms={index: np.eye(3) for index in range(3)},
        accepted_frame_calibrations=dict(calibrations),
        accepted_automatic_direct_by_sample=dict(calibrations),
        accepted_manual_direct_by_sample={},
        frame_evidence=evidence,
        rejected_frame_count=0,
    )
    calls = []
    monkeypatch.setattr(
        retry_module.cv2,
        "imread",
        lambda *_args: np.zeros((540, 960, 3), dtype=np.uint8),
    )

    def fresh(frames, **_kwargs):
        calls.append(frames)
        return {2: _calibration(2)}

    monkeypatch.setattr(retry_module, "recalibrate_frames_with_worker", fresh)
    monkeypatch.setattr(
        retry_module,
        "frame_calibration_evidence",
        lambda *_args, **_kwargs: {
            "sourceFrameIndex": 2,
            "sampleIndex": 2,
            "status": "accepted",
            "solutionStatus": "direct-accepted",
            "projectionSource": "direct",
            "backend": "pnlcalib-points-lines",
            "confidence": 0.95,
            "reprojectionError": 1.0,
            "reprojectionP95": 2.0,
            "alignmentMetrics": {"residualP95": 2.0},
            "rejectionReasons": [],
            "qualityGates": [],
        },
    )

    result = retry_module.retry_demoted_pnlcalib_anchors(
        {"payload": {"pitch": {"length": 105, "width": 68}}},
        frames,
        analysis,
        additional_attempts=2,
        residual_floor_pixels=6.5,
        best_quartile_ratio=1.6,
        max_gap_seconds=2.0,
    )

    assert calls == [[(2, frames[2][0])]]
    assert result.retried_frame_count == 1
    assert result.forced_attempt_count == 1
    assert result.recovered_frame_count == 1
    assert sorted(result.analysis.accepted_automatic_direct_by_sample) == [0, 1, 2]
    attempts = result.analysis.frame_evidence[2]["pnlcalibAttempts"]["attempts"]
    assert attempts[-1]["requestKind"] == "forced-refresh-batch-after-p95-demotion"
    assert attempts[-1]["selected"] is True
