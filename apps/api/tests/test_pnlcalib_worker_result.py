import numpy as np

from app.pnlcalib_worker_result import calibration_from_worker_result
from app.reconstruction_metric_projection import (
    attach_metric_positions as _attach_metric_positions,
)
from app.reconstruction_person_detection_contract import Detection
from app.reconstruction_track_state import TrackState
from app.track_observation_accumulator import append_track_observation


def test_worker_result_retains_pnl_quality_diagnostics():
    calibration = calibration_from_worker_result(
        {
            "frameIndex": 91,
            "method": "pnlcalib-points-lines",
            "confidence": 0.91,
            "keypointCount": 14,
            "inlierCount": 12,
            "lineCount": 8,
            "inlierRatio": 0.86,
            "reprojectionError": 2.7,
            "completedKeypointCount": 18,
            "detectedLineCount": 9,
            "groundErrorP50Metres": 0.42,
            "groundErrorP95Metres": 1.1,
            "rawKeypoints": [
                {
                    "id": 3,
                    "image": {"x": 120.0, "y": 240.0},
                    "pitch": {"x": -52.5, "z": 9.16},
                    "confidence": 0.9,
                    "inlier": True,
                }
            ],
            "rawLines": [
                {
                    "id": 2,
                    "name": "Big rect. left main",
                    "start": {"x": 101.5, "y": 88.0},
                    "end": {"x": 126.0, "y": 315.25},
                    "confidence": 0.84,
                    "groundPlane": True,
                },
                {
                    "id": 7,
                    "name": "Goal left crossbar",
                    "start": {"x": 71.0, "y": 129.0},
                    "end": {"x": 111.0, "y": 128.0},
                    "confidence": 0.72,
                    "groundPlane": False,
                },
            ],
            "pitchSide": "right",
            "imageToPitch": [
                [0.1, 0.0, -48.0],
                [0.0, 0.1, -27.0],
                [0.0, 0.0, 1.0],
            ],
        }
    )

    assert calibration is not None
    assert calibration.frame_index == 91
    assert calibration.rectangle == "field-keypoints-right"
    assert calibration.reprojection_error == 2.7
    evidence = calibration.as_dict()
    assert evidence["method"] == "pnlcalib-points-lines"
    assert evidence["detectedKeypointCount"] == 14
    assert evidence["completedKeypointCount"] == 18
    assert evidence["groundErrorP95Metres"] == 1.1
    assert evidence["rawKeypoints"][0]["id"] == 3
    assert evidence["rawLineCount"] == 9
    assert evidence["rawLines"][1]["groundPlane"] is False


def test_per_frame_metric_position_is_preserved_through_tracking():
    calibration = calibration_from_worker_result(
        {
            "frameIndex": 1,
            "confidence": 0.9,
            "keypointCount": 8,
            "inlierCount": 8,
            "imageToPitch": [
                [0.1, 0.0, -48.0],
                [0.0, 0.1, -27.0],
                [0.0, 0.0, 1.0],
            ],
        }
    )
    detection = Detection(
        600.0,
        350.0,
        20.0,
        50.0,
        0.8,
        np.zeros(12, dtype=np.float32),
    )
    balls = [{"x": 500.0, "y": 300.0, "confidence": 0.5}]

    _attach_metric_positions(
        [detection],
        balls,
        calibration,
        {"length": 105, "width": 68},
    )
    track = TrackState(id=1)
    append_track_observation(track, detection, frame_index=0, time=0.0)

    assert (detection.pitch_x, detection.pitch_z) == (12.0, 8.0)
    assert (balls[0]["pitchX"], balls[0]["pitchZ"]) == (2.0, 3.0)
    assert track.points[0]["pitchX"] == 12.0
    assert track.points[0]["pitchZ"] == 8.0
