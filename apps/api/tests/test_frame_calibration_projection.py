import numpy as np

from app.pitch_calibration_contract import PitchCalibration
from app.reconstruction_frame_calibration_projection import (
    project_inspection_people,
    resolve_frame_calibration,
)
from app.reconstruction_person_detection_contract import Detection


def _representative() -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.9,
        supported_lines=8,
        mean_line_score=0.9,
        rectangle="penalty-area-right",
        method="representative",
    )


def test_inspector_resolves_the_exact_published_frame_homography():
    reconstruction = {
        "calibration": {
            "frameEvidence": [
                {
                    "sourceFrameIndex": 124,
                    "solutionStatus": "direct-accepted",
                    "projectionSource": "direct",
                    "confidence": 0.93,
                    "imageToPitch": [
                        [0.1, 0.0, -48.0],
                        [0.0, 0.1, -27.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        }
    }

    calibration, source = resolve_frame_calibration(
        reconstruction,
        124,
        _representative(),
    )

    assert source == "published-per-frame-homography"
    assert calibration is not None
    assert calibration.frame_index == 124
    assert calibration.image_to_pitch[0, 0] == 0.1


def test_inspector_reports_outside_pitch_instead_of_approximate_fallback():
    calibration = PitchCalibration(
        image_to_pitch=np.eye(3, dtype=np.float64),
        confidence=0.9,
        supported_lines=8,
        mean_line_score=0.9,
        rectangle="penalty-area-right",
    )
    detection = Detection(
        x=61.78,
        y=-0.32,
        width=20.0,
        height=40.0,
        confidence=0.8,
        feature=np.zeros(12, dtype=np.float32),
    )

    accepted, raw = project_inspection_people(
        [detection],
        frame_size=(1920, 1080),
        pitch={"length": 105.0, "width": 68.0},
        calibration=calibration,
    )

    assert accepted == [None]
    assert raw[0] == (61.78, -0.32)
