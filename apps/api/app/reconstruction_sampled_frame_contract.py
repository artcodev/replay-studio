from __future__ import annotations

"""Data-only contracts produced while analysing sampled reconstruction frames."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_person_detection_contract import Detection


@dataclass(frozen=True)
class SampledCalibrationInputs:
    manual_reference: Mapping
    frame_calibrations: dict[int, PitchCalibration]
    calibration_warnings: list[str]
    manual_stabilized_by_sample: dict[int, PitchCalibration]
    manual_override_by_sample: dict[int, dict]


@dataclass(frozen=True)
class SampledCalibrationAnalysis:
    frame_size: tuple[int, int]
    frame_sizes: dict[int, tuple[int, int]]
    camera_motion_edges: dict[int, CameraMotionEstimate]
    camera_transforms: dict[int, np.ndarray]
    accepted_frame_calibrations: dict[int, PitchCalibration]
    accepted_automatic_direct_by_sample: dict[int, PitchCalibration]
    accepted_manual_direct_by_sample: dict[int, PitchCalibration]
    frame_evidence: list[dict]
    rejected_frame_count: int


@dataclass(frozen=True)
class SampledFrameAnalysis:
    person_frames: list[tuple[list[Detection], float]]
    generic_ball_frames: list[tuple[list[dict], float]]
    person_counts: list[int]
    ball_counts: list[int]
    calibration: SampledCalibrationAnalysis


__all__ = [
    "SampledCalibrationAnalysis",
    "SampledCalibrationInputs",
    "SampledFrameAnalysis",
]
