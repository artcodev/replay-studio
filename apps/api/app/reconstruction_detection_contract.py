from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_person_detection_contract import Detection


@dataclass(frozen=True)
class FrameAnalysisResult:
    frames: list[tuple[Path, float]]
    person_frames: list[tuple[list[Detection], float]]
    ball_frames: list[tuple[list[dict], float]]
    frame_size: tuple[int, int]
    person_counts: list[int]
    ball_counts: list[int]
    person_detection_cache_diagnostics: dict
    ball_detection_batches: list[dict]
    ball_detection_warnings: list[str]
    ball_dense_frame_metadata: dict
    identity_worker_diagnostics: dict
    identity_warnings: list[str]


@dataclass(frozen=True)
class CalibrationPhaseResult:
    calibration: PitchCalibration | None
    quality: dict
    coordinate_mode: str
    metric_calibration: bool
    frame_evidence: list[dict]
    accepted_frame_calibrations: dict[int, PitchCalibration]
    accepted_automatic_direct_by_sample: dict[int, PitchCalibration]
    accepted_manual_direct_by_sample: dict[int, PitchCalibration]
    resolved_calibrations_by_sample: dict[int, PitchCalibration]
    manual_override_by_sample: dict[int, dict]
    representative_manual_sample: int | None
    rejected_frame_count: int
    temporal_recovered_frame_count: int
    metric_person_sample_count: int
    metric_ball_sample_count: int
    warnings: list[str]
    contact_point_diagnostics: dict | None = None
