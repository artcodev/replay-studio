from __future__ import annotations

"""Immutable input assembled for one dense-frame ball projection."""

from dataclasses import dataclass

import numpy as np

from .pitch_calibration_contract import PitchCalibration


@dataclass(frozen=True)
class DenseBallProjectionContext:
    """Auditable camera and calibration state for one dense ball frame."""

    calibration: PitchCalibration | None
    camera_transform: np.ndarray
    target_size: tuple[int, int]
    nearest_sample_index: int
    calibration_frame_index: int | None
    projection_source: str
    position_uncertainty_metres: float | None
    provenance: dict
