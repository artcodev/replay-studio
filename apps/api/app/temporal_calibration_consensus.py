from __future__ import annotations

import numpy as np

from .temporal_calibration_contract import CalibrationHypothesis, TemporalCalibrationFrame
from .temporal_homography import fit_image_to_pitch_homography, project_image_points


def bidirectional_consensus_homography(
    target: TemporalCalibrationFrame,
    first: CalibrationHypothesis,
    second: CalibrationHypothesis,
) -> np.ndarray | None:
    earlier, later = sorted(
        (first, second),
        key=lambda item: (item.anchor_scene_time, item.anchor_sample_index),
    )
    span = float(later.anchor_scene_time) - float(earlier.anchor_scene_time)
    if span <= 1e-9:
        return None
    progress = (float(target.scene_time) - float(earlier.anchor_scene_time)) / span
    if progress < -1e-9 or progress > 1.0 + 1e-9:
        return None
    progress = max(0.0, min(1.0, progress))
    later_weight = progress * progress * (3.0 - 2.0 * progress)
    xs = np.linspace(target.width * 0.14, target.width * 0.86, 6)
    ys = np.linspace(target.height * 0.40, target.height * 0.92, 5)
    image_points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    earlier_pitch = project_image_points(
        image_points, earlier.calibration.image_to_pitch
    )
    later_pitch = project_image_points(image_points, later.calibration.image_to_pitch)
    valid = (
        np.isfinite(earlier_pitch).all(axis=1)
        & np.isfinite(later_pitch).all(axis=1)
        & (np.max(np.abs(earlier_pitch), axis=1) < 1e5)
        & (np.max(np.abs(later_pitch), axis=1) < 1e5)
    )
    if int(valid.sum()) < 12:
        return None
    source = image_points[valid]
    blended_pitch = (
        earlier_pitch[valid] * (1.0 - later_weight)
        + later_pitch[valid] * later_weight
    )
    return fit_image_to_pitch_homography(source, blended_pitch)
