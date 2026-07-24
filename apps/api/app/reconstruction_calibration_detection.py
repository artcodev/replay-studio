from __future__ import annotations

"""Direct pitch-calibration observation acquisition and candidate selection."""

from pathlib import Path
from typing import Callable

import numpy as np

from .calibration_worker import (
    CalibrationWorkerBatchProgress,
    CalibrationWorkerError,
    calibrate_frames_with_worker,
)
from .config import get_settings
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_inputs import (
    source_frame_index as parse_source_frame_index,
)


def best_pitch_calibration(
    calibrations: dict[int, PitchCalibration],
) -> PitchCalibration | None:
    if not calibrations:
        return None
    return max(
        calibrations.values(),
        key=lambda item: (
            item.confidence,
            item.inlier_count,
            item.keypoint_count,
            -(item.reprojection_error if item.reprojection_error is not None else 999.0),
        ),
    )


def select_calibration_anchor_frames(
    frames: list[tuple[Path, float]],
    max_gap_seconds: float,
) -> list[tuple[Path, float]]:
    """Select chronological direct-calibration anchors with a bounded gap.

    The first and last samples are always retained. For each interior span we
    greedily keep the latest available sample that does not exceed the gap, so
    a regular 10 FPS shot with a one-second gap needs roughly one expensive
    calibration per second. If the source samples themselves are farther apart
    than the configured gap, both sides of that unavoidable gap are retained.

    Manual frame overrides do not pass through this selector: they are merged
    as authoritative direct anchors by the temporal calibration pipeline.
    """

    if len(frames) <= 2:
        return list(frames)
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0.0:
        # An invalid performance setting must preserve the previous
        # accuracy-first behaviour rather than silently dropping evidence.
        return list(frames)

    selected_indices = [0]
    last_selected = 0
    candidate = 1
    epsilon = 1e-9
    while candidate < len(frames):
        elapsed = float(frames[candidate][1]) - float(frames[last_selected][1])
        if elapsed <= max_gap_seconds + epsilon:
            candidate += 1
            continue

        previous = candidate - 1
        next_anchor = previous if previous > last_selected else candidate
        selected_indices.append(next_anchor)
        last_selected = next_anchor
        if next_anchor == candidate:
            candidate += 1

    if selected_indices[-1] != len(frames) - 1:
        selected_indices.append(len(frames) - 1)
    return [frames[index] for index in selected_indices]


def automatic_frame_calibrations(
    frames: list[tuple[Path, float]],
    on_progress: Callable[[str, int, int, float, int], None] | None = None,
    *,
    on_worker_batch: Callable[[CalibrationWorkerBatchProgress], None] | None = None,
    worker_timeout: float | None = None,
    direct_calibration_max_gap_seconds: float = 0.0,
) -> tuple[dict[int, PitchCalibration], list[str]]:
    settings = get_settings()
    anchor_frames = select_calibration_anchor_frames(
        frames,
        direct_calibration_max_gap_seconds,
    )
    indexed = [(parse_source_frame_index(path), path) for path, _ in anchor_frames]
    warnings: list[str] = []
    if not settings.calibration_worker_url:
        raise CalibrationWorkerError(
            "PnLCalib calibration worker is required but is not configured"
        )
    if on_progress is not None:
        on_progress("pnlcalib", 0, len(indexed), 0.0, 0)
    calibrations = calibrate_frames_with_worker(
        indexed,
        on_progress=(
            lambda completed, total, valid: on_progress(
                "pnlcalib",
                completed,
                total,
                completed / max(1, total),
                valid,
            )
            if on_progress is not None
            else None
        ),
        on_batch=on_worker_batch,
        timeout=worker_timeout,
    )
    missing = len(indexed) - len(calibrations)
    if missing:
        warnings.append(
            f"Initial PnLCalib pass returned no direct calibration for {missing} "
            "frame(s); rejected frames will receive up to two fresh batch "
            "retry rounds with frame-local QA before temporal recovery."
        )
    if on_progress is not None:
        on_progress("pnlcalib", len(indexed), len(indexed), 1.0, len(calibrations))
    return calibrations, warnings
