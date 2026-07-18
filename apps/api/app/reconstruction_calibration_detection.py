from __future__ import annotations

"""Direct pitch-calibration observation acquisition and candidate selection."""

from pathlib import Path
from typing import Callable

import numpy as np

from .calibration_worker import CalibrationWorkerError, calibrate_frames_with_worker
from .config import get_settings
from .field_keypoints import calibration_from_pose_result
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_inputs import (
    load_model,
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
            2
            if item.method.startswith("pnlcalib")
            else 1
            if item.method == "roboflow-field-keypoints"
            else 0,
            item.confidence,
            item.inlier_count,
            item.keypoint_count,
            -(item.reprojection_error if item.reprojection_error is not None else 999.0),
        ),
    )


def positive_image_size(value) -> int | None:
    if isinstance(value, (tuple, list)):
        values = [positive_image_size(item) for item in value]
        values = [item for item in values if item is not None]
        return max(values) if values else None
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if resolved > 0 else None


def pitch_keypoint_inference_size(model, configured_size: int | None) -> int:
    """Use an explicit override, otherwise honor the checkpoint training size."""

    explicit = positive_image_size(configured_size)
    if explicit is not None:
        return explicit
    metadata_sources = (
        getattr(model, "overrides", None),
        getattr(getattr(model, "model", None), "args", None),
        getattr(getattr(model, "model", None), "yaml", None),
    )
    for metadata in metadata_sources:
        if isinstance(metadata, dict):
            native = positive_image_size(metadata.get("imgsz") or metadata.get("img_size"))
        else:
            native = positive_image_size(
                getattr(metadata, "imgsz", None) or getattr(metadata, "img_size", None)
            )
        if native is not None:
            return native
    # This is also the native size of the bundled Roboflow Sports checkpoint.
    return 640


def local_frame_calibrations(
    frames: list[tuple[Path, float]],
    requested_indices: set[int] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[int, PitchCalibration]:
    settings = get_settings()
    model_path = Path(settings.pitch_keypoint_model)
    if not model_path.is_file():
        return {}
    selected = [
        (path, parse_source_frame_index(path))
        for path, _ in frames
        if requested_indices is None
        or parse_source_frame_index(path) in requested_indices
    ]
    if not selected:
        return {}
    model = load_model(str(model_path))
    inference_size = pitch_keypoint_inference_size(
        model,
        settings.pitch_keypoint_image_size,
    )
    result: dict[int, PitchCalibration] = {}
    for start in range(0, len(selected), 4):
        batch = selected[start : start + 4]
        predictions = model.predict(
            [str(path) for path, _ in batch],
            imgsz=inference_size,
            device=settings.reconstruction_device,
            verbose=False,
        )
        for prediction, (_, source_index) in zip(predictions, batch):
            calibration = calibration_from_pose_result(prediction, source_index)
            if calibration is not None:
                result[source_index] = calibration
        if on_progress is not None:
            on_progress(min(len(selected), start + len(batch)), len(selected), len(result))
    return result


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
    worker_timeout: float | None = None,
) -> tuple[dict[int, PitchCalibration], list[str]]:
    settings = get_settings()
    anchor_frames = select_calibration_anchor_frames(
        frames,
        float(getattr(settings, "calibration_anchor_max_gap_seconds", 1.0)),
    )
    indexed = [(parse_source_frame_index(path), path) for path, _ in anchor_frames]
    warnings: list[str] = []
    calibrations: dict[int, PitchCalibration] = {}
    worker_configured = bool(settings.calibration_worker_url)
    worker_failed = False
    if worker_configured:
        if on_progress is not None:
            on_progress("pnlcalib", 0, len(indexed), 0.0, 0)
        try:
            calibrations.update(
                calibrate_frames_with_worker(
                    indexed,
                    on_progress=(
                        lambda completed, total, valid: on_progress(
                            "pnlcalib",
                            completed,
                            total,
                            0.9 * completed / max(1, total),
                            valid,
                        )
                        if on_progress is not None
                        else None
                    ),
                    timeout=worker_timeout,
                )
            )
        except CalibrationWorkerError as exc:
            worker_failed = True
            warnings.append(str(exc))
    missing = {index for index, _ in indexed} - set(calibrations)
    if missing:
        if on_progress is not None:
            on_progress("local-keypoints", 0, len(missing), 0.0 if worker_failed else 0.9, len(calibrations))
        local = local_frame_calibrations(
            anchor_frames,
            missing,
            on_progress=(
                lambda completed, total, valid: on_progress(
                    "local-keypoints",
                    completed,
                    total,
                    completed / max(1, total)
                    if worker_failed or not worker_configured
                    else 0.9 + 0.1 * completed / max(1, total),
                    len(calibrations) + valid,
                )
                if on_progress is not None
                else None
            ),
        )
        calibrations.update(local)
        if worker_configured and local:
            warnings.append(
                f"Local semantic-keypoint fallback calibrated {len(local)} frames missed by PnLCalib."
            )
    if on_progress is not None:
        backend = "local-keypoints" if worker_failed or not worker_configured else "pnlcalib"
        on_progress(backend, len(indexed), len(indexed), 1.0, len(calibrations))
    return calibrations, warnings
