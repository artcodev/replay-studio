from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_orientation import canonicalize_penalty_side
from .reconstruction_calibration_detection import best_pitch_calibration
from .reconstruction_shot_calibration_quality import evaluate_calibration_quality


@dataclass(frozen=True)
class CalibrationSelectionResult:
    calibration: PitchCalibration | None
    quality: dict
    coordinate_mode: str
    metric: bool
    representative_manual_sample: int | None
    warnings: list[str]


def select_representative_calibration(
    *,
    frames: list[tuple[Path, float]],
    frame_size: tuple[int, int],
    frame_evidence: list[dict],
    accepted_frame_calibrations: Mapping[int, PitchCalibration],
    accepted_manual_direct_by_sample: Mapping[int, PitchCalibration],
    camera_transforms: Mapping[int, np.ndarray],
    manual_stabilized_by_sample: Mapping[int, PitchCalibration],
    manual_reference: Mapping,
    rejected_frame_count: int,
    temporal_recovered_frame_count: int,
    warnings: list[str],
) -> CalibrationSelectionResult:
    calibration_warnings = list(warnings)
    if rejected_frame_count:
        calibration_warnings.append(
            f"Rejected {rejected_frame_count} frame calibrations that failed "
            "geometric QA."
        )
    if temporal_recovered_frame_count:
        calibration_warnings.append(
            f"Recovered {temporal_recovered_frame_count} frame calibrations from "
            "forward/backward camera hypotheses."
        )

    representative = best_pitch_calibration(accepted_frame_calibrations)
    representative_manual_sample: int | None = None
    if accepted_manual_direct_by_sample:
        representative_manual_sample = min(
            accepted_manual_direct_by_sample,
            key=lambda index: abs(
                float(frames[index][1])
                - float(manual_reference.get("sceneTime") or 0.0)
            ),
        )

    calibration: PitchCalibration | None = None
    if (
        representative is not None
        and representative.frame_index in camera_transforms
    ):
        try:
            image_to_stabilized_pitch = (
                representative.image_to_pitch
                @ np.linalg.inv(camera_transforms[int(representative.frame_index)])
            )
            image_to_stabilized_pitch /= image_to_stabilized_pitch[2, 2]
            calibration = replace(
                representative,
                image_to_pitch=image_to_stabilized_pitch,
            )
        except np.linalg.LinAlgError:
            calibration_warnings.append(
                "The representative frame transform could not be inverted."
            )
    if representative_manual_sample is not None:
        calibration = manual_stabilized_by_sample[representative_manual_sample]
    if calibration is not None:
        calibration = canonicalize_penalty_side(calibration, frame_size[0])

    calibration_quality = evaluate_calibration_quality(frame_evidence)
    coordinate_mode = (
        "metric"
        if calibration_quality["verdict"] in {"pass", "review"}
        else "unavailable"
    )
    return CalibrationSelectionResult(
        calibration=calibration,
        quality=calibration_quality,
        coordinate_mode=coordinate_mode,
        metric=calibration_quality["verdict"] == "pass",
        representative_manual_sample=representative_manual_sample,
        warnings=calibration_warnings,
    )
