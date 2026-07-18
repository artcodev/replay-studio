from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .pitch_calibration_contract import PitchCalibration
from .reconstruction_calibration_resolution import (
    merge_direct_calibration_anchors,
    resolve_temporal_frame_calibrations,
)
from .reconstruction_person_detection_contract import Detection
from .reconstruction_metric_projection import attach_metric_positions
from .reconstruction_motion import stabilize_detections
from .reconstruction_progress import ReconstructionProgress


@dataclass(frozen=True)
class TemporalCalibrationResult:
    resolved_by_sample: dict[int, PitchCalibration]
    anchor_by_sample: dict[int, int]
    uncertainty_by_sample: dict[int, float]
    recovered_frame_count: int
    metric_person_sample_count: int


def solve_temporal_calibration_phase(
    scene: dict,
    frames: list[tuple[Path, float]],
    frame_sizes: Mapping[int, tuple[int, int]],
    accepted_automatic_direct_by_sample: Mapping[int, PitchCalibration],
    accepted_manual_direct_by_sample: Mapping[int, PitchCalibration],
    camera_motion_edges: Mapping[int, CameraMotionEstimate],
    camera_transforms: Mapping[int, np.ndarray],
    frame_evidence: list[dict],
    person_frames: list[tuple[list[Detection], float]],
    has_manual_stabilized_calibration: bool,
    progress: ReconstructionProgress,
) -> TemporalCalibrationResult:
    accepted_direct_by_sample = merge_direct_calibration_anchors(
        accepted_automatic_direct_by_sample,
        accepted_manual_direct_by_sample,
    )
    progress.update(
        "detection",
        3,
        "Resolving camera hypotheses",
        (
            "Running forward and backward camera inference; later strong frames "
            "may recover earlier partial views."
        ),
        62,
        84,
        completed=len(frames),
        total=len(frames),
        eta_padding=3.0,
    )
    (
        resolved_by_sample,
        anchor_by_sample,
        uncertainty_by_sample,
        recovered_frame_count,
    ) = resolve_temporal_frame_calibrations(
        frames,
        frame_sizes,
        accepted_direct_by_sample,
        camera_motion_edges,
        frame_evidence,
        person_frames,
        scene["payload"]["pitch"],
        max_gap_seconds=(
            max(2.0, float(scene["duration"]))
            if has_manual_stabilized_calibration
            else 2.0
        ),
    )

    metric_person_sample_count = 0
    for sample_index, (people, _) in enumerate(person_frames):
        evidence = frame_evidence[sample_index]
        attach_metric_positions(
            people,
            [],
            resolved_by_sample.get(sample_index),
            scene["payload"]["pitch"],
            projection_source=str(evidence.get("projectionSource") or "none"),
            calibration_frame_index=anchor_by_sample.get(sample_index),
            position_uncertainty_metres=uncertainty_by_sample.get(sample_index),
        )
        metric_person_sample_count += sum(
            person.pitch_x is not None for person in people
        )
        source_frame_index = int(evidence["sourceFrameIndex"])
        stabilize_detections(
            people,
            [],
            camera_transforms.get(
                source_frame_index,
                np.eye(3, dtype=np.float64),
            ),
        )

    return TemporalCalibrationResult(
        resolved_by_sample=resolved_by_sample,
        anchor_by_sample=anchor_by_sample,
        uncertainty_by_sample=uncertainty_by_sample,
        recovered_frame_count=recovered_frame_count,
        metric_person_sample_count=metric_person_sample_count,
    )
