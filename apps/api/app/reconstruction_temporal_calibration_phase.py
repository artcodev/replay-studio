from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .config import get_settings
from .pitch_line_mask_cache import cached_pitch_line_mask_loader
from .pitch_calibration_contract import PitchCalibration
from .pose_contact_point import (
    contact_point_policy_from_settings,
    resolve_pose_contact_points,
    rtmpose_backend,
)
from .reconstruction_calibration_resolution import (
    demote_outlier_direct_anchors,
    merge_direct_calibration_anchors,
    resolve_temporal_frame_calibrations,
)
from .reconstruction_person_detection_contract import Detection
from .reconstruction_metric_projection import attach_metric_positions
from .reconstruction_motion import stabilize_detections
from .reconstruction_progress import ReconstructionProgress
from .temporal_calibration_contract import TemporalCalibrationResult


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
    *,
    contact_point_profile: str = "bbox-bottom",
    pose_backend_factory=rtmpose_backend,
) -> TemporalCalibrationResult:
    settings = get_settings()
    max_gap_seconds = (
        max(2.0, float(scene["duration"]))
        if has_manual_stabilized_calibration
        else 2.0
    )
    demoted_anchors: list[dict] = []
    if settings.calibration_anchor_p95_demotion_enabled:
        accepted_automatic_direct_by_sample, demoted_anchors = (
            demote_outlier_direct_anchors(
                dict(accepted_automatic_direct_by_sample),
                frame_evidence,
                frames,
                manual_direct=dict(accepted_manual_direct_by_sample),
                max_gap_seconds=max_gap_seconds,
                residual_floor_pixels=float(
                    settings.calibration_anchor_p95_demotion_floor
                ),
                best_quartile_ratio=float(
                    settings.calibration_anchor_p95_demotion_ratio
                ),
            )
        )
    accepted_direct_by_sample = merge_direct_calibration_anchors(
        accepted_automatic_direct_by_sample,
        accepted_manual_direct_by_sample,
    )
    progress.update(
        "calibration",
        2,
        "Resolve temporal gaps",
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
        max_gap_seconds=max_gap_seconds,
        observed_mask_loader=cached_pitch_line_mask_loader(
            Path(get_settings().media_root) / "pitch-line-masks",
            enabled=bool(get_settings().pitch_line_mask_cache_enabled),
        ),
    )

    contact_point_diagnostics: dict | None = None
    if contact_point_profile == "pose-feet":
        def contact_progress(completed: int, total: int) -> None:
            progress.update(
                "calibration",
                2,
                "Locating ground contact points",
                (
                    f"RTMPose crops {completed}/{total} · feet refine each "
                    "player's projected pitch position."
                ),
                62,
                84,
                completed=completed,
                total=total,
                eta_padding=5.0,
            )

        contact_progress(0, max(1, sum(len(people) for people, _ in person_frames)))
        contact_point_diagnostics = resolve_pose_contact_points(
            person_frames,
            policy=contact_point_policy_from_settings(),
            backend_factory=pose_backend_factory,
            on_progress=contact_progress,
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
        contact_point_diagnostics=contact_point_diagnostics,
        demoted_anchors=demoted_anchors or None,
    )
