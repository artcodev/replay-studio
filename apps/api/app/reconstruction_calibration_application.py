from __future__ import annotations

"""Apply an immutable calibration snapshot to reconstruction detections."""

from dataclasses import dataclass

from .pose_contact_point import (
    contact_point_policy_from_settings,
    resolve_pose_contact_points,
    rtmpose_backend,
)
from .reconstruction_calibration_snapshot import PersistedCalibrationSnapshot
from .reconstruction_dense_ball_phase import DenseBallDetectionResult
from .reconstruction_metric_projection import attach_metric_positions
from .reconstruction_motion import stabilize_detections
from .reconstruction_person_detection_contract import Detection
from .reconstruction_progress import ReconstructionProgress


@dataclass(frozen=True)
class PersonCalibrationApplication:
    total_count: int
    metric_count: int
    direct_metric_count: int
    temporal_metric_count: int
    contact_point_diagnostics: dict | None


def apply_snapshot_to_people(
    scene: dict,
    person_frames: list[tuple[list[Detection], float]],
    snapshot: PersistedCalibrationSnapshot,
    *,
    contact_point_profile: str,
    progress: ReconstructionProgress,
    pose_backend_factory=rtmpose_backend,
) -> PersonCalibrationApplication:
    total_people = sum(len(people) for people, _ in person_frames)
    contact_point_diagnostics = None
    if contact_point_profile == "pose-feet":

        def contact_progress(completed: int, total: int) -> None:
            progress.update(
                "detection",
                3,
                "Locating ground contact points",
                f"RTMPose crops {completed}/{total} · applying stored pitch matrices.",
                8,
                38,
                completed=completed,
                total=total,
                eta_padding=5.0,
            )

        contact_progress(0, max(1, total_people))
        contact_point_diagnostics = resolve_pose_contact_points(
            person_frames,
            policy=contact_point_policy_from_settings(),
            backend_factory=pose_backend_factory,
            on_progress=contact_progress,
        )

    metric_people = 0
    direct_metric_people = 0
    temporal_metric_people = 0
    for sample_index, (people, _) in enumerate(person_frames):
        evidence = snapshot.result.frame_evidence[sample_index]
        attach_metric_positions(
            people,
            [],
            snapshot.temporal.resolved_by_sample.get(sample_index),
            scene["payload"]["pitch"],
            projection_source=str(evidence.get("projectionSource") or "none"),
            calibration_frame_index=snapshot.temporal.anchor_by_sample.get(
                sample_index
            ),
            position_uncertainty_metres=(
                snapshot.temporal.uncertainty_by_sample.get(sample_index)
            ),
        )
        projected = sum(person.pitch_x is not None for person in people)
        metric_people += projected
        if str(evidence.get("solutionStatus") or "") == "direct-accepted":
            direct_metric_people += projected
        elif str(evidence.get("solutionStatus") or "") == "temporal-accepted":
            temporal_metric_people += projected
        stabilize_detections(
            people,
            [],
            snapshot.camera_transforms[int(evidence["sourceFrameIndex"])],
        )
    return PersonCalibrationApplication(
        total_count=total_people,
        metric_count=metric_people,
        direct_metric_count=direct_metric_people,
        temporal_metric_count=temporal_metric_people,
        contact_point_diagnostics=contact_point_diagnostics,
    )


def calibration_impact(
    snapshot: PersistedCalibrationSnapshot,
    people: PersonCalibrationApplication,
    dense_ball: DenseBallDetectionResult,
    *,
    contact_point_profile: str,
) -> dict:
    ball_candidates = sum(len(items) for items, _ in dense_ball.frames)
    ball_fallback_candidates = sum(
        bool((candidate.get("projectionProvenance") or {}).get("fallback"))
        for candidates, _ in dense_ball.frames
        for candidate in candidates
    )
    return {
        **snapshot.provenance,
        "usedFor": [
            "person-metric-projection",
            "tracking-association",
            "ball-world-projection",
            "3d-trajectory-publication",
        ],
        "personObservationCount": people.total_count,
        "metricPersonObservationCount": people.metric_count,
        "unprojectedPersonObservationCount": (
            people.total_count - people.metric_count
        ),
        "directMetricPersonObservationCount": people.direct_metric_count,
        "temporalMetricPersonObservationCount": people.temporal_metric_count,
        "ballCandidateCount": ball_candidates,
        "metricBallCandidateCount": dense_ball.metric_sample_count,
        "unprojectedBallCandidateCount": (
            ball_candidates - dense_ball.metric_sample_count
        ),
        "ballProjectionFallbackCandidateCount": ball_fallback_candidates,
        "contactPointProfile": contact_point_profile,
    }


__all__ = (
    "PersonCalibrationApplication",
    "apply_snapshot_to_people",
    "calibration_impact",
)
