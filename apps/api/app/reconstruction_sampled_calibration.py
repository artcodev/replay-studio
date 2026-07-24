from __future__ import annotations

"""Prepare and accumulate direct/manual calibration evidence for sampled frames."""

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .camera_motion_contract import CameraMotionEstimate
from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_orientation import canonicalize_penalty_side
from .reconstruction_calibration_detection import automatic_frame_calibrations
from .reconstruction_calibration_evidence import (
    calibration_attempt_payload,
    frame_calibration_evidence,
    matrix_payload,
)
from .reconstruction_calibration_overrides import manual_pitch_calibration_overrides
from .reconstruction_person_detection_contract import Detection
from .reconstruction_inputs import source_frame_index
from .reconstruction_motion import camera_motion_estimate
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_pnlcalib_retry import PnlCalibAttemptResolution
from .direct_calibration_sampling import (
    resolve_direct_calibration_max_gap_seconds,
)
from .reconstruction_sampled_frame_contract import (
    SampledCalibrationAnalysis,
    SampledCalibrationInputs,
)


def manual_calibration_inputs(
    frames: list[tuple[Path, float]],
    overrides: list[dict],
) -> tuple[dict[int, PitchCalibration], dict[int, dict]]:
    stabilized_by_sample: dict[int, PitchCalibration] = {}
    override_by_sample: dict[int, dict] = {}
    for stored in overrides:
        requested_source_frame = stored.get("sourceFrameIndex")
        if requested_source_frame is not None:
            sample_index = min(
                range(len(frames)),
                key=lambda index: abs(
                    source_frame_index(frames[index][0])
                    - int(requested_source_frame)
                ),
            )
        else:
            requested_scene_time = float(stored.get("sceneTime") or 0.0)
            sample_index = min(
                range(len(frames)),
                key=lambda index: abs(
                    float(frames[index][1]) - requested_scene_time
                ),
            )
        source_index = source_frame_index(frames[sample_index][0])
        alignment_error = stored.get("alignmentError")
        stabilized_by_sample[sample_index] = PitchCalibration(
            image_to_pitch=np.asarray(stored["imageToPitch"], dtype=np.float64),
            confidence=float(stored.get("confidence") or 0.0),
            supported_lines=int(stored.get("supportedLines") or 4),
            mean_line_score=float(stored.get("meanLineScore") or 0.0),
            rectangle=str(stored.get("preset") or "manual"),
            matched_curves=int(stored.get("matchedCurves") or 0),
            method="manual-pitch-anchors",
            keypoint_count=int(stored.get("supportedLines") or 4),
            inlier_count=int(stored.get("supportedLines") or 4),
            reprojection_error=(
                float(alignment_error) if alignment_error is not None else None
            ),
            frame_index=source_index,
            confidence_kind="manual-alignment-quality-score",
        )
        override_by_sample[sample_index] = stored
    return stabilized_by_sample, override_by_sample


def prepare_sampled_calibrations(
    frames: list[tuple[Path, float]],
    reconstruction_request: Mapping,
    progress: ReconstructionProgress,
) -> SampledCalibrationInputs:
    overrides = manual_pitch_calibration_overrides(reconstruction_request)
    direct_calibration_max_gap_seconds = (
        resolve_direct_calibration_max_gap_seconds(
            reconstruction_request.get("directCalibrationMaxGapSeconds")
        )
    )

    def calibration_progress(
        backend: str,
        completed: int,
        total: int,
        fraction: float,
        calibrated: int,
    ) -> None:
        manual_detail = (
            f" · {len(overrides)} manual frame anchor(s) will override matching samples"
            if overrides
            else ""
        )
        progress.update(
            "calibration",
            2,
            "Run direct PnLCalib",
            f"Pass 1 · infer neural field points and lines · {completed}/{total} "
            f"frames · {calibrated} valid homographies{manual_detail}. "
            "Rejected frames may receive up to two explicit retry passes next.",
            8,
            56,
            completed=completed,
            total=total,
            fraction=fraction,
            eta_padding=max(6.0, len(frames) * 0.25),
        )

    def worker_batch_progress(batch) -> None:
        if progress.run_log is None:
            return
        per_frame = batch.request_seconds / max(1, batch.batch_size)
        progress.run_log.event(
            "pnlcalib-worker-batch-finished",
            retryStage="initial-cache-aware",
            completed=batch.completed,
            total=batch.total,
            validHomographies=batch.valid,
            batchSize=batch.batch_size,
            requestSeconds=round(batch.request_seconds, 3),
            effectiveSecondsPerFrame=round(per_frame, 3),
            workerDiagnostics=batch.diagnostics,
        )

    frame_calibrations, warnings = automatic_frame_calibrations(
        frames,
        calibration_progress,
        on_worker_batch=worker_batch_progress,
        direct_calibration_max_gap_seconds=(
            direct_calibration_max_gap_seconds
        ),
    )
    manual_stabilized, manual_by_sample = manual_calibration_inputs(
        frames,
        overrides,
    )
    return SampledCalibrationInputs(
        manual_reference=overrides[-1] if overrides else {},
        frame_calibrations=frame_calibrations,
        calibration_warnings=warnings,
        manual_stabilized_by_sample=manual_stabilized,
        manual_override_by_sample=manual_by_sample,
    )


class SampledCalibrationAccumulator:
    """Consume already-decoded images while retaining sequential camera state."""

    def __init__(self, scene: dict, inputs: SampledCalibrationInputs) -> None:
        self._scene = scene
        self._inputs = inputs
        self._frame_size = (960, 540)
        self._previous_image: np.ndarray | None = None
        self._camera_transform = np.eye(3, dtype=np.float64)
        self._camera_transforms: dict[int, np.ndarray] = {}
        self._camera_motion_edges: dict[int, CameraMotionEstimate] = {}
        self._frame_sizes: dict[int, tuple[int, int]] = {}
        self._accepted_frame: dict[int, PitchCalibration] = {}
        self._accepted_automatic: dict[int, PitchCalibration] = {}
        self._accepted_manual: dict[int, PitchCalibration] = {}
        self._frame_evidence: list[dict] = []
        self._rejected = 0

    def add_frame(
        self,
        *,
        sample_index: int,
        source_index: int,
        scene_time: float,
        image: np.ndarray,
        people: list[Detection],
        automatic_observation: PnlCalibAttemptResolution | None = None,
    ) -> None:
        self._frame_size = (image.shape[1], image.shape[0])
        self._frame_sizes[sample_index] = self._frame_size
        camera_payload: dict = {
            "status": "first-frame",
            "model": "projective-homography",
            "confidence": 1.0,
            "currentToPrevious": matrix_payload(
                np.eye(3, dtype=np.float64)
            ),
            "metrics": {},
            "rejectionReasons": [],
        }
        if self._previous_image is not None:
            motion = camera_motion_estimate(self._previous_image, image)
            self._camera_motion_edges[sample_index] = motion
            camera_payload = motion.as_dict()
            self._camera_transform = (
                self._camera_transform @ motion.matrix
                if motion.reliable
                else np.eye(3, dtype=np.float64)
            )
        self._camera_transforms[source_index] = self._camera_transform.copy()

        if automatic_observation is not None:
            automatic = automatic_observation.calibration
            automatic_evidence = automatic_observation.evidence
        else:
            automatic = self._inputs.frame_calibrations.get(source_index)
            if automatic is not None:
                automatic = canonicalize_penalty_side(
                    automatic,
                    self._frame_size[0],
                )
            automatic_evidence = frame_calibration_evidence(
                self._scene,
                sample_index,
                scene_time,
                image,
                automatic,
                projection_source="direct" if automatic is not None else "none",
                people=people,
                pitch=self._scene["payload"]["pitch"],
                source_frame_index=source_index,
            )
        selected, evidence, selected_is_manual = (
            automatic,
            automatic_evidence,
            False,
        )
        manual_stabilized = self._inputs.manual_stabilized_by_sample.get(
            sample_index
        )
        if manual_stabilized is not None:
            current_to_pitch = (
                manual_stabilized.image_to_pitch @ self._camera_transform
            )
            current_to_pitch /= current_to_pitch[2, 2]
            manual = replace(
                manual_stabilized,
                image_to_pitch=current_to_pitch,
                frame_index=source_index,
            )
            manual_evidence = frame_calibration_evidence(
                self._scene,
                sample_index,
                scene_time,
                image,
                manual,
                projection_source="manual-direct",
                people=people,
                pitch=self._scene["payload"]["pitch"],
                source_frame_index=source_index,
                manual=True,
            )
            # A persisted manual anchor is an explicit operator correction for
            # this exact sample. It must win even when the automatic candidate
            # passed its frame-local gates; otherwise editing a slightly drifted
            # green frame is a no-op. A rejected manual observation intentionally
            # leaves the frame unresolved instead of silently reverting to the
            # automatic candidate. The latter remains attached below strictly as
            # diagnostic evidence.
            selected, evidence, selected_is_manual = (
                manual,
                manual_evidence,
                True,
            )
            evidence["manualObservation"] = {
                "kind": "manual",
                **calibration_attempt_payload(manual_evidence),
            }
            evidence["automaticObservation"] = {
                "kind": "automatic",
                **calibration_attempt_payload(automatic_evidence),
            }
            evidence["observations"] = [
                evidence["manualObservation"],
                evidence["automaticObservation"],
            ]
            evidence["observationChoice"] = {
                "selectedKind": "manual",
                "reason": "operator-frame-override",
                "automaticCandidateStatus": automatic_evidence.get("status"),
            }
        evidence["cameraMotion"] = {
            **camera_payload,
            "currentToReference": matrix_payload(self._camera_transform),
        }
        if evidence["status"] == "accepted":
            assert selected is not None
            self._accepted_frame[source_index] = selected
            (
                self._accepted_manual
                if selected_is_manual
                else self._accepted_automatic
            )[sample_index] = selected
        elif selected is not None:
            self._rejected += 1
        self._frame_evidence.append(evidence)
        self._previous_image = image

    def result(self) -> SampledCalibrationAnalysis:
        return SampledCalibrationAnalysis(
            frame_size=self._frame_size,
            frame_sizes=self._frame_sizes,
            camera_motion_edges=self._camera_motion_edges,
            camera_transforms=self._camera_transforms,
            accepted_frame_calibrations=self._accepted_frame,
            accepted_automatic_direct_by_sample=self._accepted_automatic,
            accepted_manual_direct_by_sample=self._accepted_manual,
            frame_evidence=self._frame_evidence,
            rejected_frame_count=self._rejected,
        )


__all__ = [
    "SampledCalibrationAccumulator",
    "manual_calibration_inputs",
    "prepare_sampled_calibrations",
]
