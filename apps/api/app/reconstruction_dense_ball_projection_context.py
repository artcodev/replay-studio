from __future__ import annotations

"""Temporal calibration and camera-state resolution for dense ball frames."""

from bisect import bisect_left
from dataclasses import replace

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_ball_projection_contract import DenseBallProjectionContext
from .reconstruction_bounded_homography import (
    interpolate_homography_bounded,
    normalise_interpolation_homography,
)
from .reconstruction_metric_projection import calibration_uncertainty_metres


DENSE_BALL_INTERPOLATION_MAX_GAP_SECONDS = 0.25


def dense_ball_projection_context(
    scene_time: float,
    sampled_times: list[float],
    frame_sizes: dict[int, tuple[int, int]],
    resolved_calibrations_by_sample: dict[int, PitchCalibration],
    calibration_anchor_by_sample: dict[int, int],
    calibration_uncertainty_by_sample: dict[int, float],
    frame_evidence: list[dict],
    camera_transforms: dict[int, np.ndarray],
    *,
    max_interpolation_gap_seconds: float = DENSE_BALL_INTERPOLATION_MAX_GAP_SECONDS,
) -> DenseBallProjectionContext:
    """Choose an exact, interpolated, or explicitly nearest dense-frame state."""

    if not sampled_times:
        raise ValueError("Dense ball projection requires at least one sampled frame")
    if len(sampled_times) != len(frame_evidence):
        raise ValueError("Sample times and calibration evidence must have equal length")

    nearest_sample_index = min(
        range(len(sampled_times)),
        key=lambda index: (abs(float(sampled_times[index]) - scene_time), index),
    )

    def source_frame_index(sample_index: int) -> int:
        raw = frame_evidence[sample_index].get("sourceFrameIndex")
        return int(raw) if raw is not None else sample_index

    def accepted(sample_index: int) -> bool:
        return (
            frame_evidence[sample_index].get("status") == "accepted"
            and sample_index in resolved_calibrations_by_sample
        )

    def nearest_context(
        reason: str | None,
        attempted_sample_indices: list[int],
        alpha: float | None,
        *,
        exact: bool = False,
    ) -> DenseBallProjectionContext:
        index = nearest_sample_index
        target_size = frame_sizes[index]
        evidence = frame_evidence[index]
        calibration = resolved_calibrations_by_sample.get(index) if accepted(index) else None
        calibration_validation_reason = None
        if calibration is not None:
            _, calibration_validation_reason = normalise_interpolation_homography(
                calibration.image_to_pitch,
                target_size,
            )
            if calibration_validation_reason is not None:
                calibration = None
        else:
            calibration_validation_reason = "not-qa-accepted"

        transform = camera_transforms.get(source_frame_index(index))
        transform_method = "nearest-sample"
        transform_validation_reason = "matrix-missing" if transform is None else None
        if transform is not None:
            transform, transform_validation_reason = normalise_interpolation_homography(
                transform,
                target_size,
            )
        if transform is None:
            transform = np.eye(3, dtype=np.float64)
            transform_method = "identity-fallback"

        base_uncertainty = calibration_uncertainty_by_sample.get(index)
        if base_uncertainty is None and calibration is not None:
            base_uncertainty = calibration_uncertainty_metres(calibration)
        time_offset = abs(float(sampled_times[index]) - scene_time)
        uncertainty = (
            round(min(12.0, float(base_uncertainty) + time_offset * 2.0), 3)
            if base_uncertainty is not None
            else None
        )
        method = "exact-calibration-sample" if exact else "nearest-qa-sample-fallback"
        final_reason = reason
        if calibration_validation_reason is not None:
            final_reason = (
                f"nearest-calibration-{calibration_validation_reason}"
                if final_reason is None
                else f"{final_reason};nearest-calibration-{calibration_validation_reason}"
            )
        if transform_validation_reason is not None:
            final_reason = (
                f"nearest-camera-{transform_validation_reason}"
                if final_reason is None
                else f"{final_reason};nearest-camera-{transform_validation_reason}"
            )
        used_fallback = (
            not exact
            or calibration is None
            or transform_method == "identity-fallback"
        )
        provenance = {
            "method": method,
            "sampleIndices": attempted_sample_indices or [index],
            "sourceFrameIndices": [
                source_frame_index(sample_index)
                for sample_index in (attempted_sample_indices or [index])
            ],
            "alpha": round(float(alpha), 6) if alpha is not None else None,
            "nearestSampleIndex": index,
            "nearestSourceFrameIndex": source_frame_index(index),
            "sampleTime": round(float(sampled_times[index]), 6),
            "sceneTime": round(float(scene_time), 6),
            "timeOffsetSeconds": round(time_offset, 6),
            "fallback": used_fallback,
            "fallbackReason": final_reason,
            "calibrationMethod": "exact-sample" if exact else "nearest-sample",
            "cameraTransformMethod": transform_method,
            "positionUncertaintyMetres": uncertainty,
        }
        return DenseBallProjectionContext(
            calibration=calibration,
            camera_transform=transform,
            target_size=target_size,
            nearest_sample_index=index,
            calibration_frame_index=calibration_anchor_by_sample.get(index),
            projection_source=(
                str(evidence.get("projectionSource") or "none")
                if calibration is not None
                else "none"
            ),
            position_uncertainty_metres=uncertainty,
            provenance=provenance,
        )

    if any(
        float(sampled_times[index]) >= float(sampled_times[index + 1])
        for index in range(len(sampled_times) - 1)
    ):
        return nearest_context("sample-times-not-strictly-increasing", [], None)

    insertion_index = bisect_left(sampled_times, scene_time)
    exact_index = None
    for candidate_index in (insertion_index - 1, insertion_index):
        if (
            0 <= candidate_index < len(sampled_times)
            and abs(float(sampled_times[candidate_index]) - scene_time) <= 1e-6
        ):
            exact_index = candidate_index
            break
    if exact_index is not None:
        nearest_sample_index = exact_index
        return nearest_context(None, [exact_index], 0.0, exact=True)

    if insertion_index <= 0 or insertion_index >= len(sampled_times):
        return nearest_context("dense-frame-outside-sample-bracket", [], None)

    lower_index, upper_index = insertion_index - 1, insertion_index
    lower_time = float(sampled_times[lower_index])
    upper_time = float(sampled_times[upper_index])
    interval = upper_time - lower_time
    alpha = (scene_time - lower_time) / interval
    attempted_indices = [lower_index, upper_index]
    if interval > max_interpolation_gap_seconds:
        return nearest_context(
            "sample-bracket-exceeds-interpolation-bound",
            attempted_indices,
            alpha,
        )
    if not accepted(lower_index) or not accepted(upper_index):
        return nearest_context(
            "bracket-calibration-not-qa-accepted",
            attempted_indices,
            alpha,
        )
    lower_size, upper_size = frame_sizes[lower_index], frame_sizes[upper_index]
    if lower_size != upper_size:
        return nearest_context("bracket-frame-size-mismatch", attempted_indices, alpha)
    upper_motion = frame_evidence[upper_index].get("cameraMotion") or {}
    if upper_motion.get("status") != "estimated":
        return nearest_context(
            "bracket-camera-motion-edge-not-reliable",
            attempted_indices,
            alpha,
        )

    lower_calibration = resolved_calibrations_by_sample[lower_index]
    upper_calibration = resolved_calibrations_by_sample[upper_index]
    calibration_matrix, calibration_reason = interpolate_homography_bounded(
        lower_calibration.image_to_pitch,
        upper_calibration.image_to_pitch,
        alpha,
        lower_size,
    )
    if calibration_matrix is None:
        return nearest_context(
            f"calibration-interpolation-{calibration_reason}",
            attempted_indices,
            alpha,
        )

    lower_transform = camera_transforms.get(source_frame_index(lower_index))
    upper_transform = camera_transforms.get(source_frame_index(upper_index))
    if lower_transform is None or upper_transform is None:
        return nearest_context(
            "camera-transform-endpoint-missing",
            attempted_indices,
            alpha,
        )
    camera_transform, camera_reason = interpolate_homography_bounded(
        lower_transform,
        upper_transform,
        alpha,
        lower_size,
    )
    if camera_transform is None:
        return nearest_context(
            f"camera-transform-interpolation-{camera_reason}",
            attempted_indices,
            alpha,
        )

    nearest_endpoint = lower_index if alpha <= 0.5 else upper_index
    interpolated_calibration = replace(
        lower_calibration,
        image_to_pitch=calibration_matrix,
        confidence=(
            float(lower_calibration.confidence) * (1.0 - alpha)
            + float(upper_calibration.confidence) * alpha
        ),
        method="dense-bounded-bracket-interpolation",
        frame_index=source_frame_index(nearest_endpoint),
        confidence_kind="bounded-temporal-interpolation-score",
    )
    lower_uncertainty = float(
        calibration_uncertainty_by_sample.get(
            lower_index,
            calibration_uncertainty_metres(lower_calibration),
        )
    )
    upper_uncertainty = float(
        calibration_uncertainty_by_sample.get(
            upper_index,
            calibration_uncertainty_metres(upper_calibration),
        )
    )
    motion_confidence = max(0.0, min(1.0, float(upper_motion.get("confidence") or 0.0)))
    midpoint_weight = 4.0 * alpha * (1.0 - alpha)
    interpolation_penalty = (
        0.15
        + 0.60 * midpoint_weight * interval / max(1e-6, max_interpolation_gap_seconds)
        + 0.75 * (1.0 - motion_confidence)
    )
    uncertainty = round(
        min(
            12.0,
            lower_uncertainty * (1.0 - alpha)
            + upper_uncertainty * alpha
            + interpolation_penalty,
        ),
        3,
    )
    anchor_frame_indices = list(
        dict.fromkeys(
            frame_index
            for frame_index in (
                calibration_anchor_by_sample.get(lower_index),
                calibration_anchor_by_sample.get(upper_index),
            )
            if frame_index is not None
        )
    )
    provenance = {
        "method": "bounded-bracketing-homography-interpolation",
        "sampleIndices": attempted_indices,
        "sourceFrameIndices": [
            source_frame_index(lower_index),
            source_frame_index(upper_index),
        ],
        "anchorFrameIndices": anchor_frame_indices,
        "sampleTimes": [round(lower_time, 6), round(upper_time, 6)],
        "sceneTime": round(float(scene_time), 6),
        "alpha": round(float(alpha), 6),
        "intervalSeconds": round(interval, 6),
        "maxIntervalSeconds": round(max_interpolation_gap_seconds, 6),
        "nearestSampleIndex": nearest_endpoint,
        "fallback": False,
        "fallbackReason": None,
        "calibrationMethod": "normalised-matrix-linear-interpolation",
        "cameraTransformMethod": "normalised-matrix-linear-interpolation",
        "endpointProjectionSources": [
            str(frame_evidence[lower_index].get("projectionSource") or "none"),
            str(frame_evidence[upper_index].get("projectionSource") or "none"),
        ],
        "motionConfidence": round(motion_confidence, 6),
        "positionUncertaintyMetres": uncertainty,
    }
    return DenseBallProjectionContext(
        calibration=interpolated_calibration,
        camera_transform=camera_transform,
        target_size=lower_size,
        nearest_sample_index=nearest_endpoint,
        calibration_frame_index=calibration_anchor_by_sample.get(nearest_endpoint),
        projection_source="dense-bracket-interpolated",
        position_uncertainty_metres=uncertainty,
        provenance=provenance,
    )
