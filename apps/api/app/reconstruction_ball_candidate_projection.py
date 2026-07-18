from __future__ import annotations

"""Metric projection and stabilization of dense-frame ball candidates."""

from copy import deepcopy

from .reconstruction_ball_projection_contract import DenseBallProjectionContext
from .reconstruction_metric_projection import attach_metric_positions
from .reconstruction_motion import stabilize_detections


def apply_dense_ball_projection(
    balls: list[dict],
    context: DenseBallProjectionContext,
    pitch: dict,
    dense_frame_index: int,
) -> int:
    """Scale, project, and stabilise candidates while retaining full provenance."""

    target_width, target_height = context.target_size
    for ball in balls:
        source_width = float(ball.get("imageWidth") or target_width)
        source_height = float(ball.get("imageHeight") or target_height)
        source_x, source_y = float(ball["x"]), float(ball["y"])
        ball["sourceImagePosition"] = {
            "x": source_x,
            "y": source_y,
            "width": source_width,
            "height": source_height,
        }
        ball["x"] = source_x * target_width / max(1.0, source_width)
        ball["y"] = source_y * target_height / max(1.0, source_height)
        ball.pop("pitchX", None)
        ball.pop("pitchZ", None)
        ball["nearestCalibrationSampleIndex"] = context.nearest_sample_index
        ball["calibrationSampleIndices"] = list(
            context.provenance.get("sampleIndices") or []
        )
        ball["calibrationInterpolationAlpha"] = context.provenance.get("alpha")
        ball["calibrationProjectionMethod"] = context.provenance.get("method")
        ball["projectionProvenance"] = deepcopy(context.provenance)
        provenance = ball.get("provenance")
        provenance = dict(provenance) if isinstance(provenance, dict) else {}
        provenance["projection"] = deepcopy(context.provenance)
        ball["provenance"] = provenance
        ball["denseFrameIndex"] = dense_frame_index

    attach_metric_positions(
        [],
        balls,
        context.calibration,
        pitch,
        projection_source=context.projection_source,
        calibration_frame_index=context.calibration_frame_index,
        position_uncertainty_metres=context.position_uncertainty_metres,
    )
    stabilize_detections([], balls, context.camera_transform)
    return sum(ball.get("pitchX") is not None for ball in balls)
