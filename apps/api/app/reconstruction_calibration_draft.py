from __future__ import annotations

"""Pure construction of an inspectable pitch-calibration draft."""

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_quality import calibration_alignment_metrics
from .pitch_geometry import (
    ANCHOR_PRESETS,
    calibration_horizon,
    projected_pitch_markings,
)
from .reconstruction_calibration_policy import CALIBRATION_PASS_REPROJECTION_P95
from .reconstruction_frame_calibration_quality import semantic_alignment_passes_review


def seed_pitch_anchors(preset: str, width: int, height: int) -> list[dict]:
    layouts = {
        "center-circle": {
            "circle-left": (0.34, 0.57),
            "circle-top": (0.50, 0.40),
            "circle-right": (0.66, 0.57),
            "circle-bottom": (0.50, 0.75),
        },
        "penalty-area-right": {
            "front-far": (0.24, 0.36),
            "front-near": (0.16, 0.76),
            "goal-far": (0.74, 0.35),
            "goal-near": (0.90, 0.78),
        },
        "goal-area-right": {
            "front-far": (0.38, 0.43),
            "front-near": (0.31, 0.69),
            "goal-far": (0.74, 0.42),
            "goal-near": (0.86, 0.71),
        },
        "penalty-area-left": {
            "goal-far": (0.10, 0.35),
            "goal-near": (0.26, 0.78),
            "front-far": (0.76, 0.36),
            "front-near": (0.84, 0.76),
        },
        "goal-area-left": {
            "goal-far": (0.14, 0.42),
            "goal-near": (0.26, 0.71),
            "front-far": (0.62, 0.43),
            "front-near": (0.69, 0.69),
        },
    }
    layout = layouts[preset]
    return [
        {
            "id": anchor_id,
            "label": label,
            "image": {
                "x": round(layout[anchor_id][0] * width, 2),
                "y": round(layout[anchor_id][1] * height, 2),
            },
            "pitch": {"x": pitch[0], "z": pitch[1]},
            "source": "seed",
        }
        for anchor_id, label, pitch in ANCHOR_PRESETS[preset]
    ]


def project_preset_anchors(
    calibration: PitchCalibration,
    preset: str,
    width: int,
    height: int,
) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return seed_pitch_anchors(preset, width, height)
    anchors = []
    for anchor_id, label, pitch in ANCHOR_PRESETS[preset]:
        projected = pitch_to_image @ np.array(
            [pitch[0], pitch[1], 1.0],
            dtype=np.float64,
        )
        if abs(float(projected[2])) < 1e-8:
            return seed_pitch_anchors(preset, width, height)
        image_x = float(projected[0] / projected[2])
        image_y = float(projected[1] / projected[2])
        if not np.isfinite([image_x, image_y]).all():
            return seed_pitch_anchors(preset, width, height)
        anchors.append(
            {
                "id": anchor_id,
                "label": label,
                "image": {"x": round(image_x, 2), "y": round(image_y, 2)},
                "pitch": {"x": pitch[0], "z": pitch[1]},
            }
        )
    inside = sum(
        -width * 0.08 <= anchor["image"]["x"] <= width * 1.08
        and -height * 0.08 <= anchor["image"]["y"] <= height * 1.08
        for anchor in anchors
    )
    if inside < 3:
        return seed_pitch_anchors(preset, width, height)
    for anchor in anchors:
        anchor["source"] = "projected"
    return anchors


def calibration_draft(
    scene: dict,
    frame_index: int,
    frame_time: float,
    image: np.ndarray,
    calibration: PitchCalibration,
    preset: str,
    source: str,
    anchors: list[dict] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    height, width = image.shape[:2]
    alignment_metrics = calibration_alignment_metrics(image, calibration)
    alignment_error = (
        round(alignment_metrics.residual_p50, 2)
        if alignment_metrics is not None
        else None
    )
    if anchors is None:
        anchors = project_preset_anchors(calibration, preset, width, height)
    quality = (
        "good"
        if alignment_metrics is not None
        and alignment_metrics.residual_p95 <= CALIBRATION_PASS_REPROJECTION_P95
        and alignment_metrics.f1 >= 0.15
        else "review"
        if semantic_alignment_passes_review(alignment_metrics)
        else "poor"
    )
    draft_warnings = list(warnings or [])
    if any(anchor.get("source") == "seed" for anchor in anchors):
        draft_warnings.append(
            "Anchor projection was outside the frame; the shown anchors are an unverified manual seed."
        )
    if alignment_error is None:
        draft_warnings.append("Not enough visible white markings to score the overlay.")
    elif alignment_error > 9.0:
        draft_warnings.append(
            "Pitch overlay is still far from the detected markings; move the anchors."
        )
    return {
        "sceneId": scene["id"],
        "sceneTime": round(frame_time, 3),
        "frameIndex": frame_index + 1,
        "frameWidth": width,
        "frameHeight": height,
        "source": source,
        "preset": preset,
        "confidence": round(calibration.confidence, 3),
        "alignmentError": alignment_error,
        "alignmentMetrics": (
            alignment_metrics.as_dict() if alignment_metrics is not None else None
        ),
        "horizon": calibration_horizon(calibration, width),
        "quality": quality,
        "anchors": anchors,
        "markings": projected_pitch_markings(calibration, width, height),
        "imageToPitch": [
            [round(float(value), 10) for value in row]
            for row in calibration.image_to_pitch
        ],
        "warnings": draft_warnings,
    }
