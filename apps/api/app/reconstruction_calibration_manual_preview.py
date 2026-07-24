from __future__ import annotations

"""Non-persistent manual-anchor calibration preview."""

from copy import deepcopy
from math import exp

from .pitch_anchor_calibration import calibration_from_anchors
from .pitch_calibration_orientation import canonicalize_penalty_side
from .pitch_calibration_quality import calibration_alignment_error
from .pitch_geometry import ANCHOR_PRESETS
from .reconstruction_calibration_draft import calibration_draft
from .reconstruction_calibration_frame_context import sampled_frame_context


def preview_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
) -> dict:
    frame_index, frame_time, image = sampled_frame_context(scene, scene_time)
    resolved_anchors = deepcopy(anchors)
    rough = calibration_from_anchors(resolved_anchors, preset, confidence=0.9)
    canonical = canonicalize_penalty_side(rough, image.shape[1])
    resolved_preset = preset
    if canonical.rectangle != rough.rectangle and canonical.rectangle in ANCHOR_PRESETS:
        resolved_preset = canonical.rectangle
        for anchor in resolved_anchors:
            anchor["pitch"]["x"] = -float(anchor["pitch"]["x"])
        rough = canonical
    alignment_error = calibration_alignment_error(image, rough)
    # Manual input remains a hypothesis until image evidence supports it.
    confidence = (
        0.35
        if alignment_error is None
        else 0.55 + 0.43 * exp(-float(alignment_error) / 6.0)
    )
    calibration = calibration_from_anchors(
        resolved_anchors,
        resolved_preset,
        confidence=confidence,
    )
    return calibration_draft(
        scene,
        frame_index,
        frame_time,
        image,
        calibration,
        resolved_preset,
        "manual",
        resolved_anchors,
    )
