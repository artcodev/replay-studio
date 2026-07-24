from __future__ import annotations

"""Fingerprint only the inputs that can change pitch calibration."""

import hashlib
import json

from .direct_calibration_sampling import (
    resolve_direct_calibration_max_gap_seconds,
)
from .scene_frame_exclusions import frame_exclusion_fingerprint_input


def calibration_input_fingerprint(
    scene: dict,
    *,
    sampling_frame_rate: float | None = None,
    direct_calibration_max_gap_seconds: float | None = None,
) -> str:
    payload = scene.get("payload", {})
    video = payload.get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    selected_sampling_frame_rate = (
        sampling_frame_rate
        if sampling_frame_rate is not None
        else reconstruction.get("samplingFrameRate") or video.get("fps")
    )
    selected_direct_gap = resolve_direct_calibration_max_gap_seconds(
        direct_calibration_max_gap_seconds
        if direct_calibration_max_gap_seconds is not None
        else reconstruction.get("directCalibrationMaxGapSeconds")
    )
    orientation = reconstruction.get("pitchOrientation") or {}
    visible_side_source = str(orientation.get("visiblePitchSideSource") or "")
    inputs = {
        "source": {
            "assetId": video.get("id"),
            "generationKey": video.get("generationKey"),
            "analysisFrameInput": video.get("analysisFrameInput"),
            "selectedSegmentId": video.get("selectedSegmentId"),
            "sourceStart": video.get("sourceStart"),
            "sourceEnd": video.get("sourceEnd"),
            "analysisFps": video.get("analysisFps"),
            "frameExclusions": frame_exclusion_fingerprint_input(scene),
            "samplingFrameRate": selected_sampling_frame_rate,
            "directCalibrationMaxGapSeconds": selected_direct_gap,
        },
        "pitch": payload.get("pitch"),
        "pitchCalibrationOverrides": reconstruction.get(
            "pitchCalibrationOverrides"
        )
        or [],
        "manualVisiblePitchSide": (
            orientation.get("visiblePitchSide")
            if visible_side_source.startswith("manual")
            else None
        ),
        "manualVisiblePitchSideSource": (
            visible_side_source if visible_side_source.startswith("manual") else None
        ),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = ["calibration_input_fingerprint"]
