from __future__ import annotations

"""Explicit per-run policy for selecting expensive direct calibrations."""

import math

from .reconstruction_errors import ReconstructionError


DEFAULT_DIRECT_CALIBRATION_MAX_GAP_SECONDS = 0.0
MAX_DIRECT_CALIBRATION_GAP_SECONDS = 5.0


def resolve_direct_calibration_max_gap_seconds(value: object | None) -> float:
    """Return the canonical direct-anchor gap; zero means every sampled frame."""

    if value is None:
        return DEFAULT_DIRECT_CALIBRATION_MAX_GAP_SECONDS
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(
            "Direct calibration sampling gap must be a number"
        ) from exc
    if (
        not math.isfinite(resolved)
        or resolved < 0.0
        or resolved > MAX_DIRECT_CALIBRATION_GAP_SECONDS
    ):
        raise ReconstructionError(
            "Direct calibration sampling gap must be between 0 and 5 seconds"
        )
    return resolved


def direct_calibration_sampling_label(max_gap_seconds: float) -> str:
    return (
        "every-frame"
        if max_gap_seconds <= 0.0
        else f"max-gap-{max_gap_seconds:g}s"
    )


__all__ = (
    "DEFAULT_DIRECT_CALIBRATION_MAX_GAP_SECONDS",
    "MAX_DIRECT_CALIBRATION_GAP_SECONDS",
    "direct_calibration_sampling_label",
    "resolve_direct_calibration_max_gap_seconds",
)
