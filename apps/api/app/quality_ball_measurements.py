from __future__ import annotations

"""Ball resolver coverage, gap, and ambiguity measurements."""

from typing import Any

from .quality_evidence import bounded_ratio, finite_number
from .quality_measurement_domain import BallTrackingMeasurements


def collect_ball_tracking_measurements(
    scene: dict[str, Any],
    reconstruction: dict[str, Any],
    diagnostics: dict[str, Any],
) -> BallTrackingMeasurements:
    tracking = (
        ((scene.get("payload") or {}).get("ball") or {}).get("diagnostics")
        or (reconstruction.get("ballDetection") or {}).get("tracking")
        or diagnostics.get("ballTracking")
        or {}
    )
    gaps = tracking.get("gaps") or {}
    return BallTrackingMeasurements(
        available=bool(tracking),
        observed_coverage=bounded_ratio(tracking.get("observedCoverage")),
        published_coverage=bounded_ratio(tracking.get("publishedCoverage")),
        frame_count=int(finite_number(tracking.get("frameCount")) or 0),
        observed_frame_count=int(
            finite_number(tracking.get("observedFrameCount")) or 0
        ),
        inferred_frame_count=int(
            finite_number(tracking.get("inferredFrameCount")) or 0
        ),
        occluded_frame_count=int(
            finite_number(tracking.get("occludedFrameCount")) or 0
        ),
        gap_count=int(finite_number(gaps.get("gapCount")) or 0),
        longest_gap_seconds=finite_number(gaps.get("longestGapSeconds")),
        path_cost_margin=finite_number(tracking.get("pathCostMargin")),
    )
