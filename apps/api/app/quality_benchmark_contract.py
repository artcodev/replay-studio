"""Stable contracts shared by labelled benchmark capabilities."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "1.0"
DEFAULT_PERSON_IOU_THRESHOLD = 0.5
DEFAULT_BALL_POINT_THRESHOLD_PX = 24.0


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark cannot be evaluated reproducibly."""


@dataclass(frozen=True)
class EvaluationThresholds:
    """Metric thresholds resolved from a benchmark manifest."""

    person_iou: float = DEFAULT_PERSON_IOU_THRESHOLD
    ball_point_px: float = DEFAULT_BALL_POINT_THRESHOLD_PX
