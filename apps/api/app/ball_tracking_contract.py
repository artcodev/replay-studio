"""Provider-neutral contracts for temporal ball trajectory resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Literal, Mapping, Sequence, TypeAlias


BallState = Literal["observed", "inferred", "occluded"]
MotionCoordinateSelector = Callable[
    [Mapping[str, Any], int], tuple[float, float, str] | None
]
PositionProjector = Callable[
    [Mapping[str, Any]], tuple[float, float] | Mapping[str, Any]
]

# Reconstruction owns the tuple form; the detection cache owns the mapping
# form.  These are the two canonical inputs to the resolver.
BallFrameInput: TypeAlias = (
    tuple[Sequence[Any], float] | Mapping[str, Any]
)


@dataclass(frozen=True)
class BallTrackingConfig:
    """Costs and physical limits used by the temporal resolver.

    Pixel limits are fallbacks. Metric limits take precedence whenever both
    detections have calibrated pitch coordinates. Limits are soft: an
    impossible transition receives a large cost instead of deleting the
    hypothesis, keeping diagnostics available for difficult clips.
    """

    top_k_per_frame: int = 6
    beam_width: int = 128
    confidence_floor: float = 1e-4
    observation_cost_weight: float = 1.0
    occlusion_cost_per_frame: float = 1.35
    occlusion_cost_per_second: float = 0.20
    occlusion_start_penalty: float = 0.20
    reacquisition_penalty: float = 0.30
    preferred_gap_seconds: float = 0.60
    long_gap_penalty_per_second: float = 1.25
    motion_penalty_weight: float = 0.80
    acceleration_penalty_weight: float = 0.25
    physical_violation_penalty: float = 24.0
    max_ball_speed_metres_per_second: float = 55.0
    max_ball_acceleration_metres_per_second_squared: float = 180.0
    max_image_speed_pixels_per_second: float = 1_800.0
    max_image_acceleration_pixels_per_second_squared: float = 8_000.0
    max_interpolation_gap_seconds: float = 0.80
    interpolation_confidence_decay_per_second: float = 0.85
    interpolation_uncertainty_metres_per_second: float = 6.0
    minimum_observed_frames: int = 2
    minimum_peak_confidence: float = 0.12
    rendering_ball_height_metres: float = 0.22

    def __post_init__(self) -> None:
        if self.top_k_per_frame < 1 or self.beam_width < 1:
            raise ValueError("top_k_per_frame and beam_width must be positive")
        if self.minimum_observed_frames < 1:
            raise ValueError("minimum_observed_frames must be positive")
        if not 0.0 < self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")
        if not 0.0 <= self.minimum_peak_confidence <= 1.0:
            raise ValueError("minimum_peak_confidence must be between 0 and 1")
        non_negative = (
            self.observation_cost_weight,
            self.occlusion_cost_per_frame,
            self.occlusion_cost_per_second,
            self.occlusion_start_penalty,
            self.reacquisition_penalty,
            self.preferred_gap_seconds,
            self.long_gap_penalty_per_second,
            self.motion_penalty_weight,
            self.acceleration_penalty_weight,
            self.physical_violation_penalty,
            self.max_interpolation_gap_seconds,
            self.interpolation_confidence_decay_per_second,
            self.interpolation_uncertainty_metres_per_second,
            self.rendering_ball_height_metres,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in non_negative):
            raise ValueError("Ball-tracking costs must be finite and non-negative")
        positive = (
            self.max_ball_speed_metres_per_second,
            self.max_ball_acceleration_metres_per_second_squared,
            self.max_image_speed_pixels_per_second,
            self.max_image_acceleration_pixels_per_second_squared,
        )
        if any(not isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("Ball-tracking physical limits must be finite and positive")


DEFAULT_BALL_TRACKING_CONFIG = BallTrackingConfig()


@dataclass(frozen=True)
class BallTrajectoryResolution:
    """Resolved editor keyframes plus JSON-serialisable temporal diagnostics."""

    keyframes: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "keyframes": deepcopy(self.keyframes),
            "diagnostics": deepcopy(self.diagnostics),
        }


__all__ = [
    "BallFrameInput",
    "BallState",
    "BallTrackingConfig",
    "BallTrajectoryResolution",
    "DEFAULT_BALL_TRACKING_CONFIG",
    "MotionCoordinateSelector",
    "PositionProjector",
]
