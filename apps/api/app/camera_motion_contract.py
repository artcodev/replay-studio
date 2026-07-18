from __future__ import annotations

"""Provider-neutral camera-motion evidence shared by producer and solvers."""

from dataclasses import dataclass

import numpy as np


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return (
        round(float(value), digits)
        if value is not None and np.isfinite(value)
        else None
    )


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


@dataclass(frozen=True)
class CameraMotionEstimate:
    """Projective motion from the current image into the previous image."""

    matrix: np.ndarray
    status: str
    confidence: float
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    residual_p50: float | None = None
    residual_p95: float | None = None
    forward_backward_p95: float | None = None
    coverage_ratio: float = 0.0
    scene_change_score: float | None = None
    reason: str | None = None

    @property
    def reliable(self) -> bool:
        return self.status == "estimated"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "model": "projective-homography",
            "confidence": round(float(self.confidence), 5),
            "currentToPrevious": _matrix_payload(self.matrix),
            "metrics": {
                "trackedCount": int(self.tracked_count),
                "inlierCount": int(self.inlier_count),
                "inlierRatio": round(float(self.inlier_ratio), 5),
                "residualP50Px": _round_optional(self.residual_p50),
                "residualP95Px": _round_optional(self.residual_p95),
                "forwardBackwardP95Px": _round_optional(
                    self.forward_backward_p95
                ),
                "coverageRatio": round(float(self.coverage_ratio), 5),
                "sceneChangeScore": _round_optional(self.scene_change_score),
            },
            "rejectionReasons": [self.reason] if self.reason else [],
        }
