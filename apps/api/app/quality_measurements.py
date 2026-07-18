from __future__ import annotations

"""Pure use case that collects every typed reconstruction QA measurement."""

from typing import Any, Iterable

from .quality_ball_measurements import collect_ball_tracking_measurements
from .quality_calibration_measurements import collect_calibration_measurements
from .quality_evidence import frame_evidence, reconstruction_document
from .quality_identity_measurements import collect_identity_measurements
from .quality_measurement_domain import ReconstructionQualityMeasurements
from .quality_motion_measurements import collect_motion_measurements
from .quality_policy import QualityThresholds
from .quality_projection_measurements import collect_projection_measurements


def collect_quality_measurements(
    scene: dict[str, Any],
    supplied_frame_evidence: Iterable[dict[str, Any]] | None,
    *,
    thresholds: QualityThresholds,
) -> ReconstructionQualityMeasurements:
    """Collect typed facts without mutating the scene or applying gate policy."""

    reconstruction = reconstruction_document(scene)
    diagnostics = reconstruction.get("diagnostics") or {}
    evidence = frame_evidence(scene, supplied_frame_evidence)
    return ReconstructionQualityMeasurements(
        processing_status=str(
            reconstruction.get("processingStatus")
            or reconstruction.get("status")
            or "unknown"
        ),
        calibration=collect_calibration_measurements(
            scene,
            reconstruction,
            diagnostics,
            evidence,
        ),
        projection=collect_projection_measurements(scene),
        motion=collect_motion_measurements(
            scene,
            diagnostics,
            evidence,
            thresholds,
        ),
        ball_tracking=collect_ball_tracking_measurements(
            scene,
            reconstruction,
            diagnostics,
        ),
        identity=collect_identity_measurements(scene),
    )


__all__ = ["collect_quality_measurements"]
