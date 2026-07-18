from __future__ import annotations

"""Ground-truth identity assignment measurements for runtime QA reports."""

from typing import Any

from .identity_metrics import evaluate_identity_assignments
from .quality_evidence import finite_number
from .quality_measurement_domain import IdentityMeasurements


def collect_identity_measurements(scene: dict[str, Any]) -> IdentityMeasurements:
    ground_truth = (scene.get("payload") or {}).get("validationGroundTruth") or {}
    frame_rate = (
        finite_number(ground_truth.get("identityAssignmentFrameRate"))
        if isinstance(ground_truth, dict)
        else None
    )
    validation = evaluate_identity_assignments(
        (
            ground_truth.get("identityAssignments")
            if isinstance(ground_truth, dict)
            else None
        ),
        frame_rate=frame_rate,
    )
    return IdentityMeasurements(validation=validation)
