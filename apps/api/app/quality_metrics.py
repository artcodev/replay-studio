from __future__ import annotations

"""Public use case for reconstruction QA report generation.

Evidence collection, metric presentation, and gate policy are separate
capabilities.  This module only hydrates canonical artifacts and assembles the
stable JSON report consumed by reconstruction publishing and the QA CLI.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Iterable

from .quality_gate_report import assess_quality_gates
from .quality_measurements import collect_quality_measurements
from .quality_metric_report import build_quality_metrics
from .quality_policy import DEFAULT_THRESHOLDS, QualityThresholds
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction


LIMITATIONS = [
    {
        "code": "ground-plane-game-state",
        "message": "Player positions are 2D foot points on a 105 x 68 metre ground plane, not 3D body reconstruction.",
    },
    {
        "code": "single-view-visibility",
        "message": "A broadcast view cannot observe off-screen players; missing players must remain unknown unless another synchronized source supplies them.",
    },
    {
        "code": "temporal-camera-hypothesis",
        "message": "Recovered calibration is inferred from direct anchor frames and QA-gated camera motion; its anchor, alternatives, and uncertainty remain attached to every recovered frame.",
    },
    {
        "code": "ball-height-unknown",
        "message": "Single-view ground homography does not recover airborne ball height; a fixed render height is not a measurement.",
    },
    {
        "code": "runtime-gates-not-benchmark",
        "message": "These gates detect engineering failures but do not replace JaC@5, HOTA, or GS-HOTA on a held-out labelled set.",
    },
]


def evaluate_reconstruction_quality(
    scene: dict[str, Any],
    frame_evidence: Iterable[dict[str, Any]] | None = None,
    *,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Return the stable serializable QA report without mutating ``scene``."""

    if frame_evidence is None:
        artifact_manifest = (
            scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("artifactManifest")
        )
        if artifact_manifest:
            scene = deepcopy(scene)
            hydrate_scene_reconstruction(scene)

    measurements = collect_quality_measurements(
        scene,
        frame_evidence,
        thresholds=thresholds,
    )
    assessment = assess_quality_gates(measurements, thresholds)
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "processingStatus": measurements.processing_status,
        "verdict": assessment.verdict,
        "summary": assessment.summary,
        "thresholds": asdict(thresholds),
        "metrics": build_quality_metrics(measurements, thresholds),
        "identityValidation": measurements.identity.validation,
        "gates": list(assessment.gates),
        "limitations": deepcopy(LIMITATIONS),
    }


__all__ = ["evaluate_reconstruction_quality"]
