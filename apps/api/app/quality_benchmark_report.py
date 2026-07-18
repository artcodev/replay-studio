"""Pure orchestration of labelled benchmark evaluators into a stable report."""

from __future__ import annotations

from typing import Any, Sequence

from .quality_benchmark_ball import evaluate_ball
from .quality_benchmark_calibration import evaluate_calibration
from .quality_benchmark_context import BenchmarkSampleContext, build_sample_contexts
from .quality_benchmark_contract import SCHEMA_VERSION, EvaluationThresholds
from .quality_benchmark_people import evaluate_people_and_identity
from .quality_benchmark_validation import (
    resolve_evaluation_thresholds,
    validate_manifest,
)


def _evaluate_contexts(
    contexts: Sequence[BenchmarkSampleContext], thresholds: EvaluationThresholds
) -> dict[str, Any]:
    person, identity = evaluate_people_and_identity(contexts, thresholds.person_iou)
    calibration = evaluate_calibration(contexts)
    ball = evaluate_ball(contexts, thresholds.ball_point_px)
    available = [
        person["available"],
        calibration["available"],
        ball["available"],
        identity["groundTruthAvailable"],
    ]
    status = "evaluated" if all(available) else "partial" if any(available) else "unavailable"
    return {
        "status": status,
        "personDetection": person,
        "calibration": calibration,
        "ball": ball,
        "identity": identity,
    }


def evaluate_benchmark(
    manifest: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one prediction export against a versioned ground-truth manifest."""

    validate_manifest(manifest)
    thresholds = resolve_evaluation_thresholds(manifest)
    contexts = build_sample_contexts(manifest, predictions)
    overall = _evaluate_contexts(contexts, thresholds)
    samples = [
        {
            "sampleId": context.sample_id,
            **_evaluate_contexts([context], thresholds),
        }
        for context in contexts
    ]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "benchmarkId": str(manifest["benchmark"]["id"]),
        "benchmarkStatus": manifest["benchmark"].get("status", "draft"),
        "predictionRun": predictions.get("run") or {},
        "sampleCount": len(contexts),
        "status": overall["status"],
        "thresholds": {
            "personIou": thresholds.person_iou,
            "ballPointPx": thresholds.ball_point_px,
        },
        "metrics": {key: value for key, value in overall.items() if key != "status"},
        "samples": samples,
        "limitations": [
            "Scores describe only the labelled samples in this manifest.",
            "HOTA and GS-HOTA are not approximated; use the official SoccerNet evaluator.",
            "A draft or partially labelled manifest is not an accuracy claim.",
        ],
    }
