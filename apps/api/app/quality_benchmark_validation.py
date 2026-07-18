"""Manifest, prediction, and threshold validation for labelled benchmarks."""

from __future__ import annotations

from typing import Any

from .quality_benchmark_contract import (
    DEFAULT_BALL_POINT_THRESHOLD_PX,
    DEFAULT_PERSON_IOU_THRESHOLD,
    SCHEMA_VERSION,
    BenchmarkValidationError,
    EvaluationThresholds,
)
from .quality_benchmark_statistics import bbox, finite_number, point


def _frame_index(frame: dict[str, Any], *, location: str) -> int:
    value = frame.get("frameIndex")
    if isinstance(value, bool):
        raise BenchmarkValidationError(f"{location}.frameIndex must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise BenchmarkValidationError(
            f"{location}.frameIndex must be a non-negative integer"
        ) from error
    if parsed < 0 or parsed != value:
        raise BenchmarkValidationError(f"{location}.frameIndex must be a non-negative integer")
    return parsed


def index_benchmark_frames(frames: Any, *, location: str) -> dict[int, dict[str, Any]]:
    """Index a frame array after enforcing unique non-negative frame indexes."""

    if frames is None:
        return {}
    if not isinstance(frames, list):
        raise BenchmarkValidationError(f"{location} must be an array")
    indexed: dict[int, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise BenchmarkValidationError(f"{location}[{index}] must be an object")
        frame_index = _frame_index(frame, location=f"{location}[{index}]")
        if frame_index in indexed:
            raise BenchmarkValidationError(
                f"{location} contains duplicate frameIndex {frame_index}"
            )
        indexed[frame_index] = frame
    return indexed


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate invariants that would make a score ambiguous or unstable."""

    if not isinstance(manifest, dict):
        raise BenchmarkValidationError("Benchmark manifest must be a JSON object")
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise BenchmarkValidationError(
            f"Unsupported benchmark schemaVersion {manifest.get('schemaVersion')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, dict) or not str(benchmark.get("id") or "").strip():
        raise BenchmarkValidationError("benchmark.id is required")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise BenchmarkValidationError("samples must be an array")
    seen_sample_ids: set[str] = set()
    for sample_index, sample in enumerate(samples):
        location = f"samples[{sample_index}]"
        if not isinstance(sample, dict):
            raise BenchmarkValidationError(f"{location} must be an object")
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            raise BenchmarkValidationError(f"{location}.id is required")
        if sample_id in seen_sample_ids:
            raise BenchmarkValidationError(f"Duplicate sample id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        source = sample.get("source")
        if not isinstance(source, dict):
            raise BenchmarkValidationError(f"{location}.source must be an object")
        frame_rate = finite_number(source.get("frameRate"))
        if frame_rate is not None and frame_rate <= 0:
            raise BenchmarkValidationError(f"{location}.source.frameRate must be positive")
        ground_truth = sample.get("groundTruth")
        if not isinstance(ground_truth, dict):
            raise BenchmarkValidationError(f"{location}.groundTruth must be an object")
        frames = index_benchmark_frames(
            ground_truth.get("frames"), location=f"{location}.groundTruth.frames"
        )
        for frame_index, frame in frames.items():
            frame_location = f"{location}.groundTruth.frames[frameIndex={frame_index}]"
            person_ids: set[str] = set()
            for person_index, person in enumerate(frame.get("persons") or []):
                if not isinstance(person, dict):
                    raise BenchmarkValidationError(
                        f"{frame_location}.persons[{person_index}] must be an object"
                    )
                identity = str(person.get("id") or "").strip()
                if not identity or identity in person_ids:
                    raise BenchmarkValidationError(
                        f"{frame_location} requires unique non-empty person ids"
                    )
                person_ids.add(identity)
                if bbox(person.get("bbox")) is None:
                    raise BenchmarkValidationError(
                        f"{frame_location}.persons[{person_index}].bbox is invalid"
                    )
            ball = frame.get("ball")
            if (
                isinstance(ball, dict)
                and ball.get("visible") is True
                and point(ball.get("center")) is None
            ):
                raise BenchmarkValidationError(
                    f"{frame_location}.ball.center is required when ball.visible is true"
                )
            point_ids: set[str] = set()
            for point_index, calibration_point in enumerate(frame.get("calibrationPoints") or []):
                if not isinstance(calibration_point, dict):
                    raise BenchmarkValidationError(
                        f"{frame_location}.calibrationPoints[{point_index}] must be an object"
                    )
                point_id = str(calibration_point.get("id") or "").strip()
                if not point_id or point_id in point_ids:
                    raise BenchmarkValidationError(
                        f"{frame_location} requires unique non-empty calibration point ids"
                    )
                point_ids.add(point_id)
                if (
                    point(calibration_point.get("image")) is None
                    or point(calibration_point.get("pitch")) is None
                ):
                    raise BenchmarkValidationError(
                        f"{frame_location}.calibrationPoints[{point_index}] needs image and pitch points"
                    )


def index_prediction_samples(
    predictions: dict[str, Any], benchmark_id: str
) -> dict[str, dict[str, Any]]:
    """Validate and index prediction samples for the requested benchmark."""

    if not isinstance(predictions, dict):
        raise BenchmarkValidationError("Predictions must be a JSON object")
    if predictions.get("schemaVersion") != SCHEMA_VERSION:
        raise BenchmarkValidationError(
            f"Unsupported prediction schemaVersion {predictions.get('schemaVersion')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    prediction_benchmark_id = str(predictions.get("benchmarkId") or "").strip()
    if prediction_benchmark_id != benchmark_id:
        raise BenchmarkValidationError(
            f"Predictions target benchmark {prediction_benchmark_id!r}, expected {benchmark_id!r}"
        )
    samples = predictions.get("samples")
    if not isinstance(samples, list):
        raise BenchmarkValidationError("predictions.samples must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise BenchmarkValidationError(f"predictions.samples[{index}] must be an object")
        sample_id = str(sample.get("sampleId") or "").strip()
        if not sample_id:
            raise BenchmarkValidationError(f"predictions.samples[{index}].sampleId is required")
        if sample_id in indexed:
            raise BenchmarkValidationError(f"Duplicate prediction sampleId {sample_id!r}")
        index_benchmark_frames(
            sample.get("frames"), location=f"predictions.samples[{index}].frames"
        )
        indexed[sample_id] = sample
    return indexed


def resolve_evaluation_thresholds(manifest: dict[str, Any]) -> EvaluationThresholds:
    """Resolve and validate evaluator thresholds from a validated manifest."""

    configuration = manifest.get("evaluation") or {}
    if not isinstance(configuration, dict):
        raise BenchmarkValidationError("evaluation must be an object")
    person_iou = finite_number(configuration.get("personIouThreshold"))
    ball_point = finite_number(configuration.get("ballPointThresholdPx"))
    thresholds = EvaluationThresholds(
        person_iou=person_iou if person_iou is not None else DEFAULT_PERSON_IOU_THRESHOLD,
        ball_point_px=(
            ball_point if ball_point is not None else DEFAULT_BALL_POINT_THRESHOLD_PX
        ),
    )
    if not 0.0 < thresholds.person_iou <= 1.0:
        raise BenchmarkValidationError("evaluation.personIouThreshold must be in (0, 1]")
    if thresholds.ball_point_px <= 0.0:
        raise BenchmarkValidationError("evaluation.ballPointThresholdPx must be positive")
    return thresholds
