"""Frame-indexed evaluation context construction for labelled samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .quality_benchmark_contract import BenchmarkValidationError
from .quality_benchmark_statistics import finite_number
from .quality_benchmark_validation import (
    index_benchmark_frames,
    index_prediction_samples,
)


@dataclass(frozen=True)
class BenchmarkSampleContext:
    """One manifest sample paired with its optional prediction frames."""

    sample_id: str
    sample_index: int
    frame_rate: float | None
    ground_truth_frames: dict[int, dict[str, Any]]
    prediction_frames: dict[int, dict[str, Any]]


def build_sample_contexts(
    manifest: dict[str, Any], predictions: dict[str, Any]
) -> list[BenchmarkSampleContext]:
    """Pair manifest and prediction samples without inventing missing data."""

    benchmark_id = str(manifest["benchmark"]["id"])
    prediction_samples = index_prediction_samples(predictions, benchmark_id)
    known_ids = {str(sample["id"]) for sample in manifest["samples"]}
    unknown_ids = sorted(set(prediction_samples) - known_ids)
    if unknown_ids:
        raise BenchmarkValidationError(
            f"Predictions contain samples outside the manifest: {', '.join(unknown_ids)}"
        )
    contexts: list[BenchmarkSampleContext] = []
    for sample_index, sample in enumerate(manifest["samples"]):
        sample_id = str(sample["id"])
        source = sample.get("source") or {}
        frame_rate = finite_number(source.get("frameRate"))
        prediction = prediction_samples.get(sample_id) or {}
        contexts.append(
            BenchmarkSampleContext(
                sample_id=sample_id,
                sample_index=sample_index,
                frame_rate=frame_rate,
                ground_truth_frames=index_benchmark_frames(
                    sample["groundTruth"].get("frames"),
                    location=f"sample {sample_id!r} ground-truth frames",
                ),
                prediction_frames=index_benchmark_frames(
                    prediction.get("frames"),
                    location=f"sample {sample_id!r} prediction frames",
                ),
            )
        )
    return contexts
