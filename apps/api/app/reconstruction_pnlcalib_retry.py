from __future__ import annotations

"""QA-aware, cache-bypassing retries for PnLCalib frames."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .calibration_worker import (
    CalibrationWorkerBatchProgress,
    recalibrate_frames_with_worker,
)
from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_orientation import canonicalize_penalty_side
from .reconstruction_calibration_evidence import (
    calibration_attempt_payload,
    frame_calibration_evidence,
)


@dataclass(frozen=True)
class PnlCalibAttemptResolution:
    calibration: PitchCalibration | None
    evidence: dict
    attempts: tuple[dict, ...]
    accepted_attempt: int | None


@dataclass(frozen=True)
class PnlCalibBatchRequest:
    sample_index: int
    source_frame_index: int
    scene_time: float
    frame_path: Path
    initial_calibration: PitchCalibration | None


def _candidate_rank(
    calibration: PitchCalibration | None,
    evidence: dict,
) -> tuple:
    status = str(evidence.get("status") or "missing")
    status_rank = 2 if status == "accepted" else 1 if status == "rejected" else 0
    residual = evidence.get("reprojectionP95")
    confidence = evidence.get("confidence")
    return (
        status_rank,
        -len(evidence.get("rejectionReasons") or []),
        -(float(residual) if residual is not None else float("inf")),
        float(confidence) if confidence is not None else -1.0,
        calibration.inlier_count if calibration is not None else 0,
    )


def _evaluate_candidate(
    scene: dict,
    request: PnlCalibBatchRequest,
    image: np.ndarray,
    candidate: PitchCalibration | None,
    *,
    attempt_number: int,
    request_kind: str,
) -> tuple[PitchCalibration | None, dict, dict]:
    canonical = (
        canonicalize_penalty_side(candidate, image.shape[1])
        if candidate is not None
        else None
    )
    evidence = frame_calibration_evidence(
        scene,
        request.sample_index,
        request.scene_time,
        image,
        canonical,
        projection_source="direct" if canonical is not None else "none",
        pitch=scene["payload"]["pitch"],
        source_frame_index=request.source_frame_index,
    )
    attempt = {
        "attempt": attempt_number,
        "requestKind": request_kind,
        **calibration_attempt_payload(evidence),
    }
    return canonical, evidence, attempt


def _finish_resolution(
    evaluated: list[tuple[PitchCalibration | None, dict, dict]],
    *,
    retry_count: int,
    accepted_attempt: int | None,
) -> PnlCalibAttemptResolution:
    selected_index = max(
        range(len(evaluated)),
        key=lambda index: _candidate_rank(evaluated[index][0], evaluated[index][1]),
    )
    calibration, evidence, _ = evaluated[selected_index]
    attempts = tuple(
        {
            **attempt,
            "selected": index == selected_index,
        }
        for index, (_, _, attempt) in enumerate(evaluated)
    )
    evidence["pnlcalibAttempts"] = {
        "attemptCount": len(attempts),
        "maximumAttempts": 1 + retry_count,
        "acceptedAttempt": accepted_attempt,
        "attempts": [dict(item) for item in attempts],
    }
    return PnlCalibAttemptResolution(
        calibration=calibration,
        evidence=evidence,
        attempts=attempts,
        accepted_attempt=accepted_attempt,
    )


def resolve_pnlcalib_frame_attempts(
    scene: dict,
    *,
    sample_index: int,
    source_frame_index: int,
    scene_time: float,
    frame_path: Path,
    image: np.ndarray,
    initial_calibration: PitchCalibration | None,
    additional_attempts: int = 2,
    worker_timeout: float | None = None,
    on_retry: Callable[[int, int, str], None] | None = None,
) -> PnlCalibAttemptResolution:
    """Return the first QA-accepted solve, trying at most two fresh reruns.

    The initial candidate may come from either cache layer. Every retry calls
    the worker's explicit refresh endpoint with exactly one frame, so neither
    cached evidence nor other frames in an inference batch can fake a rerun.
    """

    retry_count = max(0, min(2, int(additional_attempts)))
    evaluated: list[tuple[PitchCalibration | None, dict, dict]] = []

    request = PnlCalibBatchRequest(
        sample_index=sample_index,
        source_frame_index=source_frame_index,
        scene_time=scene_time,
        frame_path=frame_path,
        initial_calibration=initial_calibration,
    )

    def evaluate(
        candidate: PitchCalibration | None,
        *,
        attempt_number: int,
        request_kind: str,
    ) -> bool:
        canonical, evidence, attempt = _evaluate_candidate(
            scene,
            request,
            image,
            candidate,
            attempt_number=attempt_number,
            request_kind=request_kind,
        )
        evaluated.append((canonical, evidence, attempt))
        return evidence.get("status") == "accepted"

    accepted = evaluate(
        initial_calibration,
        attempt_number=1,
        request_kind="initial-cache-aware",
    )
    accepted_attempt = 1 if accepted else None
    for retry_index in range(1, retry_count + 1):
        if accepted_attempt is not None:
            break
        attempt_number = retry_index + 1
        fresh = recalibrate_frames_with_worker(
            [(source_frame_index, frame_path)],
            timeout=worker_timeout,
        ).get(source_frame_index)
        accepted = evaluate(
            fresh,
            attempt_number=attempt_number,
            request_kind="forced-refresh-single-frame",
        )
        if on_retry is not None:
            on_retry(
                retry_index,
                retry_count,
                str(evaluated[-1][1].get("status") or "missing"),
            )
        if accepted:
            accepted_attempt = attempt_number

    return _finish_resolution(
        evaluated,
        retry_count=retry_count,
        accepted_attempt=accepted_attempt,
    )


def resolve_pnlcalib_batch_attempts(
    scene: dict,
    requests: list[PnlCalibBatchRequest],
    *,
    additional_attempts: int = 2,
    worker_timeout: float | None = None,
    on_retry_batch: Callable[[int, int, int, float, dict[str, int]], None]
    | None = None,
    on_retry_progress: Callable[
        [int, int, int, int, int, float, dict], None
    ]
    | None = None,
) -> dict[int, PnlCalibAttemptResolution]:
    """Retry all rejected frames in at most two batch inference rounds.

    The QA decision remains frame-local, but one round is one worker request.
    This preserves independent fresh inference while avoiding the previous
    N-frames x two 15-second request pattern.
    """

    retry_count = max(0, min(2, int(additional_attempts)))
    evaluated: dict[
        int, list[tuple[PitchCalibration | None, dict, dict]]
    ] = {}
    accepted_attempt: dict[int, int | None] = {}

    def evaluate(
        request: PnlCalibBatchRequest,
        candidate: PitchCalibration | None,
        *,
        attempt_number: int,
        request_kind: str,
    ) -> str:
        image = cv2.imread(str(request.frame_path))
        if image is None:
            raise ValueError(
                f"Could not decode sampled frame {request.frame_path.name} for PnLCalib QA"
            )
        result = _evaluate_candidate(
            scene,
            request,
            image,
            candidate,
            attempt_number=attempt_number,
            request_kind=request_kind,
        )
        evaluated.setdefault(request.sample_index, []).append(result)
        status = str(result[1].get("status") or "missing")
        if status == "accepted" and accepted_attempt.get(request.sample_index) is None:
            accepted_attempt[request.sample_index] = attempt_number
        return status

    for request in requests:
        accepted_attempt[request.sample_index] = None
        evaluate(
            request,
            request.initial_calibration,
            attempt_number=1,
            request_kind="initial-cache-aware",
        )

    pending = [
        request
        for request in requests
        if accepted_attempt[request.sample_index] is None
    ]
    for retry_index in range(1, retry_count + 1):
        if not pending:
            break
        started = perf_counter()

        def publish_batch_progress(batch: CalibrationWorkerBatchProgress) -> None:
            if on_retry_progress is None:
                return
            on_retry_progress(
                retry_index,
                retry_count,
                batch.completed,
                batch.total,
                batch.valid,
                batch.request_seconds,
                batch.diagnostics,
            )

        fresh_by_source = recalibrate_frames_with_worker(
            [
                (request.source_frame_index, request.frame_path)
                for request in pending
            ],
            on_batch=publish_batch_progress,
            timeout=worker_timeout,
        )
        status_counts: dict[str, int] = {}
        attempt_number = retry_index + 1
        for request in pending:
            status = evaluate(
                request,
                fresh_by_source.get(request.source_frame_index),
                attempt_number=attempt_number,
                request_kind=f"forced-refresh-batch-round-{retry_index}",
            )
            status_counts[status] = status_counts.get(status, 0) + 1
        elapsed = perf_counter() - started
        if on_retry_batch is not None:
            on_retry_batch(
                retry_index,
                retry_count,
                len(pending),
                elapsed,
                status_counts,
            )
        pending = [
            request
            for request in pending
            if accepted_attempt[request.sample_index] is None
        ]

    return {
        request.sample_index: _finish_resolution(
            evaluated[request.sample_index],
            retry_count=retry_count,
            accepted_attempt=accepted_attempt[request.sample_index],
        )
        for request in requests
    }


__all__ = (
    "PnlCalibAttemptResolution",
    "PnlCalibBatchRequest",
    "resolve_pnlcalib_batch_attempts",
    "resolve_pnlcalib_frame_attempts",
)
