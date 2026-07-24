from __future__ import annotations

"""Retry locally valid PnLCalib anchors rejected by shot-level p95 QA."""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from .calibration_worker import (
    CalibrationWorkerBatchProgress,
    recalibrate_frames_with_worker,
)
from .pitch_calibration_orientation import canonicalize_penalty_side
from .reconstruction_calibration_evidence import (
    calibration_attempt_payload,
    frame_calibration_evidence,
)
from .reconstruction_calibration_resolution import demote_outlier_direct_anchors
from .reconstruction_inputs import source_frame_index
from .reconstruction_sampled_frame_contract import SampledCalibrationAnalysis


@dataclass(frozen=True)
class PnlCalibDemotionRetryResult:
    analysis: SampledCalibrationAnalysis
    retried_frame_count: int
    forced_attempt_count: int
    recovered_frame_count: int


def _attempts(evidence: dict) -> list[dict]:
    audit = evidence.get("pnlcalibAttempts") or {}
    return [dict(item) for item in audit.get("attempts") or []]


def _with_attempt_audit(
    evidence: dict,
    *,
    previous_evidence: dict,
    attempt_number: int,
    selected: bool,
    maximum_attempts: int,
) -> dict:
    previous_attempts = _attempts(previous_evidence)
    attempts = [
        {**item, "selected": False if selected else bool(item.get("selected"))}
        for item in previous_attempts
    ]
    attempts.append(
        {
            "attempt": attempt_number,
            "requestKind": "forced-refresh-batch-after-p95-demotion",
            **calibration_attempt_payload(evidence),
            "selected": selected,
        }
    )
    evidence["pnlcalibAttempts"] = {
        "attemptCount": len(attempts),
        "maximumAttempts": maximum_attempts,
        "acceptedAttempt": (
            attempt_number if evidence.get("status") == "accepted" else None
        ),
        "attempts": attempts,
    }
    return evidence


def retry_demoted_pnlcalib_anchors(
    scene: dict,
    frames: list[tuple[Path, float]],
    analysis: SampledCalibrationAnalysis,
    *,
    additional_attempts: int,
    residual_floor_pixels: float,
    best_quartile_ratio: float,
    max_gap_seconds: float,
    on_retry: Callable[[int, int, str], None] | None = None,
    on_retry_batch: Callable[[int, int, int, float, dict[str, int]], None]
    | None = None,
    on_retry_progress: Callable[
        [int, int, int, int, int, float, dict], None
    ]
    | None = None,
) -> PnlCalibDemotionRetryResult:
    """Use remaining per-frame attempts before an outlier becomes temporal."""

    retry_limit = max(0, min(2, int(additional_attempts)))
    if retry_limit == 0:
        return PnlCalibDemotionRetryResult(analysis, 0, 0, 0)

    automatic = dict(analysis.accepted_automatic_direct_by_sample)
    accepted_frames = dict(analysis.accepted_frame_calibrations)
    evidence = list(analysis.frame_evidence)
    initially_demoted: set[int] = set()
    retried: set[int] = set()
    forced_attempt_count = 0

    for _round in range(retry_limit):
        # The existing demotion function annotates evidence. Probe with shallow
        # copies; only the final temporal pass may mutate the published rows.
        _, demotions = demote_outlier_direct_anchors(
            dict(automatic),
            [dict(item) for item in evidence],
            frames,
            manual_direct=dict(analysis.accepted_manual_direct_by_sample),
            max_gap_seconds=max_gap_seconds,
            residual_floor_pixels=residual_floor_pixels,
            best_quartile_ratio=best_quartile_ratio,
        )
        targets = [int(item["sampleIndex"]) for item in demotions]
        initially_demoted.update(targets)
        actionable = []
        for sample_index in targets:
            used = max(0, len(_attempts(evidence[sample_index])) - 1)
            if used < retry_limit:
                actionable.append((sample_index, used))
        if not actionable:
            break

        started = perf_counter()

        def publish_batch_progress(batch: CalibrationWorkerBatchProgress) -> None:
            if on_retry_progress is None:
                return
            on_retry_progress(
                _round + 1,
                retry_limit,
                batch.completed,
                batch.total,
                batch.valid,
                batch.request_seconds,
                batch.diagnostics,
            )

        fresh_by_source = recalibrate_frames_with_worker(
            [
                (source_frame_index(frames[sample_index][0]), frames[sample_index][0])
                for sample_index, _ in actionable
            ],
            on_batch=publish_batch_progress,
        )
        status_counts: dict[str, int] = {}
        for sample_index, used in actionable:
            path, scene_time = frames[sample_index]
            image = cv2.imread(str(path))
            if image is None:
                continue
            source_index = source_frame_index(path)
            fresh = fresh_by_source.get(source_index)
            canonical = (
                canonicalize_penalty_side(fresh, image.shape[1])
                if fresh is not None
                else None
            )
            fresh_evidence = frame_calibration_evidence(
                scene,
                sample_index,
                scene_time,
                image,
                canonical,
                projection_source="direct" if canonical is not None else "none",
                pitch=scene["payload"]["pitch"],
                source_frame_index=source_index,
            )
            previous_evidence = evidence[sample_index]
            if previous_evidence.get("cameraMotion") is not None:
                fresh_evidence["cameraMotion"] = previous_evidence["cameraMotion"]
            attempt_number = len(_attempts(previous_evidence)) + 1
            forced_attempt_count += 1
            retried.add(sample_index)
            has_attempts_left = used + 1 < retry_limit
            accepted = fresh_evidence.get("status") == "accepted"
            status = str(fresh_evidence.get("status") or "missing")
            status_counts[status] = status_counts.get(status, 0) + 1
            if accepted:
                evidence[sample_index] = _with_attempt_audit(
                    fresh_evidence,
                    previous_evidence=previous_evidence,
                    attempt_number=attempt_number,
                    selected=True,
                    maximum_attempts=1 + retry_limit,
                )
                assert canonical is not None
                automatic[sample_index] = canonical
                accepted_frames[source_index] = canonical
            elif has_attempts_left:
                # Keep the globally rejected local candidate only as the input
                # to the next p95 probe; append the failed fresh attempt to its
                # audit without publishing it as the selected observation.
                retained = dict(previous_evidence)
                retained["pnlcalibAttempts"] = _with_attempt_audit(
                    dict(fresh_evidence),
                    previous_evidence=previous_evidence,
                    attempt_number=attempt_number,
                    selected=False,
                    maximum_attempts=1 + retry_limit,
                )["pnlcalibAttempts"]
                evidence[sample_index] = retained
            else:
                evidence[sample_index] = _with_attempt_audit(
                    fresh_evidence,
                    previous_evidence=previous_evidence,
                    attempt_number=attempt_number,
                    selected=True,
                    maximum_attempts=1 + retry_limit,
                )
                automatic.pop(sample_index, None)
                accepted_frames.pop(source_index, None)
            if on_retry is not None:
                on_retry(
                    used + 1,
                    retry_limit,
                    status,
                )
        if on_retry_batch is not None:
            on_retry_batch(
                _round + 1,
                retry_limit,
                len(actionable),
                perf_counter() - started,
                status_counts,
            )

    _, final_demotions = demote_outlier_direct_anchors(
        dict(automatic),
        [dict(item) for item in evidence],
        frames,
        manual_direct=dict(analysis.accepted_manual_direct_by_sample),
        max_gap_seconds=max_gap_seconds,
        residual_floor_pixels=residual_floor_pixels,
        best_quartile_ratio=best_quartile_ratio,
    )
    finally_demoted = {int(item["sampleIndex"]) for item in final_demotions}
    recovered = len(
        {
            sample
            for sample in initially_demoted
            if sample in automatic and sample not in finally_demoted
        }
    )
    return PnlCalibDemotionRetryResult(
        analysis=replace(
            analysis,
            accepted_frame_calibrations=accepted_frames,
            accepted_automatic_direct_by_sample=automatic,
            frame_evidence=evidence,
            rejected_frame_count=sum(
                1
                for item in evidence
                if item.get("status") == "rejected"
            ),
        ),
        retried_frame_count=len(retried),
        forced_attempt_count=forced_attempt_count,
        recovered_frame_count=recovered,
    )


__all__ = ("PnlCalibDemotionRetryResult", "retry_demoted_pnlcalib_anchors")
