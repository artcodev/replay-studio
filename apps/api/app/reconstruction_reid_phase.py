from __future__ import annotations

"""Extract remote ReID evidence after sampled person detections are complete."""

from copy import deepcopy
from pathlib import Path

from .config import get_settings
from .identity_worker_client import (
    embed_identity_frames,
    identity_worker_readiness,
)
from .identity_worker_contract import IdentityWorkerError
from .reconstruction_person_detection_contract import Detection
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_reid_evidence import (
    attach_identity_embeddings,
    identity_embedding_requests,
)


def extract_reid_evidence(
    frames: list[tuple[Path, float]],
    person_frames: list[tuple[list[Detection], float]],
    progress: ReconstructionProgress,
) -> tuple[dict, list[str]]:
    requests, local_items, overlap_diagnostics = identity_embedding_requests(
        frames,
        person_frames,
        overlap_iou_threshold=float(
            get_settings().identity_crop_overlap_iou_threshold
        ),
    )
    overlap_fields = (
        {"overlapFiltered": overlap_diagnostics}
        if overlap_diagnostics.get("overlapSkippedObservationCount")
        or overlap_diagnostics.get("cropRejectedObservationCount")
        or overlap_diagnostics.get("cropStoreUnavailableObservationCount")
        else {}
    )
    if not requests and not local_items:
        return (
            {
                "status": "no-observations",
                "provider": "prtreid-bpbreid-soccernet",
                "requestedObservationCount": 0,
                "usableObservationCount": 0,
                "rejectedObservationCount": 0,
                "usableCropRatio": 0.0,
                "crops": [],
                **overlap_fields,
            },
            [],
        )
    if not requests:
        # Every crop was rejected at extraction: the verdicts are complete
        # without a worker round-trip.
        attached = attach_identity_embeddings(person_frames, local_items, {})
        return ({**attached, **overlap_fields}, [])
    worker_status = identity_worker_readiness(timeout=2.0)
    if worker_status.get("status") != "ready":
        return (
            deepcopy(worker_status),
            [
                "PRTReID identity worker is unavailable; local tracklets remain "
                "provisional and are not auto-merged across gaps."
            ],
        )
    progress.update(
        "detection",
        3,
        "Extracting player identity evidence",
        f"PRTReID is evaluating {sum(len(item[2]) for item in requests)} player crops.",
        56,
        62,
        completed=0,
        total=len(requests),
    )

    def identity_progress(completed: int, total: int, usable: int) -> None:
        progress.update(
            "detection",
            3,
            "Extracting player identity evidence",
            f"PRTReID frames {completed}/{total} · {usable} usable player crops.",
            56,
            62,
            completed=completed,
            total=total,
            eta_padding=2.0,
        )

    try:
        results = embed_identity_frames(requests, identity_progress)
    except IdentityWorkerError as exc:
        return (
            {**worker_status, "status": "failed", "detail": str(exc)},
            [
                "PRTReID identity extraction failed; automatic cross-gap identity "
                "merging was disabled."
            ],
        )
    partial_failure = results.diagnostics.get("partialFailure")
    if partial_failure is not None and not results.items_by_observation_id:
        return (
            {
                **worker_status,
                "status": "failed",
                "detail": str(partial_failure.get("detail") or ""),
            },
            [
                "PRTReID identity extraction failed; automatic cross-gap identity "
                "merging was disabled."
            ],
        )
    attached = attach_identity_embeddings(
        person_frames,
        {**local_items, **results.items_by_observation_id},
        results.diagnostics,
    )
    if partial_failure is not None:
        processed = int(partial_failure.get("processedFrameCount") or 0)
        requested = int(partial_failure.get("requestedFrameCount") or 0)
        return (
            {
                **worker_status,
                **attached,
                **overlap_fields,
                "status": "partial",
                "partialFailure": deepcopy(partial_failure),
            },
            [
                f"PRTReID embedded only {processed}/{requested} frames before the "
                "worker became unavailable; unmatched tracklets remain provisional."
            ],
        )
    return ({**worker_status, **attached, **overlap_fields}, [])


__all__ = ["extract_reid_evidence"]
