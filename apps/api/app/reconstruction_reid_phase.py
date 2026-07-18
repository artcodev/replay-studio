from __future__ import annotations

"""Extract remote ReID evidence after sampled person detections are complete."""

from copy import deepcopy
from pathlib import Path

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
    requests = identity_embedding_requests(frames, person_frames)
    if not requests:
        return (
            {
                "status": "no-observations",
                "provider": "prtreid-bpbreid-soccernet",
                "requestedObservationCount": 0,
                "usableObservationCount": 0,
                "rejectedObservationCount": 0,
                "usableCropRatio": 0.0,
                "crops": [],
            },
            [],
        )
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
        82,
        84,
        completed=0,
        total=len(requests),
    )

    def identity_progress(completed: int, total: int, usable: int) -> None:
        progress.update(
            "detection",
            3,
            "Extracting player identity evidence",
            f"PRTReID frames {completed}/{total} · {usable} usable player crops.",
            82,
            84,
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
    return (
        {
            **worker_status,
            **attach_identity_embeddings(
                person_frames,
                results.items_by_observation_id,
                results.diagnostics,
            ),
        },
        [],
    )


__all__ = ["extract_reid_evidence"]
