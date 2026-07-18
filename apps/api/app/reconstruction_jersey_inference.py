"""Coordinate jersey crop materialization, OCR inference, and diagnostics."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import cv2
import numpy as np

from .jersey_ocr_contract import JerseyEvidenceSummary, JerseyOcrObservation
from .jersey_ocr_fusion import aggregate_tracklets
from .jersey_ocr_worker_client import (
    analyze_jersey_crops,
    jersey_ocr_worker_readiness,
)
from .jersey_ocr_worker_contract import JerseyCropRequest, JerseyOcrWorkerError
from .reconstruction_track_state import TrackState
from .reconstruction_inputs import source_frame_index
from .reconstruction_jersey_policy import (
    JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION,
    JERSEY_OCR_PRE_RESOLVER_FUSION_CONFIG,
    JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES,
)
from .reconstruction_jersey_sampling import (
    prospective_jersey_split_ranges,
    select_jersey_crop_points,
)


def _low_confidence_jersey_candidate(item: dict) -> tuple[str | None, float]:
    candidates = [
        candidate
        for candidate in item.get("candidates") or []
        if candidate.get("number") is not None
        and candidate.get("confidence") is not None
    ]
    if not candidates:
        return None, 0.0
    best = max(
        candidates,
        key=lambda candidate: (
            float(candidate["confidence"]),
            str(candidate["number"]),
        ),
    )
    return str(best["number"]), float(best["confidence"])


def run_jersey_ocr_for_tracklets(
    tracks: list[TrackState],
    frames: list[tuple[Path, float]],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    scene: dict | None = None,
) -> tuple[dict[str, JerseyEvidenceSummary], dict, list[str]]:
    """Extract optional OCR evidence without making reconstruction depend on it."""

    readiness = jersey_ocr_worker_readiness(timeout=2.0)
    diagnostics = {
        "schemaVersion": 1,
        **deepcopy(readiness),
        "requestedTrackletCount": len(tracks),
        "eligibleTrackletCount": 0,
        "candidateCropCount": 0,
        "selectedCropCount": 0,
        "selectionPartitionCount": 0,
        "prospectiveSplitRangeCount": 0,
        "submittedCropCount": 0,
        "cropReadFailureCount": 0,
        "recognizedCropCount": 0,
        "lowConfidenceCropCount": 0,
        "ambiguousCropCount": 0,
        "rejectedCropCount": 0,
        "backVisibilityAvailable": False,
        "cropCandidatePolicy": "bounded-per-prospective-partition-v2",
        "maxCropsPerProspectivePartition": (
            JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION
        ),
        "preResolverMaxSelectedFrames": (
            JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES
        ),
        "trackletEvidence": {},
        "crops": [],
    }
    if readiness.get("status") != "ready":
        warnings = []
        if readiness.get("status") not in {"disabled", "no-observations"}:
            warnings.append(
                "Jersey OCR is unavailable; reconstruction continued without shirt-number identity evidence."
            )
        return {}, diagnostics, warnings

    frame_by_index = {
        source_frame_index(Path(path)): Path(path)
        for path, _ in frames
    }
    selected: list[tuple[str, dict, float]] = []
    for track in tracks:
        tracklet_id = track.local_tracklet_id
        prospective_ranges = prospective_jersey_split_ranges(track, scene)
        points, candidate_count, partition_count = select_jersey_crop_points(
            track,
            set(frame_by_index),
            prospective_ranges,
        )
        diagnostics["candidateCropCount"] += candidate_count
        diagnostics["selectionPartitionCount"] += partition_count
        diagnostics["prospectiveSplitRangeCount"] += len(prospective_ranges)
        if candidate_count:
            diagnostics["eligibleTrackletCount"] += 1
        selected.extend((tracklet_id, point, quality) for point, quality in points)
    diagnostics["selectedCropCount"] = len(selected)
    if not selected:
        diagnostics["status"] = "no-crops"
        return {}, diagnostics, []

    requests: list[JerseyCropRequest] = []
    request_metadata: dict[str, dict] = {}
    image_cache: dict[int, np.ndarray | None] = {}
    with TemporaryDirectory(prefix="replay-jersey-ocr-") as directory:
        crop_root = Path(directory)
        for request_index, (tracklet_id, point, selection_quality) in enumerate(selected):
            frame_index = int(point["frameIndex"])
            if frame_index not in image_cache:
                image_cache[frame_index] = cv2.imread(str(frame_by_index[frame_index]))
            image = image_cache[frame_index]
            bbox = point["bbox"]
            if image is None:
                diagnostics["cropReadFailureCount"] += 1
                continue
            image_height, image_width = image.shape[:2]
            x1 = max(0, min(image_width, int(np.floor(float(bbox["x"])))))
            y1 = max(0, min(image_height, int(np.floor(float(bbox["y"])))))
            x2 = max(0, min(image_width, int(np.ceil(float(bbox["x"]) + float(bbox["width"])))))
            y2 = max(0, min(image_height, int(np.ceil(float(bbox["y"]) + float(bbox["height"])))))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                diagnostics["cropReadFailureCount"] += 1
                continue
            observation_id = str(
                point.get("observationId") or f"{tracklet_id}:{frame_index}"
            )
            crop_id = (
                "jersey-"
                + sha256(
                    f"{tracklet_id}:{observation_id}:{frame_index}".encode("utf-8")
                ).hexdigest()[:16]
            )
            crop_path = crop_root / f"crop-{request_index:04d}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                diagnostics["cropReadFailureCount"] += 1
                continue
            request = JerseyCropRequest(
                crop_id=crop_id,
                path=crop_path,
                observation_id=observation_id,
                tracklet_id=tracklet_id,
                frame_index=frame_index,
                timestamp=float(point.get("t") or 0.0),
            )
            requests.append(request)
            request_metadata[crop_id] = {
                "selectionQuality": round(float(selection_quality), 6),
                "clippedCropRatio": round(
                    (max(0, x2 - x1) * max(0, y2 - y1))
                    / max(1.0, float(bbox["width"]) * float(bbox["height"])),
                    6,
                ),
            }
        diagnostics["submittedCropCount"] = len(requests)
        if not requests:
            diagnostics["status"] = "no-readable-crops"
            return {}, diagnostics, []
        try:
            worker_results = analyze_jersey_crops(requests, on_progress)
        except JerseyOcrWorkerError as exc:
            diagnostics.update({"status": "failed", "detail": str(exc)})
            return (
                {},
                diagnostics,
                [
                    "Jersey OCR failed during crop analysis; reconstruction continued without shirt-number identity evidence."
                ],
            )

    worker_cache_diagnostics = deepcopy(worker_results.diagnostics)
    worker_model_contract = worker_cache_diagnostics.get("modelContract")

    observations: list[JerseyOcrObservation] = []
    requests_by_id = {request.crop_id: request for request in requests}
    status_counts: dict[str, int] = {}
    for crop_id, item in sorted(worker_results.items_by_crop_id.items()):
        request = requests_by_id[crop_id]
        status = str(item.get("status") or "rejected")
        status_counts[status] = status_counts.get(status, 0) + 1
        raw_number: str | None = None
        ocr_confidence = 0.0
        if status == "recognized":
            raw_number = str(item["number"])
            ocr_confidence = float(item["confidence"])
        elif status == "low-confidence":
            raw_number, ocr_confidence = _low_confidence_jersey_candidate(item)
        frame_quality = 1.0 if item.get("usable") is not False else 0.0
        back_visibility = 1.0
        source = str(item.get("provider") or "jersey-ocr-worker")
        observations.append(
            JerseyOcrObservation(
                id=crop_id,
                tracklet_id=str(request.tracklet_id),
                timestamp_seconds=float(request.timestamp or 0.0),
                raw_number=raw_number,
                ocr_confidence=ocr_confidence,
                # Worker-side crop QA is the authoritative readability gate.
                frame_quality=frame_quality,
                # No reliable front/back classifier exists in v1. Keep this
                # neutral and make the missing signal explicit in diagnostics.
                back_visibility=back_visibility,
                frame_index=request.frame_index,
                source=source,
                evidence_fingerprint=str(item.get("evidenceFingerprint") or "") or None,
            )
        )
        diagnostics["crops"].append(
            {
                "cropId": crop_id,
                "observationId": request.observation_id,
                "trackletId": request.tracklet_id,
                "frameIndex": request.frame_index,
                "timestamp": request.timestamp,
                "status": status,
                # Normalized raw evidence is deliberately retained even when
                # the pre-resolver top-N does not select this crop.  A manual
                # split can later give it a different final owner.
                "rawNumber": raw_number,
                "ocrConfidence": round(float(ocr_confidence), 6),
                "frameQuality": frame_quality,
                "backVisibility": back_visibility,
                "source": source,
                "evidenceFingerprint": item.get("evidenceFingerprint"),
                "number": item.get("number"),
                "confidence": item.get("confidence"),
                "candidates": deepcopy(item.get("candidates") or []),
                "quality": deepcopy(item.get("quality") or {}),
                "rejectionReasons": list(item.get("rejectionReasons") or []),
                "decisionReasons": list(item.get("decisionReasons") or []),
                **request_metadata[crop_id],
            }
        )

    summaries = aggregate_tracklets(
        observations,
        config=JERSEY_OCR_PRE_RESOLVER_FUSION_CONFIG,
    )
    diagnostics.update(
        {
            "status": "ready",
            "provider": (
                worker_model_contract.get("backend")
                if isinstance(worker_model_contract, dict)
                else next(
                    (
                        item.get("provider")
                            for item in worker_results.items_by_crop_id.values()
                        if item.get("provider")
                    ),
                    readiness.get("backend"),
                )
            ),
            "modelVersion": (
                worker_model_contract.get("modelVersion")
                if isinstance(worker_model_contract, dict)
                else next(
                    (
                        item.get("modelVersion")
                            for item in worker_results.items_by_crop_id.values()
                        if item.get("modelVersion")
                    ),
                    readiness.get("modelVersion"),
                )
            ),
            **(
                {"modelContract": deepcopy(worker_model_contract)}
                if isinstance(worker_model_contract, dict)
                else {}
            ),
            "recognizedCropCount": status_counts.get("recognized", 0),
            "lowConfidenceCropCount": status_counts.get("low-confidence", 0),
            "ambiguousCropCount": status_counts.get("ambiguous", 0),
            "rejectedCropCount": status_counts.get("rejected", 0),
            "noNumberCropCount": status_counts.get("no-number", 0),
            "rawObservationCount": len(observations),
            "rawUsableObservationCount": sum(
                observation.raw_number is not None
                and observation.frame_quality > 0.0
                for observation in observations
            ),
            "preResolverSelectedCropCount": sum(
                summary.selected_sample_count for summary in summaries.values()
            ),
            "reliableTrackletCount": sum(
                summary.status == "reliable" for summary in summaries.values()
            ),
            "provisionalTrackletCount": sum(
                summary.status == "provisional" for summary in summaries.values()
            ),
            "conflictingTrackletCount": sum(
                summary.status == "conflict" for summary in summaries.values()
            ),
            "trackletEvidence": {
                tracklet_id: summary.to_payload()
                for tracklet_id, summary in sorted(summaries.items())
            },
            "cacheHitCount": int(worker_cache_diagnostics.get("cacheHitCount") or 0),
            "providerInferenceCropCount": int(
                worker_cache_diagnostics.get("providerInferenceCropCount") or 0
            ),
            "requestDeduplicatedCount": int(
                worker_cache_diagnostics.get("requestDeduplicatedCount") or 0
            ),
            "uniqueEvidenceFingerprintCount": int(
                worker_cache_diagnostics.get("uniqueEvidenceFingerprintCount") or 0
            ),
            "duplicateEvidenceFingerprintCount": int(
                worker_cache_diagnostics.get("duplicateEvidenceFingerprintCount") or 0
            ),
            "cacheEnabled": worker_cache_diagnostics.get("cacheEnabled"),
        }
    )
    return summaries, diagnostics, []
