"""Detector observation identity and ReID embedding evidence."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from .reconstruction_errors import ReconstructionError
from .reconstruction_person_detection_contract import Detection
from .reconstruction_inputs import source_frame_index


def capture_detection_observations(
    detections: list[Detection],
    source_frame_index: int,
) -> None:
    """Freeze detector-space evidence before stabilization mutates x/y."""

    generated_ids: set[str] = set()
    for detection in detections:
        detection.source_frame_index = int(source_frame_index)
        detection.image_x = float(detection.x)
        detection.image_y = float(detection.y)
        if not detection.observation_id:
            if detection.annotation_id:
                stable_source = f"annotation-{detection.annotation_id}"
            else:
                feature = np.asarray(
                    detection.feature,
                    dtype=np.float32,
                ).reshape(-1)
                fingerprint = "|".join(
                    [
                        "person-observation-v2",
                        str(int(source_frame_index)),
                        f"{float(detection.image_x):.6f}",
                        f"{float(detection.image_y):.6f}",
                        f"{float(detection.width):.6f}",
                        f"{float(detection.height):.6f}",
                        f"{float(detection.confidence):.6f}",
                        feature.tobytes().hex(),
                    ]
                )
                digest = sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
                stable_source = f"observation-{digest}"
            observation_id = (
                f"frame-{int(source_frame_index):06d}:{stable_source}"
            )
            if observation_id in generated_ids:
                raise ReconstructionError(
                    "Detector produced indistinguishable duplicate person "
                    "observations; identity evidence was rejected instead of "
                    "assigning order-based IDs"
                )
            detection.observation_id = observation_id
        if str(detection.observation_id) in generated_ids:
            raise ReconstructionError(
                f"Duplicate person observationId: {detection.observation_id}"
            )
        generated_ids.add(str(detection.observation_id))


def _bbox_iou(first: dict, second: dict) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = (
        first["width"] * first["height"]
        + second["width"] * second["height"]
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def _local_rejected_item(
    observation: dict,
    frame_index: int,
    reasons: list[str],
) -> dict:
    """A rejected result produced without a worker round-trip."""

    return {
        "observationId": str(observation["observationId"]),
        "frameIndex": int(frame_index),
        "usable": False,
        "quality": dict(observation.get("quality") or {}),
        "rejectionReasons": list(reasons),
        "embedding": None,
        "visibilityScores": None,
        "role": None,
        "roleConfidence": None,
        "evidenceFingerprint": None,
        "cacheHit": False,
        "cacheSource": "crop-store",
    }


def identity_embedding_requests(
    frames: list[tuple[Path, float]],
    person_frames: list[tuple[list[Detection], float]],
    *,
    overlap_iou_threshold: float = 0.0,
) -> tuple[list[tuple[int, str, list[dict]]], dict[str, dict], dict]:
    """Build per-frame crop requests, resolving QA rejections locally.

    The detection pass already cut every crop into the person crop store, so
    a request references the crop digest instead of frame pixels. Crops that
    failed extraction QA (or never reached the store) become local rejected
    items without a worker round-trip. A crop whose bbox is strongly
    overlapped by another detection contains two players: its embedding is
    noise that can hard-break the online tracker's ReID gate, so it is
    skipped explicitly and reported per observation. A threshold of 0
    disables the overlap filter.
    """

    requests: list[tuple[int, str, list[dict]]] = []
    local_items: dict[str, dict] = {}
    skipped: list[dict] = []
    total_skipped = 0
    crop_rejected_count = 0
    store_unavailable_count = 0
    for (path, _), (people, _) in zip(frames, person_frames):
        observations = []
        for person in people:
            if not person.observation_id:
                continue
            image_x = person.image_x if person.image_x is not None else person.x
            image_y = person.image_y if person.image_y is not None else person.y
            observations.append(
                {
                    "observationId": person.observation_id,
                    "bbox": {
                        "x": float(image_x) - person.width / 2,
                        "y": float(image_y) - person.height,
                        "width": person.width,
                        "height": person.height,
                    },
                    "cropSha256": person.crop_sha256,
                    "cropFrameSha256": person.crop_frame_sha256,
                    "quality": dict(person.crop_quality or {}),
                    "cropRejectionReasons": list(
                        person.crop_rejection_reasons or ()
                    ),
                }
            )
        if overlap_iou_threshold > 0.0 and len(observations) > 1:
            kept: list[dict] = []
            for observation in observations:
                worst_overlap = max(
                    (
                        _bbox_iou(observation["bbox"], other["bbox"])
                        for other in observations
                        if other is not observation
                    ),
                    default=0.0,
                )
                if worst_overlap > overlap_iou_threshold:
                    if len(skipped) < 40:
                        skipped.append(
                            {
                                "observationId": observation["observationId"],
                                "overlapIou": round(worst_overlap, 4),
                            }
                        )
                    continue
                kept.append(observation)
            total_skipped += len(observations) - len(kept)
            observations = kept
        frame_index = source_frame_index(path)
        sendable: list[dict] = []
        frame_sha: str | None = None
        for observation in observations:
            reasons = observation["cropRejectionReasons"]
            if reasons:
                crop_rejected_count += 1
                local_items[str(observation["observationId"])] = (
                    _local_rejected_item(observation, frame_index, reasons)
                )
                continue
            if not observation["cropSha256"] or not observation["cropFrameSha256"]:
                store_unavailable_count += 1
                local_items[str(observation["observationId"])] = (
                    _local_rejected_item(
                        observation,
                        frame_index,
                        ["crop-store-unavailable"],
                    )
                )
                continue
            frame_sha = str(observation["cropFrameSha256"])
            sendable.append(
                {
                    "observationId": observation["observationId"],
                    "cropSha256": observation["cropSha256"],
                    "quality": observation["quality"],
                }
            )
        if sendable and frame_sha:
            requests.append((frame_index, frame_sha, sendable))
    diagnostics = {
        "overlapIouThreshold": float(overlap_iou_threshold),
        "overlapSkippedObservationCount": total_skipped,
        "overlapSkippedObservations": skipped,
        "cropRejectedObservationCount": crop_rejected_count,
        "cropStoreUnavailableObservationCount": store_unavailable_count,
    }
    return requests, local_items, diagnostics


def attach_identity_embeddings(
    person_frames: list[tuple[list[Detection], float]],
    results: dict[str, dict],
    worker_diagnostics: dict | None = None,
) -> dict:
    requested = usable = rejected = 0
    provider = model_version = None
    role_observations = 0
    crop_diagnostics: list[dict] = []
    for observation_id, item in sorted(results.items()):
        is_usable = item.get("usable") is True
        crop_diagnostics.append(
            {
                "observationId": str(observation_id),
                "frameIndex": item.get("frameIndex"),
                "status": "usable" if is_usable else "rejected",
                "usable": is_usable,
                "quality": deepcopy(item.get("quality") or {}),
                "rejectionReasons": list(item.get("rejectionReasons") or []),
                "evidenceFingerprint": item.get("evidenceFingerprint"),
                "provider": item.get("provider"),
                "modelVersion": item.get("modelVersion"),
                "role": item.get("role") if is_usable else None,
                "roleConfidence": (
                    item.get("roleConfidence") if is_usable else None
                ),
            }
        )
    for people, _ in person_frames:
        for person in people:
            item = results.get(str(person.observation_id or ""))
            if item is None:
                continue
            requested += 1
            provider = item.get("provider") or provider
            model_version = item.get("modelVersion") or model_version
            person.reid_quality = deepcopy(item.get("quality") or {})
            person.reid_evidence_fingerprint = (
                str(item.get("evidenceFingerprint") or "") or None
            )
            if item.get("usable") is not True:
                rejected += 1
                continue
            vector = np.asarray(item.get("embedding"), dtype=np.float32)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                rejected += 1
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                rejected += 1
                continue
            person.reid_feature = vector / norm
            person.reid_role = item.get("role")
            person.reid_role_confidence = item.get("roleConfidence")
            role_observations += int(person.reid_role is not None)
            usable += 1
    diagnostics = deepcopy(worker_diagnostics or {})
    model_contract = diagnostics.get("modelContract")
    if isinstance(model_contract, dict):
        provider = model_contract.get("backend") or provider
        model_version = model_contract.get("modelVersion") or model_version
    return {
        "status": "ready" if requested else "no-observations",
        "provider": provider,
        "modelVersion": model_version,
        **(
            {"modelContract": deepcopy(model_contract)}
            if isinstance(model_contract, dict)
            else {}
        ),
        "requestedObservationCount": requested,
        "usableObservationCount": usable,
        "rejectedObservationCount": rejected,
        "roleObservationCount": role_observations,
        "usableCropRatio": round(usable / max(1, requested), 3),
        "cacheHitCount": int(diagnostics.get("cacheHitCount") or 0),
        "cacheMissCount": int(diagnostics.get("cacheMissCount") or 0),
        "deduplicatedObservationCount": int(
            diagnostics.get("deduplicatedObservationCount") or 0
        ),
        "uniqueEvidenceFingerprintCount": int(
            diagnostics.get("uniqueEvidenceFingerprintCount") or 0
        ),
        "duplicateEvidenceFingerprintCount": int(
            diagnostics.get("duplicateEvidenceFingerprintCount") or 0
        ),
        "concurrentDeduplicatedCount": int(
            diagnostics.get("concurrentDeduplicatedCount") or 0
        ),
        "providerInferenceCount": int(
            diagnostics.get("providerInferenceCount") or 0
        ),
        "crops": crop_diagnostics,
        "corruptCacheMissCount": int(
            diagnostics.get("corruptCacheMissCount") or 0
        ),
        "expiredCacheMissCount": int(
            diagnostics.get("expiredCacheMissCount") or 0
        ),
        **(
            {"cache": diagnostics["cache"]}
            if "cache" in diagnostics
            else {}
        ),
    }
