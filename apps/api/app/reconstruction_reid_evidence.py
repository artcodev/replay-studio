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


def identity_embedding_requests(
    frames: list[tuple[Path, float]],
    person_frames: list[tuple[list[Detection], float]],
) -> list[tuple[int, Path, list[dict]]]:
    requests: list[tuple[int, Path, list[dict]]] = []
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
                }
            )
        if observations:
            requests.append((source_frame_index(path), path, observations))
    return requests


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
