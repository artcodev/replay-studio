from __future__ import annotations

"""Bounded ReID evidence reservoir and role voting for a person track."""

from math import log1p, sqrt

import numpy as np

from .reconstruction_person_detection_contract import Detection
from .reconstruction_track_state import TrackState


def rebuild_track_reid_reservoir(track: TrackState) -> None:
    """Select quality-ranked, genuinely time-separated ReID views."""

    selected: list[dict] = []
    selected_fingerprints: set[str] = set()
    for candidate in sorted(
        track.reid_sample_candidates,
        key=lambda item: (
            -float(item["quality"]),
            float(item["time"]),
            int(item["frameIndex"]),
        ),
    ):
        fingerprint = str(candidate.get("evidenceFingerprint") or "")
        if fingerprint and fingerprint in selected_fingerprints:
            continue
        if any(
            abs(float(candidate["time"]) - float(previous["time"])) < 0.45
            for previous in selected
        ):
            continue
        selected.append(candidate)
        if fingerprint:
            selected_fingerprints.add(fingerprint)
        if len(selected) >= 12:
            break
    selected.sort(key=lambda item: (float(item["time"]), int(item["frameIndex"])))
    track.reid_samples = [item["vector"].copy() for item in selected]
    track.reid_selected_metadata = [
        {
            "time": float(item["time"]),
            "frameIndex": int(item["frameIndex"]),
            "quality": float(item["quality"]),
            "evidenceFingerprint": item.get("evidenceFingerprint"),
        }
        for item in selected
    ]
    track.reid_feature_count = len(track.reid_samples)
    track.reid_feature_sum = (
        np.sum(np.stack(track.reid_samples), axis=0)
        if track.reid_samples
        else None
    )


def _add_reid_sample(
    track: TrackState,
    vector: np.ndarray,
    detection: Detection,
    frame_index: int,
    time: float,
) -> bool:
    quality = detection.reid_quality or {}
    crop_width = max(0.0, float(quality.get("cropWidth") or detection.width))
    crop_height = max(0.0, float(quality.get("cropHeight") or detection.height))
    crop_area_score = min(1.0, sqrt(crop_width * crop_height) / 120.0)
    sharpness = max(0.0, float(quality.get("sharpness") or 0.0))
    sharpness_score = min(1.0, log1p(sharpness) / log1p(500.0))
    detector_score = max(0.0, min(1.0, float(detection.confidence)))
    quality_score = (
        0.42 * detector_score
        + 0.33 * crop_area_score
        + 0.25 * sharpness_score
        - (0.08 if quality.get("borderClipped") else 0.0)
    )
    temporal_bin = int(float(time) / 0.25)
    observation_id = str(
        detection.observation_id or f"{track.local_tracklet_id}:{int(frame_index)}"
    )
    candidate = {
        "time": float(time),
        "frameIndex": int(
            detection.source_frame_index
            if detection.source_frame_index is not None
            else frame_index
        ),
        "quality": round(max(0.0, min(1.0, quality_score)), 6),
        "temporalBin": temporal_bin,
        "observationId": observation_id,
        "evidenceFingerprint": str(
            detection.reid_evidence_fingerprint or "observation:" + observation_id
        ),
        "vector": vector.copy(),
    }
    track.reid_observation_ids.add(observation_id)
    evidence_fingerprint = str(candidate["evidenceFingerprint"])
    if evidence_fingerprint in track.reid_evidence_fingerprints:
        track.reid_duplicate_evidence_count += 1
        track.reid_observation_count = len(track.reid_evidence_fingerprints)
        return False
    track.reid_evidence_fingerprints.add(evidence_fingerprint)
    existing_index = next(
        (
            index
            for index, item in enumerate(track.reid_sample_candidates)
            if int(item["temporalBin"]) == temporal_bin
        ),
        None,
    )
    if existing_index is None:
        track.reid_sample_candidates.append(candidate)
    elif (candidate["quality"], -candidate["frameIndex"]) > (
        track.reid_sample_candidates[existing_index]["quality"],
        -track.reid_sample_candidates[existing_index]["frameIndex"],
    ):
        track.reid_sample_candidates[existing_index] = candidate
    track.reid_sample_candidates = sorted(
        track.reid_sample_candidates,
        key=lambda item: (-float(item["quality"]), int(item["frameIndex"])),
    )[:64]
    track.reid_observation_count = len(track.reid_evidence_fingerprints)
    rebuild_track_reid_reservoir(track)
    return True


def accumulate_track_reid_observation(
    track: TrackState,
    detection: Detection,
    point: dict,
    *,
    observation_id: str,
    frame_index: int,
    time: float,
) -> None:
    """Validate one ReID vector and incorporate independent evidence."""

    if detection.reid_feature is None:
        return
    vector = np.asarray(detection.reid_feature, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or norm <= 1e-8:
        return
    vector = vector / norm
    point["_hasReidEvidence"] = True
    point["_reidEvidenceFingerprint"] = str(
        detection.reid_evidence_fingerprint or "observation:" + observation_id
    )
    independent_evidence = _add_reid_sample(
        track, vector, detection, frame_index, time
    )
    if not independent_evidence:
        return
    if (
        detection.reid_role not in {"player", "goalkeeper", "referee", "other"}
        or detection.reid_role_confidence is None
        or float(detection.reid_role_confidence) < 0.60
    ):
        return
    role = str(detection.reid_role)
    confidence = float(detection.reid_role_confidence)
    point["_reidRole"] = role
    point["_reidRoleConfidence"] = confidence
    track.reid_role_votes[role] = track.reid_role_votes.get(role, 0.0) + confidence
    if not track.manual_kind:
        track.role = max(
            track.reid_role_votes,
            key=lambda value: (track.reid_role_votes[value], value),
        )


__all__ = ["accumulate_track_reid_observation", "rebuild_track_reid_reservoir"]

