from __future__ import annotations

"""Identity review read-model helpers for exact-frame linking."""

from .bounding_box_geometry import intersection_over_union
from .reconstruction_identity_read_model import (
    bbox_payload_box,
    canonical_analysis_subjects,
)

def raw_person_bbox(raw: dict) -> dict:
    return {
        "x": round(float(raw["x"]) - float(raw["width"]) / 2, 2),
        "y": round(float(raw["y"]) - float(raw["height"]), 2),
        "width": round(float(raw["width"]), 2),
        "height": round(float(raw["height"]), 2),
    }


def track_observation_schema_version(scene: dict) -> int | None:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    try:
        version = int(reconstruction.get("trackObservationSchemaVersion") or 0)
        if version >= 1:
            return version
    except (TypeError, ValueError):
        pass
    return 1 if any(
        "observations" in track
        for track in [
            *(scene.get("payload", {}).get("tracks") or []),
            *(scene.get("payload", {}).get("canonicalPeople") or []),
        ]
    ) else None


def has_track_observation_schema(scene: dict) -> bool:
    return track_observation_schema_version(scene) is not None


def frame_track_observations(scene: dict, frame_index: int) -> list[tuple[dict, dict]]:
    return [
        (subject, observation)
        for subject in canonical_analysis_subjects(scene)
        for observation in subject.get("observations") or []
        if observation.get("frameIndex") is not None
        and int(observation["frameIndex"]) == frame_index
        and observation.get("bbox")
    ]


def pair_detections_to_stored_observations(
    detection_boxes: list[dict],
    observations: list[tuple[dict, dict]],
) -> tuple[dict[int, int], set[int]]:
    """Pair fresh detector rows only to recover metadata, never identity."""

    pairs = sorted(
        (
            (
                intersection_over_union(
                    bbox_payload_box(box),
                    bbox_payload_box(observation["bbox"]),
                ),
                observation_index,
                detection_index,
            )
            for observation_index, (_, observation) in enumerate(observations)
            for detection_index, box in enumerate(detection_boxes)
        ),
        reverse=True,
    )
    by_observation: dict[int, int] = {}
    used_detections: set[int] = set()
    for overlap, observation_index, detection_index in pairs:
        if overlap < 0.20:
            break
        if observation_index in by_observation or detection_index in used_detections:
            continue
        by_observation[observation_index] = detection_index
        used_detections.add(detection_index)
    return by_observation, used_detections
