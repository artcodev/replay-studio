from __future__ import annotations

"""Saved detector-observation ownership for roster corrections."""

from copy import deepcopy
from math import isfinite

from .reconstruction_errors import ReconstructionError


def replace_roster_annotation_references(
    scene: dict,
    canonical_person_id: str,
    old_annotation_id: str,
    new_annotation_id: str,
) -> None:
    """Move published references when a split correction changes its owner id."""

    payload = scene.get("payload", {})
    for person in payload.get("canonicalPeople") or []:
        person_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        annotation_ids = {
            str(value)
            for value in person.get("annotationIds") or []
            if str(value) != old_annotation_id
        }
        if person_id == canonical_person_id:
            annotation_ids.add(new_annotation_id)
            _replace_observation_references(
                person.get("observations") or [],
                old_annotation_id,
                new_annotation_id,
            )
        person["annotationIds"] = sorted(annotation_ids)

    for track in payload.get("tracks") or []:
        is_owner = str(track.get("canonicalPersonId") or "") == canonical_person_id
        annotation_ids = {
            str(value)
            for value in track.get("annotationIds") or []
            if str(value) != old_annotation_id
        }
        if is_owner:
            annotation_ids.add(new_annotation_id)
            _replace_observation_references(
                track.get("observations") or [],
                old_annotation_id,
                new_annotation_id,
            )
        if annotation_ids:
            track["annotationIds"] = sorted(annotation_ids)
        else:
            track.pop("annotationIds", None)


def _replace_observation_references(
    observations: list[dict], old_annotation_id: str, new_annotation_id: str
) -> None:
    for observation in observations:
        if str(observation.get("annotationId") or "") == old_annotation_id:
            observation["annotationId"] = new_annotation_id
        old_ids = {
            str(value) for value in observation.get("annotationIds") or []
        }
        annotation_ids = {
            value for value in old_ids if value != old_annotation_id
        }
        if old_annotation_id in old_ids:
            annotation_ids.add(new_annotation_id)
        if annotation_ids:
            observation["annotationIds"] = sorted(annotation_ids)
        else:
            observation.pop("annotationIds", None)


def saved_detector_observation_for_binding(
    person: dict,
    existing_annotation: dict | None,
    scene_duration: float,
    *,
    preserve_existing: bool = False,
) -> dict:
    """Choose a durable image-space anchor without consulting a live frame."""

    raw_candidates = (
        [] if preserve_existing else list(person.get("observations") or [])
    )
    if isinstance((existing_annotation or {}).get("targetObservation"), dict):
        raw_candidates.append(existing_annotation["targetObservation"])

    candidates = [
        normalized
        for observation in raw_candidates
        if isinstance(observation, dict)
        and (
            normalized := _normalized_detector_observation(
                observation, scene_duration
            )
        )
        is not None
    ]
    if not candidates:
        raise ReconstructionError(
            "This canonical person has no saved detector observation to anchor the roster binding"
        )

    return max(
        candidates,
        key=lambda item: (
            1 if item.get("sourceTrackletId") else 0,
            1 if not item.get("annotationId") else 0,
            1 if item.get("metricStatus") == "accepted" else 0,
            float(item.get("confidence") or 0.0),
            float(item["bbox"]["width"]) * float(item["bbox"]["height"]),
            -int(item["frameIndex"]),
            str(item["observationId"]),
        ),
    )


def _normalized_detector_observation(
    observation: dict, scene_duration: float
) -> dict | None:
    observation_id = str(
        observation.get("observationId") or observation.get("id") or ""
    ).strip()
    bbox = observation.get("bbox")
    try:
        frame_index = int(observation["frameIndex"])
        scene_time = float(observation["sceneTime"])
        values = (
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["width"]),
            float(bbox["height"]),
            scene_time,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not observation_id
        or frame_index < 0
        or not all(isfinite(value) for value in values)
        or scene_time < 0.0
        or scene_time > scene_duration + 1e-6
        or values[0] < 0.0
        or values[1] < 0.0
        or values[2] < 4.0
        or values[3] < 4.0
    ):
        return None
    return {
        **deepcopy(observation),
        "id": observation_id,
        "observationId": observation_id,
        "frameIndex": frame_index,
        "sceneTime": scene_time,
        "bbox": {
            "x": values[0],
            "y": values[1],
            "width": values[2],
            "height": values[3],
        },
    }
