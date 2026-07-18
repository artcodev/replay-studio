"""Choose bounded, partition-aware jersey crops from track observations."""

from __future__ import annotations

from math import hypot, isfinite

from .reconstruction_track_state import TrackState
from .reconstruction_identity_persistence import previous_canonical_people
from .reconstruction_identity_semantics import (
    annotation_role,
    annotation_source_identity,
    identity_annotations,
    split_range,
)
from .reconstruction_jersey_policy import (
    JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION,
    JERSEY_OCR_MIN_CROP_GAP_SECONDS,
)
from .bounding_box_geometry import intersection_over_union


def jersey_crop_point_quality(point: dict) -> float:
    bbox = point.get("bbox") or {}
    width = max(0.0, float(bbox.get("width") or 0.0))
    height = max(0.0, float(bbox.get("height") or 0.0))
    confidence = max(0.0, min(1.0, float(point.get("confidence") or 0.0)))
    height_score = min(1.0, height / 96.0)
    width_score = min(1.0, width / 40.0)
    aspect = width / max(1.0, height)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.42) / 0.42)
    return max(
        0.0,
        min(
            1.0,
            0.50 * confidence
            + 0.25 * height_score
            + 0.15 * width_score
            + 0.10 * aspect_score,
        ),
    )


def select_jersey_crop_points(
    track: TrackState,
    available_frame_indices: set[int],
    prospective_split_ranges: tuple[tuple[float, float], ...] = (),
) -> tuple[list[tuple[dict, float]], int, int]:
    """Select temporally diverse crops for every prospective split partition."""

    role = annotation_role(track.manual_kind) or track.role
    if role in {"referee", "other"}:
        return [], 0, 0
    candidates: list[tuple[dict, float]] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox") or {}
        bbox_values = tuple(
            float(bbox.get(key) or 0.0)
            for key in ("x", "y", "width", "height")
        )
        if (
            frame_index is None
            or int(frame_index) not in available_frame_indices
            or not all(isfinite(value) for value in bbox_values)
            or bbox_values[2] <= 0.0
            or bbox_values[3] <= 0.0
        ):
            continue
        candidates.append((point, jersey_crop_point_quality(point)))
    partitions: dict[tuple[int, ...], list[tuple[dict, float]]] = {}
    for candidate in candidates:
        timestamp = float(candidate[0].get("t") or 0.0)
        membership = tuple(
            index
            for index, (start, end) in enumerate(prospective_split_ranges)
            if start <= timestamp < end
        )
        partitions.setdefault(membership, []).append(candidate)

    selected: list[tuple[dict, float]] = []
    for partition in partitions.values():
        partition.sort(
            key=lambda item: (
                -item[1],
                -float(item[0].get("confidence") or 0.0),
                float(item[0].get("t") or 0.0),
                int(item[0].get("frameIndex") or 0),
                str(item[0].get("observationId") or ""),
            )
        )
        partition_selected: list[tuple[dict, float]] = []
        for candidate in partition:
            candidate_time = float(candidate[0].get("t") or 0.0)
            if any(
                abs(candidate_time - float(item[0].get("t") or 0.0))
                < JERSEY_OCR_MIN_CROP_GAP_SECONDS
                for item in partition_selected
            ):
                continue
            partition_selected.append(candidate)
            if len(partition_selected) >= JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION:
                break
        selected.extend(partition_selected)

    return (
        sorted(
            selected,
            key=lambda item: (
                float(item[0].get("t") or 0.0),
                int(item[0].get("frameIndex") or 0),
                str(item[0].get("observationId") or ""),
            ),
        ),
        len(candidates),
        len(partitions),
    )


def prospective_jersey_split_ranges(
    track: TrackState,
    scene: dict | None,
) -> tuple[tuple[float, float], ...]:
    """Return persisted split ranges that can own observations of ``track``."""

    if not scene:
        return ()
    observation_ids = {
        str(point.get("observationId"))
        for point in track.points
        if point.get("observationId")
    }
    source_tracklet_ids = set(track.source_tracklet_ids or {track.local_tracklet_id})
    source_tracklet_ids.add(track.local_tracklet_id)
    previous_by_id = {
        str(person.get("canonicalPersonId") or person.get("id")): person
        for person in previous_canonical_people(scene)
        if person.get("canonicalPersonId") or person.get("id")
    }
    ranges: set[tuple[float, float]] = set()
    for annotation in identity_annotations(scene):
        time_range = split_range(annotation)
        if time_range is None:
            continue
        target_id = str(annotation.get("targetObservationId") or "")
        source_id = annotation_source_identity(annotation)
        relevant = target_id in observation_ids
        snapshot = annotation.get("targetObservation") or {}
        snapshot_bbox = snapshot.get("bbox") or {}
        if (
            not relevant
            and snapshot.get("frameIndex") is not None
            and all(
                snapshot_bbox.get(key) is not None
                for key in ("x", "y", "width", "height")
            )
        ):
            frame_index = int(snapshot["frameIndex"])
            target_box = (
                float(snapshot_bbox["x"]),
                float(snapshot_bbox["y"]),
                float(snapshot_bbox["x"]) + float(snapshot_bbox["width"]),
                float(snapshot_bbox["y"]) + float(snapshot_bbox["height"]),
            )
            for point in track.points:
                bbox = point.get("bbox") or {}
                if int(point.get("frameIndex", -1)) != frame_index or not bbox:
                    continue
                box = (
                    float(bbox["x"]),
                    float(bbox["y"]),
                    float(bbox["x"]) + float(bbox["width"]),
                    float(bbox["y"]) + float(bbox["height"]),
                )
                scale = max(
                    1.0,
                    min(float(snapshot_bbox["height"]), float(bbox["height"])),
                )
                normalized_center = hypot(
                    (box[0] + box[2] - target_box[0] - target_box[2]) / 2.0,
                    (box[1] + box[3] - target_box[1] - target_box[3]) / 2.0,
                ) / scale
                if (
                    intersection_over_union(target_box, box) >= 0.50
                    and normalized_center <= 0.50
                ):
                    relevant = True
                    break
        if source_id and source_id == track.canonical_person_id:
            relevant = True
        previous = previous_by_id.get(str(source_id or ""))
        if previous is not None:
            previous_observation_ids = {
                str(item.get("observationId") or item.get("id"))
                for item in previous.get("observations") or []
                if item.get("observationId") or item.get("id")
            }
            previous_tracklet_ids = {
                str(item)
                for item in (
                    previous.get("sourceTrackletIds")
                    or previous.get("memberTrackletIds")
                    or []
                )
            }
            relevant = relevant or bool(observation_ids & previous_observation_ids)
            relevant = relevant or bool(source_tracklet_ids & previous_tracklet_ids)
        if relevant:
            ranges.add(time_range)
    return tuple(sorted(ranges))
