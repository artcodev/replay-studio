from __future__ import annotations

"""Persistent canonical identity assignment across reconstruction rebuilds."""

from copy import deepcopy
from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bounding_box_geometry import intersection_over_union
from .reconstruction_canonical_person_id import derive_canonical_person_id
from .reconstruction_errors import ReconstructionError
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_role, annotation_team
from .reconstruction_track_observations import merge_track_observations

def track_state_observations(
    track: TrackState,
    *,
    canonical_person_id: str | None = None,
    source_start: float = 0.0,
) -> list[dict]:
    """Publish image evidence independently from 3D trajectory acceptance."""

    rows: list[dict] = []
    for point in track.points:
        if point.get("frameIndex") is None or not point.get("bbox"):
            continue
        frame_index = int(point["frameIndex"])
        source_tracklet_id = str(
            point.get("sourceTrackletId") or track.local_tracklet_id
        )
        observation_id = str(
            point.get("observationId") or f"{source_tracklet_id}:{frame_index}"
        )
        row = {
            "id": observation_id,
            "observationId": observation_id,
            "frameIndex": frame_index,
            "sourceFrameIndex": frame_index,
            "sceneTime": round(float(point["t"]), 3),
            "sourceTime": round(source_start + float(point["t"]), 3),
            "bbox": {
                "x": round(float(point["bbox"]["x"]), 2),
                "y": round(float(point["bbox"]["y"]), 2),
                "width": round(float(point["bbox"]["width"]), 2),
                "height": round(float(point["bbox"]["height"]), 2),
            },
            "confidence": round(float(point.get("confidence") or 0.0), 3),
            "annotationId": point.get("annotationId"),
            "sourceTrackletId": source_tracklet_id,
            "canonicalPersonId": canonical_person_id,
        }
        if point.get("pitchX") is not None and point.get("pitchZ") is not None:
            row.update(
                {
                    "metricStatus": "accepted",
                    "metricReason": None,
                    "pitch": {
                        "x": round(float(point["pitchX"]), 2),
                        "z": round(float(point["pitchZ"]), 2),
                    },
                    "positionSource": "observation",
                }
            )
        else:
            row.update(
                {
                    "metricStatus": "unprojected",
                    "metricReason": "metric-projection-unavailable",
                    "positionSource": "track-inferred",
                }
            )
        if point.get("projectionSource"):
            row["projectionSource"] = str(point["projectionSource"])
        if point.get("calibrationFrameIndex") is not None:
            row["calibrationFrameIndex"] = int(point["calibrationFrameIndex"])
        if point.get("positionUncertaintyMetres") is not None:
            row["positionUncertaintyMetres"] = round(
                float(point["positionUncertaintyMetres"]), 3
            )
        rows.append(row)
    return merge_track_observations(rows)


def previous_canonical_people(scene: dict) -> list[dict]:
    payload = scene.get("payload", {})
    return [deepcopy(item) for item in payload.get("canonicalPeople") or []]


def _canonical_match_score(
    track: TrackState,
    previous: dict,
    team_id: str | None = None,
) -> float:
    """Score only evidence strong enough to preserve a canonical ID.

    Exact manual/roster evidence is authoritative. Automatic image remapping
    requires several shared observations over time; one crowded-frame IoU is
    deliberately worth zero.
    """

    previous_annotations = set(previous.get("annotationIds") or [])
    annotation_overlap = len(track.annotation_ids & previous_annotations)
    previous_external_id = previous.get("externalPlayerId")
    if (
        track.manual_external_player_id
        and previous_external_id
        and track.manual_external_player_id != previous_external_id
    ):
        return 0.0
    resolved_team = team_id or annotation_team(track.manual_kind)
    previous_team = previous.get("teamId")
    if resolved_team and previous_team and resolved_team != previous_team:
        return 0.0
    resolved_role = annotation_role(track.manual_kind) or track.role
    previous_role = previous.get("role")
    if resolved_role and previous_role and resolved_role != previous_role:
        return 0.0
    score = annotation_overlap * 100.0
    if track.manual_external_player_id and track.manual_external_player_id == previous_external_id:
        score += 80.0
    previous_by_frame = {
        int(item["frameIndex"]): item
        for item in previous.get("observations") or []
        if item.get("frameIndex") is not None and item.get("bbox")
    }
    overlaps: list[float] = []
    normalized_center_residuals: list[float] = []
    matched_times: list[float] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox")
        if frame_index is None or not bbox or int(frame_index) not in previous_by_frame:
            continue
        old_bbox = previous_by_frame[int(frame_index)]["bbox"]
        overlap = intersection_over_union(
            (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            ),
            (
                float(old_bbox["x"]),
                float(old_bbox["y"]),
                float(old_bbox["x"]) + float(old_bbox["width"]),
                float(old_bbox["y"]) + float(old_bbox["height"]),
            ),
        )
        new_center_x = float(bbox["x"]) + float(bbox["width"]) / 2.0
        new_center_y = float(bbox["y"]) + float(bbox["height"]) / 2.0
        old_center_x = float(old_bbox["x"]) + float(old_bbox["width"]) / 2.0
        old_center_y = float(old_bbox["y"]) + float(old_bbox["height"]) / 2.0
        scale = max(
            1.0,
            min(float(bbox["height"]), float(old_bbox["height"])),
        )
        overlaps.append(float(overlap))
        normalized_center_residuals.append(
            hypot(new_center_x - old_center_x, new_center_y - old_center_y) / scale
        )
        matched_times.append(float(point.get("t") or 0.0))

    if score >= 80.0:
        return score + sum(overlap >= 0.25 for overlap in overlaps)
    if len(overlaps) < 3:
        return 0.0
    time_span = max(matched_times) - min(matched_times)
    median_iou = float(np.median(overlaps))
    residual_p90 = float(np.percentile(normalized_center_residuals, 90))
    if time_span < 0.4 or median_iou < 0.25 or residual_p90 > 1.5:
        return 0.0
    observation_denominator = max(
        1,
        min(len(track.points), len(previous_by_frame)),
    )
    coverage = len(overlaps) / observation_denominator
    return round(
        10.0
        + len(overlaps)
        + median_iou * 2.0
        + min(1.0, coverage)
        + min(2.0, time_span),
        6,
    )


def assign_persistent_canonical_person_ids(
    tracks: list[TrackState],
    scene: dict,
    mapping: dict[int, str] | None = None,
) -> None:
    """Keep identity IDs stable across rebuilds when image evidence overlaps."""

    previous = previous_canonical_people(scene)
    previous_by_identifier: dict[str, list[str]] = {}
    for item in previous:
        canonical_id = str(
            item.get("canonicalPersonId") or item.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        for identifier in {canonical_id, str(item.get("id") or "").strip()}:
            if identifier:
                previous_by_identifier.setdefault(identifier, []).append(canonical_id)

    manual_claims: dict[str, list[TrackState]] = {}
    for track in tracks:
        if len(track.manual_identity_owner_ids) != 1:
            continue
        owner_id = next(iter(track.manual_identity_owner_ids))
        matches = sorted(set(previous_by_identifier.get(owner_id, [])))
        if len(matches) > 1:
            raise ReconstructionError(
                f"Explicit canonical owner {owner_id} resolves to multiple saved identities"
            )
        if not matches:
            continue
        canonical_id = matches[0]
        manual_claims.setdefault(canonical_id, []).append(track)
    duplicate_manual_claims = {
        canonical_id: claimants
        for canonical_id, claimants in manual_claims.items()
        if len(claimants) > 1
    }
    if duplicate_manual_claims:
        canonical_id = sorted(duplicate_manual_claims)[0]
        raise ReconstructionError(
            f"Explicit canonical owner {canonical_id} reached multiple unresolved tracks"
        )
    for canonical_id, claimants in manual_claims.items():
        claimants[0].canonical_person_id = canonical_id

    preassigned: dict[str, list[TrackState]] = {}
    for track in tracks:
        if track.canonical_person_id:
            preassigned.setdefault(str(track.canonical_person_id), []).append(track)
    duplicate_preassigned = {
        canonical_id: claimants
        for canonical_id, claimants in preassigned.items()
        if len(claimants) > 1
    }
    if duplicate_preassigned:
        canonical_id = sorted(duplicate_preassigned)[0]
        raise ReconstructionError(
            f"Canonical identity {canonical_id} is claimed by multiple resolved tracks"
        )

    claimed_previous_ids = set(manual_claims)
    claimed_previous_ids.update(
        str(track.canonical_person_id)
        for track in tracks
        if track.canonical_person_id
    )
    if tracks and previous:
        scores = np.zeros((len(tracks), len(previous)), dtype=np.float64)
        for track_index, track in enumerate(tracks):
            if track.canonical_person_id:
                continue
            for previous_index, item in enumerate(previous):
                previous_id = str(
                    item.get("canonicalPersonId") or item.get("id") or ""
                )
                if previous_id in claimed_previous_ids:
                    continue
                scores[track_index, previous_index] = _canonical_match_score(
                    track,
                    item,
                    (mapping or {}).get(track.id),
                )
        rows, columns = linear_sum_assignment(-scores)
        for track_index, previous_index in zip(rows.tolist(), columns.tolist()):
            if tracks[track_index].canonical_person_id:
                continue
            score = float(scores[track_index, previous_index])
            row_alternatives = sorted(
                float(value)
                for index, value in enumerate(scores[track_index])
                if index != previous_index
            )
            column_alternatives = sorted(
                float(value)
                for index, value in enumerate(scores[:, previous_index])
                if index != track_index
            )
            ambiguous = (
                (row_alternatives and score - row_alternatives[-1] < 0.35)
                or (column_alternatives and score - column_alternatives[-1] < 0.35)
            )
            if score <= 0.0 or ambiguous:
                if score > 0.0 and ambiguous:
                    tracks[track_index].identity_conflicts.append(
                        {
                            "id": f"canonical-remap:{tracks[track_index].local_tracklet_id}",
                            "code": "canonical-id-remap-ambiguous",
                            "message": "Previous canonical identity was not reused because another candidate had similar image evidence.",
                            "severity": "review",
                            "relatedTrackletIds": [tracks[track_index].local_tracklet_id],
                        }
                    )
                continue
            previous_id = previous[previous_index].get("canonicalPersonId") or previous[
                previous_index
            ].get("id")
            if previous_id:
                tracks[track_index].canonical_person_id = str(previous_id)

    # Previous IDs that failed or were ambiguous remain reserved. Otherwise a
    # deterministic bbox-derived ID could silently recreate the very mapping
    # that the evidence gate rejected.
    used = {
        str(track.canonical_person_id)
        for track in tracks
        if track.canonical_person_id
    }
    used.update(
        str(item.get("canonicalPersonId") or item.get("id"))
        for item in previous
        if item.get("canonicalPersonId") or item.get("id")
    )
    for track in sorted(tracks, key=lambda item: (float(item.points[0]["t"]), item.id)):
        if track.canonical_person_id:
            continue
        base = derive_canonical_person_id(track)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        track.canonical_person_id = candidate
        used.add(candidate)
