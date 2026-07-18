from __future__ import annotations

"""Identity-split anchor remapping and partition state reconstruction."""

from math import hypot

import numpy as np

from .reconstruction_errors import IdentityCorrectionError
from .reconstruction_track_state import TrackState
from .track_reid_accumulator import rebuild_track_reid_reservoir
from .reconstruction_identity_contract import CANONICAL_ROSTER_BINDING_CORRECTION
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
    is_identity_unbind_tombstone,
)
from .bounding_box_geometry import intersection_over_union

def resolve_split_target_point(
    tracks: list[TrackState],
    annotation: dict,
    *,
    require_source_identity: bool = False,
) -> tuple[TrackState, dict]:
    """Remap a snapshotted observation without trusting detector ordering."""

    correction_id = str(annotation.get("id") or "split")
    snapshot = annotation.get("targetObservation") or {}
    snapshot_bbox = snapshot.get("bbox")
    if snapshot.get("frameIndex") is None or not snapshot_bbox:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} has no immutable target snapshot",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="missing-target-observation-snapshot",
            source_track_id=annotation_source_identity(annotation),
        )
    target_id = str(annotation.get("targetObservationId") or "")
    expected_source_id = str(annotation_source_identity(annotation) or "")
    lineage_tracks = [
        track
        for track in tracks
        if str(track.canonical_person_id or "") == expected_source_id
    ]
    if require_source_identity and not lineage_tracks:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} cannot find its produced parent identity",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="split-source-lineage-not-found",
            source_track_id=expected_source_id or None,
            target_id=target_id or None,
        )
    candidate_tracks = lineage_tracks or tracks
    frame_index = int(snapshot["frameIndex"])
    target_box = (
        float(snapshot_bbox["x"]),
        float(snapshot_bbox["y"]),
        float(snapshot_bbox["x"]) + float(snapshot_bbox["width"]),
        float(snapshot_bbox["y"]) + float(snapshot_bbox["height"]),
    )
    candidates: list[tuple[int, float, float, TrackState, dict]] = []
    rejected_exact: list[dict] = []
    for track in candidate_tracks:
        for point in track.points:
            if int(point.get("frameIndex", -1)) != frame_index or not point.get("bbox"):
                continue
            bbox = point["bbox"]
            box = (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            )
            overlap = float(intersection_over_union(target_box, box))
            target_center = (
                (target_box[0] + target_box[2]) / 2.0,
                (target_box[1] + target_box[3]) / 2.0,
            )
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            scale = max(1.0, min(float(snapshot_bbox["height"]), float(bbox["height"])))
            normalized_center = hypot(
                center[0] - target_center[0],
                center[1] - target_center[1],
            ) / scale
            exact_id = int(str(point.get("observationId") or "") == target_id)
            diagnostic = {
                "rawTrackId": track.id,
                "observationId": point.get("observationId"),
                "frameIndex": frame_index,
                "bboxIou": round(overlap, 4),
                "normalizedCenterDistance": round(normalized_center, 4),
                "exactObservationId": bool(exact_id),
            }
            if overlap >= 0.50 and normalized_center <= 0.50:
                candidates.append((exact_id, overlap, normalized_center, track, point))
            elif exact_id:
                # Detector-index IDs from an older scene may now name a nearby
                # person. Keep the evidence for diagnostics, but never trust it.
                rejected_exact.append(diagnostic)
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].id))
    candidate_diagnostics = [
        {
            "rawTrackId": track.id,
            "observationId": point.get("observationId"),
            "frameIndex": frame_index,
            "bboxIou": round(overlap, 4),
            "normalizedCenterDistance": round(center_distance, 4),
            "exactObservationId": bool(exact_id),
        }
        for exact_id, overlap, center_distance, track, point in candidates
    ]
    if not candidates:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} could not remap its target observation",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="target-observation-not-found",
            source_track_id=annotation_source_identity(annotation),
            target_id=target_id or None,
            candidates=rejected_exact,
        )
    # Any second geometrically viable row is unsafe. This deliberately prefers
    # a failed rebuild over splitting the person standing next to the target.
    if len(candidates) != 1:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} is ambiguous at the target frame",
            correction_id=correction_id,
            action="split",
            status="ambiguous",
            reason="multiple-target-observation-matches",
            source_track_id=annotation_source_identity(annotation),
            target_id=target_id or None,
            candidates=candidate_diagnostics,
        )
    return candidates[0][3], candidates[0][4]


def split_annotation_partition(
    annotation_id: str,
    annotation: dict | None,
    inside: list[dict],
    outside: list[dict],
    start: float,
    end: float,
) -> str:
    """Locate a semantic correction on the side containing its anchor.

    Point annotation ids are authoritative.  The persisted target observation
    and time are fallbacks for detector reorder/outage rebuilds where the new
    observation id cannot equal the old snapshot id.
    """

    def point_has_annotation(point: dict) -> bool:
        return annotation_id == str(point.get("annotationId") or "") or annotation_id in {
            str(value) for value in point.get("annotationIds") or []
        }

    in_point = any(point_has_annotation(point) for point in inside)
    out_point = any(point_has_annotation(point) for point in outside)
    if in_point != out_point:
        return "range" if in_point else "remaining"
    if in_point and out_point:
        return "ambiguous"
    if annotation is None:
        return "unknown"

    target_observation_id = str(annotation.get("targetObservationId") or "").strip()
    if target_observation_id:
        in_observation = any(
            str(point.get("observationId") or "") == target_observation_id
            for point in inside
        )
        out_observation = any(
            str(point.get("observationId") or "") == target_observation_id
            for point in outside
        )
        if in_observation != out_observation:
            return "range" if in_observation else "remaining"
        if in_observation and out_observation:
            return "ambiguous"

    snapshot = annotation.get("targetObservation")
    anchor_time = snapshot.get("sceneTime") if isinstance(snapshot, dict) else None
    if anchor_time is None:
        anchor_time = annotation.get("sceneTime")
    try:
        anchor_time = float(anchor_time)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(anchor_time):
        return "unknown"
    return "range" if start <= anchor_time < end else "remaining"


def partition_external_player_ids(
    source: TrackState,
    source_annotation_ids: set[str],
    split_annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
    *,
    correction_id: str,
    source_identity_id: str,
    split_identity_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Assign confirmed roster semantics only to their anchored partition."""

    dedicated_decisions: dict[str, tuple[str, str | None]] = {}
    for annotation_id in source.annotation_ids:
        annotation = annotations_by_id.get(annotation_id)
        if annotation is None:
            continue
        if (
            annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("rosterBindingState") in {"bound", "unbound"}
        ):
            dedicated_decisions[annotation_id] = (
                str(annotation["rosterBindingState"]),
                str(annotation.get("externalPlayerId") or "").strip() or None,
            )
            continue
    if not dedicated_decisions:
        return None, None, None, None

    def value_for(
        annotation_ids: set[str], canonical_person_id: str
    ) -> tuple[str | None, str | None]:
        dedicated = {
            decision
            for annotation_id, decision in dedicated_decisions.items()
            if annotation_id in annotation_ids
        }
        if len(dedicated) > 1:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} assigned conflicting roster edits to one partition",
                correction_id=correction_id,
                action="split",
                status="conflict",
                reason="conflicting-dedicated-roster-decisions",
                source_track_id=source_identity_id,
                target_id=canonical_person_id,
                candidates=[
                    {"rosterBindingState": state, "externalPlayerId": external_id}
                    for state, external_id in sorted(
                        dedicated,
                        key=lambda item: (item[0], item[1] or ""),
                    )
                ],
            )
        if dedicated:
            state, external_id = next(iter(dedicated))
            return (external_id if state == "bound" else None), state
        return None, None

    source_external_id, source_roster_state = value_for(
        source_annotation_ids, source_identity_id
    )
    split_external_id, split_roster_state = value_for(
        split_annotation_ids, split_identity_id
    )
    return (
        source_external_id,
        split_external_id,
        source_roster_state,
        split_roster_state,
    )


def partition_manual_semantics(
    annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
) -> tuple[str | None, str | None, bool]:
    """Rebuild manual role/label only from positive local corrections."""

    rows = [
        annotations_by_id[annotation_id]
        for annotation_id in annotation_ids
        if annotation_id in annotations_by_id
        and annotation_action(annotations_by_id[annotation_id])
        in {"confirm", "merge", "split"}
        and not is_identity_unbind_tombstone(annotations_by_id[annotation_id])
        and annotations_by_id[annotation_id].get("kind") != "ignore"
    ]
    rows.sort(
        key=lambda item: (
            1 if annotation_action(item) == "split" else 0,
            str(item.get("updatedAt") or ""),
            float(item.get("sceneTime") or 0.0),
            int(item.get("frameIndex") or 0),
            str(item.get("id") or ""),
        )
    )
    if not rows:
        return None, None, False
    kind = next(
        (str(item["kind"]) for item in reversed(rows) if item.get("kind")),
        None,
    )
    label = next(
        (
            str(item["label"]).strip()
            for item in reversed(rows)
            if str(item.get("label") or "").strip()
        ),
        None,
    )
    return kind, label, True


def refresh_split_track_state(track: TrackState) -> None:
    track.points.sort(key=lambda point: (float(point["t"]), int(point.get("frameIndex") or 0)))
    if not track.points:
        return
    track.last_frame = max(int(point.get("frameIndex") or 0) for point in track.points)
    track.last_height = float((track.points[-1].get("bbox") or {}).get("height") or track.last_height)
    appearance_features = [
        np.asarray(point["_appearanceFeature"], dtype=np.float32)
        for point in track.points
        if point.get("_appearanceFeature") is not None
    ]
    if appearance_features:
        track.feature_sum = np.sum(np.stack(appearance_features), axis=0)
        track.feature_count = len(appearance_features)
    else:
        # A split partition must never retain aggregate evidence that none of
        # its own observations can substantiate.
        track.feature_sum = None
        track.feature_count = 0
    role_votes: dict[str, float] = {}
    for point in track.points:
        role = str(point.get("_reidRole") or "")
        if role not in {"player", "goalkeeper", "referee", "other"}:
            continue
        try:
            confidence = float(point.get("_reidRoleConfidence"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(confidence) and confidence >= 0.60:
            role_votes[role] = role_votes.get(role, 0.0) + confidence
    track.reid_role_votes = role_votes
    if not track.manual_kind:
        track.role = (
            max(role_votes, key=lambda value: (role_votes[value], value))
            if role_votes
            else None
        )
    retained_observation_ids = {
        str(point.get("observationId"))
        for point in track.points
        if point.get("observationId")
    }
    track.reid_observation_ids.intersection_update(retained_observation_ids)
    if track.reid_sample_candidates:
        track.reid_sample_candidates = [
            item
            for item in track.reid_sample_candidates
            if str(item.get("observationId") or "") in retained_observation_ids
        ]
        rebuild_track_reid_reservoir(track)
    evidence_rows = [
        point
        for point in track.points
        if point.get("_hasReidEvidence") and point.get("observationId")
    ]
    if evidence_rows:
        fingerprints = [
            str(
                point.get("_reidEvidenceFingerprint")
                or "observation:" + str(point["observationId"])
            )
            for point in evidence_rows
        ]
        track.reid_observation_ids = {
            str(point["observationId"]) for point in evidence_rows
        }
        track.reid_evidence_fingerprints = set(fingerprints)
        track.reid_observation_count = len(track.reid_evidence_fingerprints)
        track.reid_duplicate_evidence_count = len(fingerprints) - len(
            track.reid_evidence_fingerprints
        )
    elif track.reid_sample_candidates:
        track.reid_evidence_fingerprints = {
            str(
                item.get("evidenceFingerprint")
                or "observation:" + str(item.get("observationId") or "")
            )
            for item in track.reid_sample_candidates
        }
        track.reid_observation_count = len(track.reid_evidence_fingerprints)
        track.reid_duplicate_evidence_count = 0
    else:
        track.reid_evidence_fingerprints.clear()
        track.reid_observation_count = 0
        track.reid_duplicate_evidence_count = 0
    track.identity_tombstone_ids.intersection_update(track.annotation_ids)
