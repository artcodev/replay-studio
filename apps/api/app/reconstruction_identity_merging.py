from __future__ import annotations

"""Raw identity-track merge invariants and state consolidation."""

from copy import deepcopy

from .reconstruction_errors import IdentityCorrectionError, ReconstructionError
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_source_identity
from .track_reid_accumulator import rebuild_track_reid_reservoir

def confirmed_external_player_conflict(
    target_external_player_id: object,
    source_external_player_id: object,
) -> tuple[str, str] | None:
    target_id = str(target_external_player_id or "").strip()
    source_id = str(source_external_player_id or "").strip()
    if target_id and source_id and target_id != source_id:
        return target_id, source_id
    return None


def dedicated_roster_binding_conflict(
    target: TrackState,
    source: TrackState,
) -> tuple[str, str] | None:
    for track in (target, source):
        if track.manual_external_player_id and track.roster_binding_state != "bound":
            raise ReconstructionError(
                "Confirmed roster identity is missing its dedicated binding state"
            )
    if target.roster_binding_state is None or source.roster_binding_state is None:
        return None
    target_decision = (
        str(target.manual_external_player_id)
        if target.roster_binding_state == "bound"
        else "<unbound>"
    )
    source_decision = (
        str(source.manual_external_player_id)
        if source.roster_binding_state == "bound"
        else "<unbound>"
    )
    if target_decision != source_decision:
        return target_decision, source_decision
    return None


def raise_manual_merge_external_player_conflict(
    target: TrackState,
    source: TrackState,
    annotation: dict,
) -> None:
    dedicated_conflict = dedicated_roster_binding_conflict(target, source)
    conflict = dedicated_conflict
    if conflict is None:
        return
    target_external_id, source_external_id = conflict
    correction_id = str(annotation.get("id") or "identity-merge")
    raise IdentityCorrectionError(
        (
            f"Identity correction {correction_id} cannot merge confirmed roster "
            f"players {source_external_id} and {target_external_id}"
        ),
        correction_id=correction_id,
        action="merge",
        status="conflict",
        reason="conflicting-confirmed-external-player-ids",
        source_track_id=annotation_source_identity(annotation),
        target_id=str(annotation.get("mergeTargetId") or "") or None,
        candidates=[
            {"rawTrackId": source.id, "externalPlayerId": source_external_id},
            {"rawTrackId": target.id, "externalPlayerId": target_external_id},
        ],
    )


def merge_raw_track_states(
    target: TrackState,
    source: TrackState,
    *,
    allow_manual_owner_merge: bool = False,
    manual_target_owner_id: str | None = None,
) -> None:
    owner_conflict = bool(
        target.manual_identity_owner_ids
        and source.manual_identity_owner_ids
        and target.manual_identity_owner_ids.isdisjoint(
            source.manual_identity_owner_ids
        )
    )
    if owner_conflict and not allow_manual_owner_merge:
        raise ReconstructionError(
            "Cannot automatically merge different explicitly confirmed canonical identities"
        )
    dedicated_conflict = dedicated_roster_binding_conflict(target, source)
    conflict = dedicated_conflict
    if conflict is not None:
        target_external_id, source_external_id = conflict
        raise ReconstructionError(
            "Cannot merge identities with different confirmed roster players: "
            f"{source_external_id} and {target_external_id}"
        )
    points_by_time: dict[float, dict] = {}
    for point in [*target.points, *source.points]:
        key = round(float(point["t"]), 4)
        previous = points_by_time.get(key)
        point_priority = (
            1 if point.get("annotationId") else 0,
            float(point.get("confidence") or 0.0),
        )
        previous_priority = (
            1 if previous and previous.get("annotationId") else 0,
            float(previous.get("confidence") or 0.0) if previous else 0.0,
        )
        if previous is None or point_priority >= previous_priority:
            points_by_time[key] = point
    target.points = [points_by_time[key] for key in sorted(points_by_time)]
    if source.feature_sum is not None:
        if target.feature_sum is None:
            target.feature_sum = source.feature_sum.copy()
        else:
            target.feature_sum += source.feature_sum
    target.feature_count += source.feature_count
    target.last_frame = max(target.last_frame, source.last_frame)
    target.last_height = max(target.last_height, source.last_height)
    target.role = target.role or source.role
    target.annotation_ids.update(source.annotation_ids)
    target.identity_tombstone_ids.update(source.identity_tombstone_ids)
    target.identity_tombstone_ids.intersection_update(target.annotation_ids)
    if (
        source.manual_semantic_key is not None
        and (
            target.manual_semantic_key is None
            or source.manual_semantic_key >= target.manual_semantic_key
        )
    ):
        target.manual_kind = source.manual_kind
        target.manual_label = source.manual_label
        target.manual_semantic_key = source.manual_semantic_key
    if source.roster_binding_state is not None:
        target.roster_binding_state = source.roster_binding_state
        target.roster_binding_annotation_ids.update(
            source.roster_binding_annotation_ids
        )
        target.manual_external_player_id = source.manual_external_player_id
    if allow_manual_owner_merge and manual_target_owner_id:
        # The user selected this canonical target as the survivor.  The source
        # owner becomes an alias authorized by the merge; it must not replace
        # the target merely because the target raw fragment had no frame-level
        # confirmation of its own.
        target.manual_identity_owner_ids = {manual_target_owner_id}
    elif owner_conflict:
        # An explicit Merge correction chooses the target identity as the
        # survivor. Do not keep both cannot-link owner labels on one raw track.
        target.manual_identity_owner_ids = set(
            target.manual_identity_owner_ids
            or source.manual_identity_owner_ids
        )
    else:
        target.manual_identity_owner_ids.update(source.manual_identity_owner_ids)
    target.source_tracklet_ids.update(source.source_tracklet_ids or {source.local_tracklet_id})
    if target.reid_sample_candidates or source.reid_sample_candidates:
        target.reid_observation_ids.update(source.reid_observation_ids)
        overlapping_fingerprints = target.reid_evidence_fingerprints.intersection(
            source.reid_evidence_fingerprints
        )
        target.reid_duplicate_evidence_count += (
            source.reid_duplicate_evidence_count + len(overlapping_fingerprints)
        )
        target.reid_evidence_fingerprints.update(source.reid_evidence_fingerprints)
        target.reid_observation_count = len(target.reid_evidence_fingerprints)
        target.reid_sample_candidates.extend(
            {
                **item,
                "vector": item["vector"].copy(),
            }
            for item in source.reid_sample_candidates
        )
        # IDs and timestamps are metadata, not independent evidence. Collapse
        # identical decoded crops before the temporal reservoir is rebuilt.
        best_by_fingerprint: dict[str, dict] = {}
        for item in target.reid_sample_candidates:
            fingerprint = str(
                item.get("evidenceFingerprint")
                or "observation:" + str(item.get("observationId") or "")
            )
            previous = best_by_fingerprint.get(fingerprint)
            if previous is None or (
                float(item["quality"]),
                -int(item["frameIndex"]),
            ) > (
                float(previous["quality"]),
                -int(previous["frameIndex"]),
            ):
                best_by_fingerprint[fingerprint] = item
        target.reid_sample_candidates = list(best_by_fingerprint.values())
        # Deduplicate temporal bins after a manual merge/split and recompute
        # the representative mean from independent quality-ranked views.
        best_by_bin: dict[int, dict] = {}
        for item in target.reid_sample_candidates:
            temporal_bin = int(item["temporalBin"])
            previous = best_by_bin.get(temporal_bin)
            if previous is None or (
                float(item["quality"]),
                -int(item["frameIndex"]),
            ) > (
                float(previous["quality"]),
                -int(previous["frameIndex"]),
            ):
                best_by_bin[temporal_bin] = item
        target.reid_sample_candidates = sorted(
            best_by_bin.values(),
            key=lambda item: (-float(item["quality"]), int(item["frameIndex"])),
        )[:64]
        rebuild_track_reid_reservoir(target)
    role_rows = [
        point
        for point in target.points
        if point.get("_reidRole") in {"player", "goalkeeper", "referee", "other"}
        and point.get("_reidRoleConfidence") is not None
    ]
    if role_rows:
        target.reid_role_votes = {}
        seen_role_fingerprints: set[str] = set()
        for point in role_rows:
            fingerprint = str(
                point.get("_reidEvidenceFingerprint")
                or "observation:" + str(point.get("observationId") or "")
            )
            if fingerprint in seen_role_fingerprints:
                continue
            seen_role_fingerprints.add(fingerprint)
            role = str(point["_reidRole"])
            confidence = float(point["_reidRoleConfidence"])
            target.reid_role_votes[role] = (
                target.reid_role_votes.get(role, 0.0) + confidence
            )
    else:
        for role, weight in source.reid_role_votes.items():
            target.reid_role_votes[role] = (
                target.reid_role_votes.get(role, 0.0) + weight
            )
    if target.reid_role_votes and not target.manual_kind:
        target.role = max(
            target.reid_role_votes,
            key=lambda value: (target.reid_role_votes[value], value),
        )
    target.identity_evidence.extend(deepcopy(source.identity_evidence))
    target.identity_conflicts.extend(deepcopy(source.identity_conflicts))
