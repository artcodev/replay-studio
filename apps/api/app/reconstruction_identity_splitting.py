from __future__ import annotations

"""Manual identity split remapping, partitioning, and state rebuild."""

from copy import deepcopy

from .reconstruction_errors import IdentityCorrectionError
from .reconstruction_track_state import TrackState
from .reconstruction_identity_correction_graph import ordered_split_corrections
from .reconstruction_identity_split_partition import (
    partition_external_player_ids,
    partition_manual_semantics,
    refresh_split_track_state,
    resolve_split_target_point,
    split_annotation_partition,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
    bound_roster_semantics_compatible,
    identity_annotations,
    is_identity_unbind_tombstone,
    split_range,
)
from .reconstruction_canonical_person_id import derive_canonical_person_id











def apply_canonical_split_corrections(
    tracks: list[TrackState],
    scene: dict,
) -> tuple[list[TrackState], dict]:
    """Partition resolved identities by persisted [start, end) manual ranges.

    Splits run after automatic identity resolution, which makes each persisted
    range a real cannot-link barrier rather than another hint the resolver may
    immediately undo.
    """

    result = list(tracks)
    annotations_by_id = {
        str(item.get("id")): item
        for item in identity_annotations(scene)
        if item.get("id")
    }
    splits = ordered_split_corrections(identity_annotations(scene))
    produced_split_ids = {
        str(annotation.get("splitCanonicalPersonId") or "").strip()
        for annotation in splits
        if str(annotation.get("splitCanonicalPersonId") or "").strip()
    }
    next_track_id = max((track.id for track in result), default=0) + 1
    applied: list[dict] = []
    for annotation in splits:
        correction_id = str(annotation["id"])
        time_range = split_range(annotation)
        if time_range is None:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} has an invalid range",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="invalid-split-range",
                source_track_id=annotation_source_identity(annotation),
            )
        start, end = time_range
        source_identity_id_hint = str(annotation_source_identity(annotation) or "")
        source, target_point = resolve_split_target_point(
            result,
            annotation,
            require_source_identity=source_identity_id_hint in produced_split_ids,
        )
        inside = [point for point in source.points if start <= float(point["t"]) < end]
        outside = [point for point in source.points if not start <= float(point["t"]) < end]
        if target_point not in inside:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} target is outside its range",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="target-outside-split-range",
                source_track_id=annotation_source_identity(annotation),
                target_id=str(annotation.get("targetObservationId") or "") or None,
            )
        if not inside or not outside:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} would consume the complete identity",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="empty-split-partition",
                source_track_id=annotation_source_identity(annotation),
                target_id=str(annotation.get("targetObservationId") or "") or None,
            )

        original_annotation_ids = set(source.annotation_ids)
        source_annotation_ids: set[str] = set()
        split_annotation_ids: set[str] = {correction_id}
        for annotation_id in sorted(original_annotation_ids):
            semantic_annotation = annotations_by_id.get(annotation_id)
            partition = split_annotation_partition(
                annotation_id,
                semantic_annotation,
                inside,
                outside,
                start,
                end,
            )
            if partition == "range":
                split_annotation_ids.add(annotation_id)
                continue
            if partition == "ambiguous" and (
                is_identity_unbind_tombstone(semantic_annotation)
                or str((semantic_annotation or {}).get("externalPlayerId") or "").strip()
            ):
                raise IdentityCorrectionError(
                    f"Split correction {correction_id} cannot localize a roster correction",
                    correction_id=correction_id,
                    action="split",
                    status="ambiguous",
                    reason="ambiguous-roster-correction-partition",
                    source_track_id=annotation_source_identity(annotation),
                    target_id=annotation_id,
                )
            # Fail closed: an unlocalized semantic correction stays with the
            # original identity instead of being guessed onto the new range.
            source_annotation_ids.add(annotation_id)

        split_track = deepcopy(source)
        split_track.id = next_track_id
        next_track_id += 1
        split_track.points = [deepcopy(point) for point in inside]
        source.points = [deepcopy(point) for point in outside]

        source_identity_id = str(
            annotation.get("canonicalPersonId")
            or source.canonical_person_id
            or derive_canonical_person_id(source)
        )
        split_identity_id = str(annotation.get("splitCanonicalPersonId") or "")
        if not split_identity_id or split_identity_id == source_identity_id:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} has no distinct identity key",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="invalid-split-identity",
                source_track_id=source_identity_id,
            )
        (
            source_external_player_id,
            split_external_player_id,
            source_roster_binding_state,
            split_roster_binding_state,
        ) = (
            partition_external_player_ids(
                source,
                source_annotation_ids,
                split_annotation_ids,
                annotations_by_id,
                correction_id=correction_id,
                source_identity_id=source_identity_id,
                split_identity_id=split_identity_id,
            )
        )
        source.annotation_ids = source_annotation_ids
        split_track.annotation_ids = split_annotation_ids
        source.identity_tombstone_ids = {
            annotation_id
            for annotation_id in source_annotation_ids
            if annotation_id in source.identity_tombstone_ids
            or is_identity_unbind_tombstone(annotations_by_id.get(annotation_id))
        }
        split_track.identity_tombstone_ids = {
            annotation_id
            for annotation_id in split_annotation_ids
            if annotation_id in split_track.identity_tombstone_ids
            or is_identity_unbind_tombstone(annotations_by_id.get(annotation_id))
        }
        source.manual_external_player_id = source_external_player_id
        split_track.manual_external_player_id = split_external_player_id
        source.roster_binding_state = source_roster_binding_state
        split_track.roster_binding_state = split_roster_binding_state
        source.roster_binding_annotation_ids.intersection_update(
            source_annotation_ids
        )
        split_track.roster_binding_annotation_ids.intersection_update(
            split_annotation_ids
        )
        source_kind, source_label, source_has_positive_semantics = (
            partition_manual_semantics(source_annotation_ids, annotations_by_id)
        )
        split_kind, split_label, _ = partition_manual_semantics(
            split_annotation_ids, annotations_by_id
        )
        original_has_known_semantics = any(
            annotation_id in annotations_by_id
            and annotation_action(annotations_by_id[annotation_id])
            in {"confirm", "merge", "split"}
            for annotation_id in original_annotation_ids
        )
        if source_has_positive_semantics:
            source.manual_kind = source_kind
            source.manual_label = source_label
        elif original_has_known_semantics:
            source.manual_kind = None
            source.manual_label = None
        split_track.manual_kind = split_kind or str(
            annotation.get("kind") or "other"
        )
        split_track.manual_label = split_label
        for partition_name, partition_track in (
            ("remaining", source),
            ("range", split_track),
        ):
            if (
                partition_track.roster_binding_state == "bound"
                and not bound_roster_semantics_compatible(
                    partition_track.manual_kind,
                    partition_track.annotation_ids,
                    annotations_by_id,
                )
            ):
                raise IdentityCorrectionError(
                    f"Split correction {correction_id} gives its bound roster partition incompatible team or role semantics",
                    correction_id=correction_id,
                    action="split",
                    status="conflict",
                    reason="bound-roster-partition-semantics-conflict",
                    source_track_id=source_identity_id,
                    target_id=(
                        source_identity_id
                        if partition_name == "remaining"
                        else split_identity_id
                    ),
                    candidates=[
                        {
                            "partition": partition_name,
                            "kind": partition_track.manual_kind,
                            "externalPlayerId": partition_track.manual_external_player_id,
                        }
                    ],
                )
        source.canonical_person_id = source_identity_id
        split_track.canonical_person_id = split_identity_id
        source.manual_identity_owner_ids = {source_identity_id}
        split_track.manual_identity_owner_ids = {split_identity_id}
        source.identity_split_partitions[correction_id] = "remaining"
        split_track.identity_split_partitions[correction_id] = "range"
        split_track.identity_group_id = split_identity_id
        split_track.identity_status = "resolved"
        split_track.identity_confidence = 1.0
        # The selected range is a new manual identity partition, not another
        # vote for the source tracklet's jersey/roster prior.
        split_track.source_tracklet_ids = {split_track.local_tracklet_id}
        split_track.identity_evidence = []
        split_track.identity_conflicts = []
        evidence = {
            "id": f"{correction_id}:manual-split",
            "kind": "manual",
            "label": "Manual identity split",
            "value": f"[{start:.3f}, {end:.3f})",
            "supportCount": len(inside),
            "source": "identity-correction",
            "manual": True,
        }
        source.identity_evidence.append({**evidence, "partition": "remaining"})
        split_track.identity_evidence.append({**evidence, "partition": "range"})
        refresh_split_track_state(source)
        refresh_split_track_state(split_track)
        result.append(split_track)
        applied.append(
            {
                "correctionId": correction_id,
                "sourceCanonicalPersonId": source_identity_id,
                "splitCanonicalPersonId": split_identity_id,
                "rangeStart": round(start, 3),
                "rangeEnd": round(end, 3),
                "affectedObservationCount": len(inside),
                "remainingObservationCount": len(outside),
                "targetObservationId": annotation.get("targetObservationId"),
            }
        )
    return (
        sorted(result, key=lambda track: (float(track.points[0]["t"]), track.id)),
        {"appliedCount": len(applied), "applied": applied},
    )
