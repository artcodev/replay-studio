from __future__ import annotations

"""Pure Clear planning across roster split/merge undo lineage."""

from copy import deepcopy

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    SPLIT_ROSTER_UNDO_FIELD,
)
from .reconstruction_identity_correction_graph import correction_endpoint_ids
from .reconstruction_identity_roster_corrections import roster_decision_origin_id
from .reconstruction_identity_semantics import annotation_action
from .reconstruction_identity_validation import validate_identity_corrections
from .reconstruction_identity_reference_cleanup import remove_annotation_references


def roster_undo_snapshot_owner_id(snapshot: dict) -> str | None:
    """Return one internally consistent persisted owner for an undo snapshot."""

    owner_ids = {
        str(value).strip()
        for value in (
            snapshot.get("canonicalPersonId"),
            (
                snapshot.get("targetObservation") or {}
            ).get("canonicalPersonId")
            if isinstance(snapshot.get("targetObservation"), dict)
            else None,
        )
        if str(value or "").strip()
    }
    return next(iter(owner_ids)) if len(owner_ids) == 1 else None


def plan_clear_unbound_roster_correction(
    scene: dict,
    annotation: dict,
    *,
    active_merge_ids: set[str] | None = None,
    active_split_ids: set[str] | None = None,
) -> dict:
    """Plan removal of one Unbind and every active undo resurrection path."""

    if (
        annotation.get("correctionKind")
        != CANONICAL_ROSTER_BINDING_CORRECTION
        or annotation.get("rosterBindingState") != "unbound"
        or not annotation.get("id")
    ):
        raise ReconstructionError(
            "Unbind the roster player before clearing its roster decision"
        )
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    annotation_id = str(annotation["id"])
    cleared_annotation_ids = {annotation_id}
    cleared_origin_ids = {roster_decision_origin_id(annotation)}
    active_merge_ids = {
        str(value).strip()
        for value in active_merge_ids or set()
        if str(value or "").strip()
    }
    active_split_ids = {
        str(value).strip()
        for value in active_split_ids or set()
        if str(value or "").strip()
    }

    # A merge of two compatible Unbind decisions keeps one live correction and
    # stores the others solely for Undo Merge.  From the merged identity those
    # rows are one semantic negative decision, so Clear must remove the complete
    # active lineage.  Validate the snapshots before changing the scene so
    # malformed metadata fails closed instead of being partially discarded.
    for item in annotations:
        item_id = str(item.get("id") or "")
        if item_id not in active_merge_ids:
            continue
        has_rows = "consolidatedRosterCorrections" in item
        has_ids = "consolidatedRosterCorrectionIds" in item
        if has_ids and not has_rows:
            raise ReconstructionError(
                "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
            )
        if not has_rows:
            continue
        merge_source_ids, merge_target_ids = correction_endpoint_ids(
            scene,
            item,
            annotations,
        )
        merge_owner_ids = merge_source_ids | merge_target_ids
        if not merge_source_ids or not merge_target_ids:
            raise ReconstructionError(
                "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
            )
        stored_rows = item.get("consolidatedRosterCorrections")
        if not isinstance(stored_rows, list):
            raise ReconstructionError(
                "The merge has invalid roster undo metadata"
            )
        stored_ids: list[str] = []
        for stored in stored_rows:
            if not isinstance(stored, dict) or (
                stored.get("correctionKind")
                != CANONICAL_ROSTER_BINDING_CORRECTION
                or stored.get("rosterBindingState") != "unbound"
                or not stored.get("id")
            ):
                raise ReconstructionError(
                    "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            stored_owner_id = roster_undo_snapshot_owner_id(stored)
            if not stored_owner_id or stored_owner_id not in merge_owner_ids:
                raise ReconstructionError(
                    "The merge roster undo metadata belongs to another identity; rebuild before clearing the roster decision"
                )
            stored_id = str(stored["id"])
            stored_ids.append(stored_id)
            cleared_annotation_ids.add(stored_id)
            cleared_origin_ids.add(roster_decision_origin_id(stored))
        if has_ids:
            metadata_ids = item.get("consolidatedRosterCorrectionIds")
            if not isinstance(metadata_ids, list) or any(
                not str(value or "").strip() for value in metadata_ids
            ):
                raise ReconstructionError(
                    "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            if sorted(set(stored_ids)) != sorted(
                {str(value).strip() for value in metadata_ids}
            ):
                raise ReconstructionError(
                    "The merge has inconsistent roster undo metadata; rebuild before clearing the roster decision"
                )

    remaining = []
    for item in annotations:
        if str(item.get("id") or "") == annotation_id:
            continue
        item = deepcopy(item)
        if (
            str(item.get("id") or "") in active_split_ids
            and annotation_action(item) == "split"
            and SPLIT_ROSTER_UNDO_FIELD in item
        ):
            stored_rows = item.get(SPLIT_ROSTER_UNDO_FIELD)
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The split has invalid roster undo metadata"
                )
            split_source_ids, _split_target_ids = correction_endpoint_ids(
                scene,
                item,
                annotations,
            )
            if not split_source_ids:
                raise ReconstructionError(
                    "The split has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The split has unsafe roster undo metadata; rebuild before clearing the roster decision"
                    )
                stored_owner_id = roster_undo_snapshot_owner_id(stored)
                if not stored_owner_id or stored_owner_id not in split_source_ids:
                    raise ReconstructionError(
                        "The split roster undo metadata belongs to another identity; rebuild before clearing the roster decision"
                    )
                if roster_decision_origin_id(stored) in cleared_origin_ids:
                    cleared_annotation_ids.add(str(stored["id"]))
            item[SPLIT_ROSTER_UNDO_FIELD] = [
                stored
                for stored in stored_rows
                if roster_decision_origin_id(stored)
                not in cleared_origin_ids
            ]
        if (
            str(item.get("id") or "") in active_merge_ids
            and annotation_action(item) == "merge"
        ):
            item.pop("consolidatedRosterCorrectionIds", None)
            item.pop("consolidatedRosterCorrections", None)
        remaining.append(item)
    validate_identity_corrections(scene, remaining)
    reconstruction["frameAnnotations"] = sorted(
        remaining,
        key=lambda item: (
            int(item.get("frameIndex") or 0),
            str(item.get("id") or ""),
        ),
    )
    video["reconstruction"] = reconstruction
    remove_annotation_references(scene, cleared_annotation_ids)
    return annotation
