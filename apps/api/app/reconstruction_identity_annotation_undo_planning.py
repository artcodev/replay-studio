from __future__ import annotations

"""Pure frame-level identity annotation mutation planning."""

from copy import deepcopy
from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    ROSTER_DECISION_ORIGIN_FIELD,
    ROSTER_IDENTITY_DEPENDENCIES_FIELD,
    SPLIT_ROSTER_UNDO_FIELD,
)
from .reconstruction_identity_correction_graph import (
    canonical_correction_identity_key,
    terminal_identity_target,
)
from .reconstruction_identity_roster_corrections import (
    canonical_roster_binding_annotation_id,
    dedicated_roster_corrections_by_owner,
    roster_decision_origin_id,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
)
from .reconstruction_identity_reference_cleanup import remove_annotation_references
from .reconstruction_identity_validation import validate_identity_corrections


def plan_frame_person_annotation_delete(
    scene: dict,
    annotation_id: str,
) -> dict:
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    annotation = next((item for item in annotations if item.get("id") == annotation_id), None)
    if annotation is None:
        raise ReconstructionError("Frame annotation was not found")
    if annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION:
        raise ReconstructionError(
            "Canonical roster corrections can only be changed through Bind / Unbind / Clear"
        )
    dependent_unbound_ids: set[str] = set()
    restored_roster_corrections: list[dict] = []
    if annotation_action(annotation) in {"split", "merge"}:
        if annotation_action(annotation) == "merge":
            post_merge_roster_corrections = [
                item
                for item in annotations
                if item.get("correctionKind")
                == CANONICAL_ROSTER_BINDING_CORRECTION
                and str(annotation_id)
                in {
                    str(value)
                    for value in item.get(ROSTER_IDENTITY_DEPENDENCIES_FIELD)
                    or []
                }
            ]
            if post_merge_roster_corrections:
                correction_ids = ", ".join(
                    sorted(
                        str(item.get("id") or "unknown")
                        for item in post_merge_roster_corrections
                    )
                )
                raise ReconstructionError(
                    "A roster decision was created or changed after this merge "
                    "and cannot be distributed safely. Unbind any bound player, "
                    "then delete the roster correction before undoing the merge: "
                    + correction_ids
                )
        dependency_keys = {str(annotation_id)}
        if annotation_action(annotation) == "split":
            produced_identity = str(
                annotation.get("splitCanonicalPersonId") or ""
            ).strip()
            if produced_identity:
                dependency_keys.add(produced_identity)
        direct_dependents = []
        for item in annotations:
            if item is annotation:
                continue
            references = {
                str(value).strip()
                for value in (
                    annotation_source_identity(item),
                    item.get("mergeTargetId")
                    if annotation_action(item) == "merge"
                    else None,
                )
                if str(value or "").strip()
            }
            if references & dependency_keys:
                direct_dependents.append(item)
        blocking_dependents = [
            item
            for item in direct_dependents
            if not (
                item.get("correctionKind")
                == CANONICAL_ROSTER_BINDING_CORRECTION
                and item.get("rosterBindingState") == "unbound"
            )
        ]
        if blocking_dependents:
            dependent_ids = ", ".join(
                sorted(str(item.get("id") or "unknown") for item in blocking_dependents)
            )
            raise ReconstructionError(
                "Delete dependent identity corrections before undoing this split or merge: "
                + dependent_ids
            )
        annotation_by_id = {
            str(item.get("id")): item for item in annotations if item.get("id")
        }
        affected_identity_ids = {
            value
            for value in (
                canonical_correction_identity_key(
                    scene,
                    annotation_by_id,
                    annotation_source_identity(annotation),
                ),
                canonical_correction_identity_key(
                    scene,
                    annotation_by_id,
                    str(annotation.get("splitCanonicalPersonId") or "")
                    if annotation_action(annotation) == "split"
                    else terminal_identity_target(
                        str(annotation.get("mergeTargetId") or ""),
                        annotation_by_id,
                    ),
                ),
            )
            if value
        }
        dependent = [
            item
            for owner, rows in dedicated_roster_corrections_by_owner(
                scene, annotations
            ).items()
            if owner in affected_identity_ids
            for item in rows
        ]
        if any(item.get("rosterBindingState") == "bound" for item in dependent):
            raise ReconstructionError(
                "Unbind roster players on the affected identities before undoing this split or merge"
            )
        if annotation_action(annotation) == "split":
            dependent_unbound_ids = {
                str(item.get("id")) for item in dependent if item.get("id")
            }
            stored_rows = annotation.get(SPLIT_ROSTER_UNDO_FIELD) or []
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The split has invalid roster undo metadata"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The split has unsafe roster undo metadata; rebuild before deleting it"
                    )
            candidates = [deepcopy(item) for item in stored_rows] or [
                deepcopy(item)
                for item in dependent
                if item.get("rosterBindingState") == "unbound"
            ]
            if candidates:
                # All explicit Unbind rows carry the same compatible negative
                # decision. Prefer the exact pre-split snapshot when available,
                # otherwise retain one deterministic current row and migrate it
                # to the recombined source identity instead of deleting it.
                retained = min(
                    candidates,
                    key=lambda item: (
                        roster_decision_origin_id(item),
                        str(item.get("id") or ""),
                    ),
                )
                source_owner = str(
                    canonical_correction_identity_key(
                        scene,
                        annotation_by_id,
                        annotation_source_identity(annotation),
                    )
                    or annotation_source_identity(annotation)
                    or ""
                ).strip()
                if not source_owner:
                    raise ReconstructionError(
                        "The split source identity is missing from roster undo metadata"
                    )
                retained_id = canonical_roster_binding_annotation_id(source_owner)
                retained["id"] = retained_id
                retained["canonicalPersonId"] = source_owner
                retained["sourceTrackId"] = None
                retained[ROSTER_DECISION_ORIGIN_FIELD] = (
                    roster_decision_origin_id(retained) or retained_id
                )
                retained[ROSTER_IDENTITY_DEPENDENCIES_FIELD] = sorted(
                    {
                        str(value)
                        for value in retained.get(
                            ROSTER_IDENTITY_DEPENDENCIES_FIELD
                        )
                        or []
                        if str(value) != str(annotation_id)
                    }
                )
                target_observation = retained.get("targetObservation")
                if isinstance(target_observation, dict):
                    target_observation = deepcopy(target_observation)
                    target_observation["canonicalPersonId"] = source_owner
                    target_observation["annotationId"] = retained_id
                    target_observation["annotationIds"] = sorted(
                        {
                            str(value)
                            for value in target_observation.get("annotationIds")
                            or []
                            if str(value) not in dependent_unbound_ids
                        }
                        | {retained_id}
                    )
                    retained["targetObservation"] = target_observation
                restored_roster_corrections.append(retained)
        else:
            stored_rows = annotation.get("consolidatedRosterCorrections") or []
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The merge has invalid roster undo metadata"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The merge has unsafe roster undo metadata; rebuild before deleting it"
                    )
                restored_roster_corrections.append(deepcopy(stored))
    remaining = [
        item
        for item in annotations
        if item.get("id") != annotation_id
        and str(item.get("id") or "") not in dependent_unbound_ids
    ]
    existing_ids = {str(item.get("id") or "") for item in remaining}
    conflicting_restore_ids = sorted(
        str(item["id"])
        for item in restored_roster_corrections
        if str(item["id"]) in existing_ids
    )
    if conflicting_restore_ids:
        raise ReconstructionError(
            "Roster undo metadata conflicts with current corrections: "
            + ", ".join(conflicting_restore_ids)
        )
    remaining.extend(restored_roster_corrections)
    remaining.sort(
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or ""))
    )
    validate_identity_corrections(scene, remaining)
    reconstruction["frameAnnotations"] = remaining
    video["reconstruction"] = reconstruction
    restored_ids = {
        str(item.get("id"))
        for item in restored_roster_corrections
        if item.get("id")
    }
    remove_annotation_references(
        scene, dependent_unbound_ids - restored_ids
    )
    return annotation
