from __future__ import annotations

"""Dedicated roster-correction ownership, consolidation, and decisions."""

from copy import deepcopy
from hashlib import sha256

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    ROSTER_DECISION_ORIGIN_FIELD,
)
from .reconstruction_identity_correction_graph import (
    canonical_correction_identity_key,
    terminal_identity_target,
)
from .reconstruction_identity_read_model import bbox_payload_box
from .reconstruction_identity_semantics import annotation_source_identity
from .bounding_box_geometry import intersection_over_union

def canonical_roster_binding_annotation_id(canonical_person_id: str) -> str:
    digest = sha256(canonical_person_id.encode("utf-8")).hexdigest()[:16]
    return f"roster-binding-{digest}"


def dedicated_roster_corrections_by_owner(
    scene: dict,
    annotations: list[dict],
) -> dict[str, list[dict]]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    result: dict[str, list[dict]] = {}
    for annotation in annotations:
        if (
            annotation.get("correctionKind")
            != CANONICAL_ROSTER_BINDING_CORRECTION
            or annotation.get("rosterBindingState") not in {"bound", "unbound"}
        ):
            continue
        owners = roster_binding_correction_owner_ids(scene, annotation)
        if not owners:
            persisted_owner = annotation_source_identity(annotation)
            owners = {persisted_owner} if persisted_owner else set()
        for owner in owners:
            canonical_owner = canonical_correction_identity_key(
                scene, annotation_by_id, owner
            )
            if canonical_owner:
                result.setdefault(canonical_owner, []).append(annotation)
    return result


def roster_correction_decision(annotation: dict) -> tuple[str, str | None]:
    state = str(annotation.get("rosterBindingState") or "")
    external_id = str(annotation.get("externalPlayerId") or "").strip() or None
    return state, external_id if state == "bound" else None


def consolidate_compatible_merge_roster_corrections(
    scene: dict,
    annotations: list[dict],
    merge_annotation: dict,
) -> tuple[list[dict], list[dict]]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    source_key = canonical_correction_identity_key(
        scene,
        annotation_by_id,
        annotation_source_identity(merge_annotation),
    )
    terminal_id = terminal_identity_target(
        str(merge_annotation.get("mergeTargetId") or ""), annotation_by_id
    )
    terminal_annotation = annotation_by_id.get(terminal_id)
    target_key = canonical_correction_identity_key(
        scene,
        annotation_by_id,
        annotation_source_identity(terminal_annotation)
        if terminal_annotation is not None
        else terminal_id,
    )
    if not source_key or not target_key or source_key == target_key:
        return annotations, []
    by_owner = dedicated_roster_corrections_by_owner(scene, annotations)
    source_rows = by_owner.get(source_key, [])
    target_rows = by_owner.get(target_key, [])
    if not source_rows or not target_rows:
        return annotations, []
    decisions = {
        roster_correction_decision(item) for item in [*source_rows, *target_rows]
    }
    if len(decisions) != 1:
        raise ReconstructionError(
            "Cannot merge identities with different dedicated Bind / Unbind decisions"
        )
    decision = next(iter(decisions))
    if decision[0] == "bound":
        raise ReconstructionError(
            "Cannot merge two identities that both carry a dedicated roster binding; unbind one duplicate first"
        )
    keep = min(target_rows, key=lambda item: str(item.get("id") or ""))
    removed_ids = sorted(
        {
            str(item.get("id"))
            for item in [*source_rows, *target_rows]
            if item is not keep and item.get("id")
        }
    )
    if not removed_ids:
        return annotations, []
    removed = [
        deepcopy(item)
        for item in [*source_rows, *target_rows]
        if str(item.get("id") or "") in removed_ids
    ]
    return (
        [item for item in annotations if str(item.get("id") or "") not in removed_ids],
        removed,
    )


def roster_decision_origin_id(annotation: dict) -> str:
    """Return the stable lineage key retained while a decision is re-keyed."""

    return str(
        annotation.get(ROSTER_DECISION_ORIGIN_FIELD)
        or annotation.get("id")
        or ""
    ).strip()


def roster_binding_correction_owner_ids(scene: dict, correction: dict) -> set[str]:
    """Find canonical people that currently own a roster correction anchor.

    A post-resolver split intentionally changes that ownership without
    rewriting reconstruction input.  The next roster edit can therefore rekey
    the old correction transactionally instead of leaving a second positive
    binding behind.
    """

    correction_id = str(correction.get("id") or "").strip()
    target_observation_id = str(correction.get("targetObservationId") or "").strip()
    people = scene.get("payload", {}).get("canonicalPeople") or []
    strong_owners: set[str] = set()
    for person in people:
        canonical_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        observations = [
            item for item in person.get("observations") or [] if isinstance(item, dict)
        ]
        observation_ids = {
            str(item.get("observationId") or item.get("id") or "").strip()
            for item in observations
        }
        observation_annotation_ids = {
            str(item.get("annotationId") or "").strip() for item in observations
        }
        if (
            correction_id in set(person.get("annotationIds") or [])
            or correction_id in observation_annotation_ids
            or target_observation_id
            and target_observation_id in observation_ids
        ):
            strong_owners.add(canonical_id)
    if strong_owners:
        return strong_owners

    snapshot = correction.get("targetObservation")
    if not isinstance(snapshot, dict) or snapshot.get("frameIndex") is None:
        return set()
    snapshot_bbox = snapshot.get("bbox")
    if not isinstance(snapshot_bbox, dict):
        return set()
    try:
        frame_index = int(snapshot["frameIndex"])
        snapshot_time = float(snapshot.get("sceneTime"))
        target_box = bbox_payload_box(snapshot_bbox)
    except (KeyError, TypeError, ValueError):
        return set()
    geometric_owners: set[str] = set()
    for person in people:
        canonical_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        for observation in person.get("observations") or []:
            if not isinstance(observation, dict) or not observation.get("bbox"):
                continue
            try:
                if int(observation["frameIndex"]) != frame_index:
                    continue
                if abs(float(observation.get("sceneTime")) - snapshot_time) > 0.08:
                    continue
                overlap = intersection_over_union(
                    bbox_payload_box(observation["bbox"]),
                    target_box,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if overlap >= 0.75:
                geometric_owners.add(canonical_id)
                break
    return geometric_owners
