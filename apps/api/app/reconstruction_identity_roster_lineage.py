from __future__ import annotations

"""Correction-lineage queries for durable roster decisions."""

from .reconstruction_identity_correction_graph import (
    canonical_correction_identity_key,
    correction_endpoint_ids,
    terminal_identity_target,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
)


def active_merge_dependencies(
    scene: dict, person: dict, annotations: list[dict]
) -> set[str]:
    """Return merge corrections currently represented by a published person.

    A roster decision authored after a merge cannot later be assigned to either
    pre-merge identity merely from its persisted canonical owner. Recording the
    active merge ids at authoring time gives undo a durable fail-closed gate.
    """

    person_annotation_ids = {
        str(value).strip()
        for value in person.get("annotationIds") or []
        if str(value or "").strip()
    }
    canonical_person_id = str(
        person.get("canonicalPersonId") or person.get("id") or ""
    ).strip()
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    dependencies: set[str] = set()
    for annotation in annotations:
        if not annotation.get("id") or annotation_action(annotation) != "merge":
            continue
        correction_id = str(annotation["id"])
        if correction_id in person_annotation_ids:
            dependencies.add(correction_id)
            continue
        terminal_id = terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        terminal_annotation = annotation_by_id.get(terminal_id)
        target_owner = canonical_correction_identity_key(
            scene,
            annotation_by_id,
            annotation_source_identity(terminal_annotation)
            if terminal_annotation is not None
            else terminal_id,
        )
        if canonical_person_id and target_owner == canonical_person_id:
            dependencies.add(correction_id)
    return dependencies


def active_split_dependencies(
    scene: dict, person: dict, annotations: list[dict]
) -> set[str]:
    """Return direct split branches and the selected partition's ancestors."""

    person_annotation_ids = {
        str(value).strip()
        for value in person.get("annotationIds") or []
        if str(value or "").strip()
    }
    canonical_person_id = str(
        person.get("canonicalPersonId") or person.get("id") or ""
    ).strip()
    split_endpoints: dict[str, tuple[set[str], set[str]]] = {}
    for correction in annotations:
        if not correction.get("id") or annotation_action(correction) != "split":
            continue
        correction_id = str(correction["id"])
        split_endpoints[correction_id] = correction_endpoint_ids(
            scene,
            correction,
            annotations,
        )

    dependencies = set(person_annotation_ids) & set(split_endpoints)
    dependencies.update(
        correction_id
        for correction_id, (source_ids, target_ids) in split_endpoints.items()
        if canonical_person_id in source_ids | target_ids
    )

    # Walk only toward ancestors. An undirected closure would absorb sibling
    # partitions and let Clear on one branch erase a distinct sibling branch.
    ancestor_identity_ids = {canonical_person_id}
    for correction_id in dependencies:
        ancestor_identity_ids.update(split_endpoints[correction_id][0])
    changed = True
    while changed:
        changed = False
        for correction_id, (source_ids, target_ids) in split_endpoints.items():
            if correction_id in dependencies or not (
                target_ids & ancestor_identity_ids
            ):
                continue
            dependencies.add(correction_id)
            ancestor_identity_ids.update(source_ids)
            changed = True
    return dependencies
