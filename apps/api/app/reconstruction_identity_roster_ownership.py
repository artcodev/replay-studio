from __future__ import annotations

"""Fail-closed ownership queries for canonical roster decisions."""

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import CANONICAL_ROSTER_BINDING_CORRECTION
from .reconstruction_identity_roster_corrections import (
    roster_binding_correction_owner_ids,
)
from .reconstruction_identity_semantics import annotation_action, annotation_scope


def canonical_person_for_binding(scene: dict, canonical_person_id: str) -> dict:
    people = scene.get("payload", {}).get("canonicalPeople") or []
    person = next(
        (
            item
            for item in people
            if str(item.get("canonicalPersonId") or item.get("id") or "")
            == canonical_person_id
        ),
        None,
    )
    if person is None:
        raise ReconstructionError("The canonical person no longer exists")
    return person


def ensure_external_player_is_available(
    scene: dict,
    annotations: list[dict],
    canonical_person_id: str,
    external_player_id: str,
) -> None:
    conflicting_person = next(
        (
            item
            for item in scene.get("payload", {}).get("canonicalPeople") or []
            if str(item.get("canonicalPersonId") or item.get("id") or "")
            != canonical_person_id
            and str(item.get("externalPlayerId") or "") == external_player_id
        ),
        None,
    )
    if conflicting_person is not None:
        raise ReconstructionError(
            "The selected roster player is already bound to another canonical person"
        )

    conflicting_correction = next(
        (
            item
            for item in annotations
            if annotation_action(item) == "confirm"
            and annotation_scope(item) == "identity"
            and str(item.get("externalPlayerId") or "") == external_player_id
            and str(item.get("canonicalPersonId") or "") != canonical_person_id
            and roster_binding_correction_owner_ids(scene, item)
            != {canonical_person_id}
        ),
        None,
    )
    if conflicting_correction is not None:
        raise ReconstructionError(
            "The selected roster player is already bound to another canonical person"
        )


def roster_correction_index_for_set(
    scene: dict,
    annotations: list[dict],
    canonical_person_id: str,
    desired_annotation_id: str,
) -> int | None:
    """Resolve exactly one editable correction or fail on ambiguous ownership."""

    owned_indices: list[int] = []
    for index, item in enumerate(annotations):
        if item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = roster_binding_correction_owner_ids(scene, item)
        persisted_owner = str(item.get("canonicalPersonId") or "").strip()
        if canonical_person_id in owners and owners != {canonical_person_id}:
            raise ReconstructionError(
                "The roster correction anchor is owned by multiple canonical people; rebuild before editing"
            )
        if owners == {canonical_person_id} or (
            not owners and persisted_owner == canonical_person_id
        ):
            owned_indices.append(index)
    if len(owned_indices) > 1:
        raise ReconstructionError(
            "This canonical person has multiple durable roster corrections; rebuild before editing"
        )

    desired_id_index = next(
        (
            index
            for index, item in enumerate(annotations)
            if str(item.get("id") or "") == desired_annotation_id
        ),
        None,
    )
    existing_index = owned_indices[0] if owned_indices else desired_id_index
    if existing_index is not None and existing_index not in owned_indices:
        candidate = annotations[existing_index]
        existing_owners = roster_binding_correction_owner_ids(scene, candidate)
        persisted_owner = str(candidate.get("canonicalPersonId") or "")
        if (
            existing_owners
            and existing_owners != {canonical_person_id}
            or persisted_owner != canonical_person_id
        ):
            raise ReconstructionError(
                "The roster correction id is owned by another canonical person; edit that identity first"
            )
    if (
        owned_indices
        and desired_id_index is not None
        and desired_id_index != owned_indices[0]
    ):
        raise ReconstructionError(
            "The roster correction cannot be rekeyed because its target id already exists"
        )
    return existing_index


def roster_correction_for_clear(
    scene: dict,
    annotations: list[dict],
    canonical_person_id: str,
) -> dict:
    owned: list[dict] = []
    for item in annotations:
        if item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = roster_binding_correction_owner_ids(scene, item)
        persisted_owner = str(item.get("canonicalPersonId") or "").strip()
        if owners == {canonical_person_id} or (
            not owners and persisted_owner == canonical_person_id
        ):
            owned.append(item)
    if not owned:
        raise ReconstructionError(
            "This canonical person has no roster decision to clear"
        )
    if len(owned) > 1:
        raise ReconstructionError(
            "This canonical person has multiple durable roster corrections; rebuild before clearing"
        )
    correction = owned[0]
    if correction.get("rosterBindingState") != "unbound":
        raise ReconstructionError(
            "Unbind the roster player before clearing its roster decision"
        )
    return correction
