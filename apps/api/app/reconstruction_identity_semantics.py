from __future__ import annotations

"""Identity annotation schema, parsing, and semantic invariants."""

import numpy as np

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import CANONICAL_ROSTER_BINDING_CORRECTION

def annotation_team(kind: str | None) -> str | None:
    if kind in {"home-player", "home-goalkeeper"}:
        return "home"
    if kind in {"away-player", "away-goalkeeper"}:
        return "away"
    if kind == "referee":
        return "officials"
    if kind == "other":
        return "unknown"
    return None


def annotation_role(kind: str | None) -> str | None:
    if kind in {"home-goalkeeper", "away-goalkeeper"}:
        return "goalkeeper"
    if kind == "referee":
        return "referee"
    if kind == "other":
        return "other"
    if kind in {"home-player", "away-player"}:
        return "player"
    return None


def annotation_action(annotation: dict) -> str:
    action = str(annotation.get("action") or "").strip().lower()
    if action in {"confirm", "exclude", "merge", "split"}:
        return action
    raise ReconstructionError("Identity correction action is required")


def annotation_scope(annotation: dict) -> str:
    scope = str(annotation.get("scope") or "").strip().lower()
    if scope in {"observation", "range", "identity"}:
        return scope
    raise ReconstructionError("Identity correction scope is required")


def is_identity_unbind_tombstone(annotation: dict | None) -> bool:
    """Return whether an annotation is an explicit negative roster decision."""

    return bool(
        annotation
        and annotation.get("correctionKind") == "canonical-roster-binding-v1"
        and annotation.get("rosterBindingState") == "unbound"
        and annotation.get("externalPlayerId") is None
    )


def annotation_manual_semantic_key(
    annotation: dict,
) -> tuple[int, str, int, str]:
    """Order role/label edits by authoring metadata, never video time."""

    updated_at = str(annotation.get("updatedAt") or "").strip()
    is_dedicated_roster = int(
        annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION
    )
    return (
        int(bool(updated_at)),
        updated_at,
        is_dedicated_roster,
        str(annotation.get("id") or ""),
    )


def identity_annotations(scene: dict) -> list[dict]:
    return list(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("frameAnnotations")
        or []
    )


def annotation_source_identity(annotation: dict | None) -> str | None:
    if not annotation:
        return None
    value = annotation.get("canonicalPersonId") or annotation.get("sourceTrackId")
    return str(value).strip() or None if value is not None else None


def split_range(annotation: dict) -> tuple[float, float] | None:
    if annotation_action(annotation) != "split":
        return None
    try:
        start = float(annotation["rangeStart"])
        end = float(annotation["rangeEnd"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([start, end]).all() or end <= start:
        return None
    return start, end


def observation_identifier(observation: dict) -> str | None:
    value = observation.get("observationId") or observation.get("id")
    return str(value).strip() or None if value is not None else None


def bound_roster_semantics_compatible(
    manual_kind: str | None,
    annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
) -> bool:
    """Check the local role/team against its durable bound roster anchor.

    A split may intentionally change the semantics of the newly-created
    partition.  It may not, however, carry a bound home/away player into a
    partition labelled as the other team, referee, or unknown.  The durable
    correction's kind is used as the team anchor so this also works while the
    canonical output is being rebuilt.
    """

    bound_rows = [
        annotations_by_id[annotation_id]
        for annotation_id in annotation_ids
        if annotation_id in annotations_by_id
        and annotations_by_id[annotation_id].get("correctionKind")
        == CANONICAL_ROSTER_BINDING_CORRECTION
        and annotations_by_id[annotation_id].get("rosterBindingState") == "bound"
    ]
    if not bound_rows:
        return True
    expected_teams = {
        team
        for row in bound_rows
        if (team := annotation_team(str(row.get("kind") or "")))
        in {"home", "away"}
    }
    requested_team = annotation_team(manual_kind)
    requested_role = annotation_role(manual_kind)
    return (
        len(expected_teams) == 1
        and requested_team in expected_teams
        and requested_role in {"player", "goalkeeper"}
    )
