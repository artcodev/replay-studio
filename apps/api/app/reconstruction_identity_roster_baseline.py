from __future__ import annotations

"""Partition-local identity baseline used when a roster decision is removed."""

from .reconstruction_identity_contract import CANONICAL_ROSTER_BINDING_CORRECTION
from .reconstruction_identity_semantics import annotation_action


def partition_local_identity_baseline(
    person: dict,
    annotations: list[dict],
) -> tuple[str, str, float | None, str]:
    local_annotation_ids = {str(value) for value in person.get("annotationIds") or []}
    positive = [
        annotation
        for annotation in annotations
        if str(annotation.get("id") or "") in local_annotation_ids
        and annotation.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
        and annotation_action(annotation) in {"confirm", "merge", "split"}
        and annotation.get("kind") != "ignore"
    ]
    positive.sort(
        key=lambda item: (
            str(item.get("updatedAt") or ""),
            float(item.get("sceneTime") or 0.0),
            str(item.get("id") or ""),
        )
    )
    display_name = next(
        (
            str(item.get("label") or "").strip()
            for item in reversed(positive)
            if str(item.get("label") or "").strip()
        ),
        "",
    )
    if not display_name:
        team = str(person.get("teamId") or "").strip()
        role = str(person.get("role") or "player").strip()
        display_name = (
            "Referee"
            if role == "referee"
            else "Other person"
            if role == "other"
            else f"{team.title()} goalkeeper"
            if team in {"home", "away"} and role == "goalkeeper"
            else f"{team.title()} person"
            if team in {"home", "away"}
            else "Unassigned person"
        )
    if positive:
        return display_name, "resolved", 1.0, "manual"
    if person.get("identitySource") != "manual":
        return (
            display_name,
            str(person.get("identityStatus") or "provisional"),
            person.get("identityConfidence"),
            str(person.get("identitySource") or "tracker+trajectory"),
        )
    return display_name, "provisional", None, "tracker+trajectory"
