from __future__ import annotations

"""Identity editor defaults and API response projections."""

from copy import deepcopy

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_correction_graph import terminal_identity_target
from .reconstruction_identity_read_model import canonical_analysis_subjects
from .reconstruction_identity_semantics import annotation_action, annotation_scope

def track_annotation_kind(track: dict) -> str:
    team = track.get("teamId")
    role = track.get("role")
    if role == "referee" or team == "officials":
        return "referee"
    if role == "other" or team == "unknown":
        return "other"
    if team == "away":
        return "away-goalkeeper" if role == "goalkeeper" else "away-player"
    return "home-goalkeeper" if role == "goalkeeper" else "home-player"


def identity_target_defaults(
    scene: dict,
    annotations: list[dict],
    target_id: str,
) -> tuple[str, str | None, str | None]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    terminal_id = terminal_identity_target(target_id, annotation_by_id)
    target_annotation = annotation_by_id.get(terminal_id)
    if target_annotation is not None:
        return (
            str(target_annotation.get("kind") or "other"),
            target_annotation.get("label"),
            target_annotation.get("externalPlayerId"),
        )
    target_track = next(
        (
            track
            for track in canonical_analysis_subjects(scene)
            if str(track.get("id") or "") == terminal_id
            or str(track.get("canonicalPersonId") or "") == terminal_id
        ),
        None,
    )
    if target_track is None:
        raise ReconstructionError("The merge target no longer exists")
    return (
        track_annotation_kind(target_track),
        target_track.get("displayName") or target_track.get("label"),
        target_track.get("externalPlayerId"),
    )


def identity_annotation_response(annotation: dict) -> dict:
    action = annotation_action(annotation)
    return {
        **annotation,
        "action": action,
        "scope": annotation_scope(annotation),
        "mergeTargetId": annotation.get("mergeTargetId") if action == "merge" else None,
        "sourceTrackId": annotation.get("sourceTrackId"),
        "canonicalPersonId": annotation.get("canonicalPersonId"),
        "targetObservationId": annotation.get("targetObservationId") if action == "split" else None,
        "targetObservation": deepcopy(annotation.get("targetObservation")) if action == "split" else None,
        "rangeStart": annotation.get("rangeStart") if action == "split" else None,
        "rangeEnd": annotation.get("rangeEnd") if action == "split" else None,
        "splitCanonicalPersonId": annotation.get("splitCanonicalPersonId") if action == "split" else None,
        "affectedPreview": deepcopy(annotation.get("affectedPreview")) if action == "split" else None,
        "previewState": {
            "confirm": "confirmed",
            "exclude": "excluded",
            "merge": "merged",
            "split": "split",
        }[action],
    }
