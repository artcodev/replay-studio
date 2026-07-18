from __future__ import annotations

"""Apply identity-level exclude and merge corrections to rebuilt tracks."""

from .reconstruction_errors import IdentityCorrectionError
from .reconstruction_track_state import TrackState
from .reconstruction_identity_correction_graph import terminal_identity_target
from .reconstruction_identity_read_model import canonical_analysis_subjects
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_scope,
    annotation_source_identity,
    identity_annotations,
)
from .reconstruction_identity_merging import (
    merge_raw_track_states,
    raise_manual_merge_external_player_conflict,
)
from .reconstruction_identity_remapping import resolve_previous_identity_track

def apply_track_identity_corrections(tracks: list[TrackState], scene: dict) -> list[TrackState]:
    """Resolve explicit identity merges after online association and before QA."""

    result = list(tracks)
    annotations = identity_annotations(scene)
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    scene_tracks: dict[str, dict] = {}
    for track in canonical_analysis_subjects(scene):
        for identifier in (track.get("id"), track.get("canonicalPersonId")):
            if identifier:
                scene_tracks[str(identifier)] = track
    excluded_track_ids = {
        str(annotation_source_identity(annotation))
        for annotation in annotations
        if annotation_action(annotation) == "exclude"
        and annotation_scope(annotation) == "identity"
        and annotation_source_identity(annotation)
    }
    for track_id in excluded_track_ids:
        correction = next(
            annotation
            for annotation in annotations
            if annotation_action(annotation) == "exclude"
            and annotation_scope(annotation) == "identity"
            and str(annotation_source_identity(annotation) or "") == track_id
        )
        correction_id = str(correction.get("id") or track_id)
        exact = [
            track for track in result if correction_id in track.annotation_ids
        ]
        if len(exact) == 1:
            result.remove(exact[0])
            continue
        if len(exact) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} matched multiple raw tracks",
                correction_id=correction_id,
                action="exclude",
                status="ambiguous",
                reason="multiple-exact-source-anchors",
                source_track_id=track_id,
                candidates=[{"rawTrackId": track.id} for track in exact],
            )
        previous = scene_tracks.get(track_id)
        if previous is None:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} references a missing source track",
                correction_id=correction_id,
                action="exclude",
                status="unresolved",
                reason="missing-source-track",
                source_track_id=track_id,
            )
        result.remove(
            resolve_previous_identity_track(
                result,
                previous,
                correction_id=correction_id,
                action="exclude",
                source_track_id=track_id,
                target_id=track_id,
            )
        )
    for annotation in annotations:
        if annotation_action(annotation) != "merge" or not annotation.get("id"):
            continue
        source = next(
            (
                track
                for track in result
                if str(annotation["id"]) in track.annotation_ids
            ),
            None,
        )
        if source is None:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} did not attach to a raw source track",
                correction_id=str(annotation["id"]),
                action="merge",
                status="unresolved",
                reason="missing-source-anchor",
                source_track_id=annotation_source_identity(annotation),
                target_id=str(annotation.get("mergeTargetId") or "") or None,
            )
        terminal_id = terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        exact_targets = [
            track for track in result if terminal_id in track.annotation_ids
        ]
        if len(exact_targets) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} matched multiple merge targets",
                correction_id=str(annotation["id"]),
                action="merge",
                status="ambiguous",
                reason="multiple-exact-merge-targets",
                source_track_id=annotation_source_identity(annotation),
                target_id=terminal_id,
                candidates=[{"rawTrackId": track.id} for track in exact_targets],
            )
        target = exact_targets[0] if exact_targets else None
        if target is None and terminal_id in scene_tracks:
            target = resolve_previous_identity_track(
                result,
                scene_tracks[terminal_id],
                correction_id=str(annotation["id"]),
                action="merge",
                source_track_id=annotation_source_identity(annotation),
                target_id=terminal_id,
                exclude=source,
            )
        if target is None:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} could not resolve its merge target",
                correction_id=str(annotation["id"]),
                action="merge",
                status="unresolved",
                reason="missing-merge-target",
                source_track_id=annotation_source_identity(annotation),
                target_id=terminal_id,
            )
        if target is source:
            continue
        raise_manual_merge_external_player_conflict(target, source, annotation)
        terminal_annotation = annotation_by_id.get(terminal_id)
        terminal_subject = scene_tracks.get(terminal_id) or (
            scene_tracks.get(str(annotation_source_identity(terminal_annotation) or ""))
            if terminal_annotation is not None
            else None
        )
        target_owner_id = str(
            (terminal_subject or {}).get("canonicalPersonId")
            or annotation_source_identity(terminal_annotation)
            or terminal_id
        ).strip()
        merge_raw_track_states(
            target,
            source,
            allow_manual_owner_merge=True,
            manual_target_owner_id=target_owner_id or None,
        )
        result.remove(source)
    return result
