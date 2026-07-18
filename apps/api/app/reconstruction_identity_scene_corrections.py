from __future__ import annotations

"""Apply identity corrections to published scene track documents."""

from copy import deepcopy

from .reconstruction_errors import IdentityCorrectionError, ReconstructionError
from .reconstruction_identity_correction_graph import terminal_identity_target
from .reconstruction_latent_presence import materialize_continuous_presence
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_source_identity,
    identity_annotations,
)
from .reconstruction_identity_merging import confirmed_external_player_conflict
from .reconstruction_track_observations import merge_track_observations

def merge_scene_track_documents(
    target: dict,
    source: dict,
    annotation: dict,
    scene: dict,
) -> dict:
    conflict = confirmed_external_player_conflict(
        target.get("externalPlayerId"), source.get("externalPlayerId")
    )
    if conflict is not None:
        target_external_id, source_external_id = conflict
        raise ReconstructionError(
            "Cannot merge identities with different confirmed roster players: "
            f"{source_external_id} and {target_external_id}"
        )
    keyframes_by_time: dict[float, dict] = {}
    observed_keyframes = [
        keyframe
        for keyframe in [*(target.get("keyframes") or []), *(source.get("keyframes") or [])]
        if keyframe.get("observed") is not False
    ]
    for keyframe in observed_keyframes:
        key = round(float(keyframe["t"]), 4)
        previous = keyframes_by_time.get(key)
        if previous is None or float(keyframe.get("confidence") or 0.0) >= float(
            previous.get("confidence") or 0.0
        ):
            keyframes_by_time[key] = keyframe
    merged_from = set(
        (target.get("identityCorrection") or {}).get("mergedTrackIds") or []
    )
    if source.get("id") and source.get("id") != target.get("id"):
        merged_from.add(str(source["id"]))
    correction_annotations = set(
        (target.get("identityCorrection") or {}).get("annotationIds") or []
    )
    correction_annotations.add(str(annotation["id"]))
    merged_keyframes, presence = materialize_continuous_presence(
        [keyframes_by_time[key] for key in sorted(keyframes_by_time)],
        float(scene.get("duration") or 0.0),
        scene.get("payload", {}).get("pitch") or {"length": 105, "width": 68},
        sum((index + 1) * ord(character) for index, character in enumerate(str(target["id"]))),
    )
    merged_observations = merge_track_observations(
        list(target.get("observations") or []),
        list(source.get("observations") or []),
    )
    return {
        **target,
        "annotationIds": sorted(
            {
                *(target.get("annotationIds") or []),
                *(source.get("annotationIds") or []),
            }
        ),
        "identityCorrection": {
            "status": "merged",
            "targetId": str(target["id"]),
            "annotationIds": sorted(correction_annotations),
            "mergedTrackIds": sorted(merged_from),
        },
        "presence": presence,
        "observations": merged_observations,
        "keyframes": merged_keyframes,
    }


def apply_scene_track_identity_corrections(tracks: list[dict], scene: dict) -> list[dict]:
    """Publish one stable scene identity for every explicit merge directive."""

    result = [deepcopy(track) for track in tracks]
    annotations = identity_annotations(scene)
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    previous_tracks: dict[str, dict] = {}
    for track in scene.get("payload", {}).get("tracks") or []:
        for identifier in (track.get("id"), track.get("canonicalPersonId")):
            if identifier:
                previous_tracks[str(identifier)] = track
    def by_annotation(annotation_id: str) -> dict | None:
        return next(
            (
                track
                for track in result
                if annotation_id in (track.get("annotationIds") or [])
            ),
            None,
        )

    def crosses_manual_split_barrier(left: dict, right: dict) -> bool:
        left_partitions = left.get("identitySplitPartitions") or {}
        right_partitions = right.get("identitySplitPartitions") or {}
        return any(
            correction_id in right_partitions
            and right_partitions[correction_id] != partition
            for correction_id, partition in left_partitions.items()
        )

    for annotation in annotations:
        if annotation_action(annotation) != "merge" or not annotation.get("id"):
            continue
        source = by_annotation(str(annotation["id"]))
        if source is None:
            continue
        terminal_id = terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        target = (
            by_annotation(terminal_id)
            if terminal_id in annotation_by_id
            else next(
                (
                    track
                    for track in result
                    if str(track.get("id") or "") == terminal_id
                    or str(track.get("canonicalPersonId") or "") == terminal_id
                ),
                None,
            )
        )
        if target is source:
            source["identityCorrection"] = {
                "status": "merged",
                "targetId": str(source["id"]),
                "annotationIds": sorted(
                    {
                        *((source.get("identityCorrection") or {}).get("annotationIds") or []),
                        str(annotation["id"]),
                    }
                ),
                "mergedTrackIds": sorted(
                    set((source.get("identityCorrection") or {}).get("mergedTrackIds") or [])
                ),
            }
            continue
        if target is None and terminal_id in previous_tracks:
            previous = previous_tracks[terminal_id]
            target = {
                **source,
                **{
                    key: previous[key]
                    for key in (
                        "id",
                        "canonicalPersonId",
                        "label",
                        "teamId",
                        "color",
                        "number",
                        "role",
                        "externalPlayerId",
                    )
                    if key in previous
                },
            }
            result[result.index(source)] = target
        if target is None:
            continue
        if crosses_manual_split_barrier(source, target):
            # The post-resolver split is an explicit cannot-link. Older merge
            # corrections may still exist for audit/undo, but cannot reconnect
            # the two partitions in the published scene document.
            continue
        conflict = confirmed_external_player_conflict(
            target.get("externalPlayerId"), source.get("externalPlayerId")
        )
        if conflict is not None:
            target_external_id, source_external_id = conflict
            correction_id = str(annotation["id"])
            raise IdentityCorrectionError(
                (
                    f"Identity correction {correction_id} cannot merge confirmed roster "
                    f"players {source_external_id} and {target_external_id}"
                ),
                correction_id=correction_id,
                action="merge",
                status="conflict",
                reason="conflicting-confirmed-external-player-ids",
                source_track_id=annotation_source_identity(annotation),
                target_id=terminal_id,
                candidates=[
                    {
                        "trackId": source.get("id"),
                        "externalPlayerId": source_external_id,
                    },
                    {
                        "trackId": target.get("id"),
                        "externalPlayerId": target_external_id,
                    },
                ],
            )
        merged = merge_scene_track_documents(target, source, annotation, scene)
        target_index = result.index(target)
        result[target_index] = merged
        if source in result and source is not target:
            result.remove(source)
    return result
