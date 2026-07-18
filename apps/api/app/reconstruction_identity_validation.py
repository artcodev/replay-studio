from __future__ import annotations

"""Fail-closed validation for the persisted identity-correction graph."""

from math import isfinite

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_correction_graph import (
    canonical_correction_identity_key,
    ordered_split_corrections,
    terminal_identity_target,
)
from .reconstruction_identity_read_model import canonical_analysis_subjects
from .reconstruction_identity_roster_corrections import (
    dedicated_roster_corrections_by_owner,
    roster_correction_decision,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_scope,
    annotation_source_identity,
    bound_roster_semantics_compatible,
    observation_identifier,
    split_range,
)

def validate_identity_corrections(scene: dict, annotations: list[dict]) -> None:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    identity_subjects = canonical_analysis_subjects(scene)
    subjects_by_id: dict[str, dict] = {}
    for subject in identity_subjects:
        for identifier in (subject.get("id"), subject.get("canonicalPersonId")):
            if identifier:
                subjects_by_id[str(identifier)] = subject
    track_ids = {
        str(identifier)
        for track in identity_subjects
        for identifier in (track.get("id"), track.get("canonicalPersonId"))
        if identifier
    }
    excluded_track_ids = {
        str(annotation_source_identity(annotation))
        for annotation in annotations
        if annotation_action(annotation) == "exclude"
        and annotation_scope(annotation) == "identity"
        and annotation_source_identity(annotation)
    }
    split_ranges: dict[str, list[tuple[float, float, str]]] = {}
    split_target_ids: dict[str, str] = {}
    dedicated_by_owner = dedicated_roster_corrections_by_owner(
        scene, annotations
    )
    ordered_splits = ordered_split_corrections(annotations)
    produced_split_ids = {
        str(annotation.get("splitCanonicalPersonId") or "").strip()
        for annotation in ordered_splits
        if str(annotation.get("splitCanonicalPersonId") or "").strip()
    }
    for annotation in annotations:
        referenced_ids = {
            str(value).strip()
            for value in (
                annotation_source_identity(annotation),
                annotation.get("mergeTargetId")
                if annotation_action(annotation) == "merge"
                else None,
            )
            if str(value or "").strip()
        }
        orphaned = sorted(
            value
            for value in referenced_ids
            if value.startswith("canonical-split-")
            and value not in produced_split_ids
            and value not in annotation_by_id
        )
        if orphaned:
            raise ReconstructionError(
                "Identity correction references a split identity whose parent correction is missing: "
                + ", ".join(orphaned)
            )
    for annotation_id, annotation in annotation_by_id.items():
        action = annotation_action(annotation)
        if (
            action == "exclude"
            and annotation_scope(annotation) == "identity"
            and not annotation_source_identity(annotation)
        ):
            raise ReconstructionError(
                "Choose the tracked identity before excluding the whole trajectory"
            )
        if action == "split":
            source_id = annotation_source_identity(annotation)
            target_observation_id = str(annotation.get("targetObservationId") or "").strip()
            target_snapshot = annotation.get("targetObservation")
            time_range = split_range(annotation)
            if not source_id:
                raise ReconstructionError("Choose the canonical identity before splitting it")
            if not target_observation_id or not isinstance(target_snapshot, dict):
                raise ReconstructionError(
                    "Split requires one immutable tracked observation; rebuild or select another frame"
                )
            if observation_identifier(target_snapshot) != target_observation_id:
                raise ReconstructionError("The split observation snapshot does not match its target")
            snapshot_identity = str(target_snapshot.get("canonicalPersonId") or "").strip()
            if snapshot_identity and snapshot_identity != source_id:
                raise ReconstructionError("The split observation belongs to another canonical identity")
            prior_target = split_target_ids.get(target_observation_id)
            if prior_target and prior_target != annotation_id:
                raise ReconstructionError("The same observation cannot anchor two split corrections")
            split_target_ids[target_observation_id] = annotation_id
            if target_snapshot.get("frameIndex") is None or not target_snapshot.get("bbox"):
                raise ReconstructionError("The split observation snapshot is incomplete")
            if time_range is None:
                raise ReconstructionError("Split range must have a valid start before its end")
            start, end = time_range
            if start < 0.0 or end > float(scene.get("duration") or 0.0) + 1e-6:
                raise ReconstructionError("Split range is outside this scene")
            target_time = float(target_snapshot.get("sceneTime") or 0.0)
            if not start <= target_time < end:
                raise ReconstructionError("The target observation must be inside the split range")
            split_identity_id = str(annotation.get("splitCanonicalPersonId") or "").strip()
            if not split_identity_id or split_identity_id == source_id:
                raise ReconstructionError("The split identity key is missing or invalid")
            prior_ranges = split_ranges.setdefault(source_id, [])
            if any(max(start, old_start) < min(end, old_end) - 1e-6 for old_start, old_end, _ in prior_ranges):
                raise ReconstructionError("Split ranges for the same identity cannot overlap")
            prior_ranges.append((start, end, annotation_id))
            source_key = canonical_correction_identity_key(
                scene, annotation_by_id, source_id
            )
            for roster_correction in dedicated_by_owner.get(source_key or "", []):
                if roster_correction.get("rosterBindingState") != "bound":
                    continue
                roster_snapshot = roster_correction.get("targetObservation")
                roster_time = (
                    roster_snapshot.get("sceneTime")
                    if isinstance(roster_snapshot, dict)
                    else roster_correction.get("sceneTime")
                )
                try:
                    roster_time = float(roster_time)
                except (TypeError, ValueError):
                    raise ReconstructionError(
                        "The bound roster correction has no usable split anchor; rebuild before splitting"
                    ) from None
                if not isfinite(roster_time):
                    raise ReconstructionError(
                        "The bound roster correction has no usable split anchor; rebuild before splitting"
                    )
                if start <= roster_time < end and not bound_roster_semantics_compatible(
                    str(annotation.get("kind") or ""),
                    {
                        annotation_id,
                        str(roster_correction.get("id") or ""),
                    },
                    annotation_by_id,
                ):
                    raise ReconstructionError(
                        "Unbind the roster player before splitting its anchored partition into another team or non-player role"
                    )
            continue
        if action != "merge":
            continue
        target_id = str(annotation.get("mergeTargetId") or "")
        if not target_id:
            raise ReconstructionError("Choose an existing track or labeled person to merge into")
        if target_id == annotation_id:
            raise ReconstructionError("A person cannot be merged into itself")
        if target_id == str(annotation_source_identity(annotation) or ""):
            raise ReconstructionError("The selected detection already belongs to that track")
        if target_id not in track_ids and target_id not in annotation_by_id:
            raise ReconstructionError("The merge target no longer exists")
        target_annotation = annotation_by_id.get(target_id)
        if target_annotation is not None and annotation_action(target_annotation) == "exclude":
            raise ReconstructionError("An excluded person cannot be an identity merge target")
        terminal_id = terminal_identity_target(annotation_id, annotation_by_id)
        if terminal_id in excluded_track_ids:
            raise ReconstructionError("An excluded track cannot be an identity merge target")
        source_subject = subjects_by_id.get(
            str(annotation_source_identity(annotation) or "")
        )
        terminal_annotation = annotation_by_id.get(terminal_id)
        target_subject = subjects_by_id.get(terminal_id) or subjects_by_id.get(
            str(annotation_source_identity(terminal_annotation or {}) or "")
        )
        source_key = canonical_correction_identity_key(
            scene,
            annotation_by_id,
            annotation_source_identity(annotation),
        )
        target_key = canonical_correction_identity_key(
            scene,
            annotation_by_id,
            (
                annotation_source_identity(terminal_annotation)
                if terminal_annotation is not None
                else terminal_id
            ),
        )
        source_roster_corrections = dedicated_by_owner.get(source_key or "", [])
        target_roster_corrections = dedicated_by_owner.get(target_key or "", [])
        if source_roster_corrections and target_roster_corrections:
            source_decisions = {
                roster_correction_decision(item)
                for item in source_roster_corrections
            }
            target_decisions = {
                roster_correction_decision(item)
                for item in target_roster_corrections
            }
            if source_decisions != target_decisions:
                raise ReconstructionError(
                    "Cannot merge identities with different dedicated Bind / Unbind decisions"
                )
            raise ReconstructionError(
                "Compatible dedicated roster corrections must be consolidated before merging identities"
            )
        source_external_ids = {
            str(value).strip()
            for value in ((source_subject or {}).get("externalPlayerId"),)
            if str(value or "").strip()
        }
        target_external_ids = {
            str(value).strip()
            for value in (
                (terminal_annotation or {}).get("externalPlayerId"),
                (target_subject or {}).get("externalPlayerId"),
            )
            if str(value or "").strip()
        }
        confirmed_external_ids = sorted(source_external_ids | target_external_ids)
        if len(confirmed_external_ids) > 1:
            raise ReconstructionError(
                "Cannot merge identities with different confirmed roster players: "
                + " and ".join(confirmed_external_ids)
            )
