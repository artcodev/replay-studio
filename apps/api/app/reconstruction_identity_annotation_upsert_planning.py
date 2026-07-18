from __future__ import annotations

"""Pure frame-level identity annotation mutation planning."""

from copy import deepcopy
from hashlib import sha256

import numpy as np

from .reconstruction_frame_annotation_contract import FrameAnnotationTarget
from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    ROSTER_DECISION_ORIGIN_FIELD,
    ROSTER_IDENTITY_DEPENDENCIES_FIELD,
    SPLIT_ROSTER_UNDO_FIELD,
)
from .reconstruction_identity_correction_graph import (
    split_target_snapshot,
)
from .reconstruction_identity_read_model import canonical_analysis_subjects
from .reconstruction_identity_response import identity_target_defaults
from .reconstruction_identity_roster_corrections import (
    consolidate_compatible_merge_roster_corrections,
    dedicated_roster_corrections_by_owner,
    roster_decision_origin_id,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_role,
    annotation_team,
)
from .reconstruction_identity_reference_cleanup import remove_annotation_references
from .reconstruction_identity_validation import validate_identity_corrections


def plan_frame_person_annotation_upsert(
    scene: dict,
    values: dict,
    *,
    target: FrameAnnotationTarget,
    annotation_id: str,
    updated_at: str,
) -> dict:
    requested_annotation_id = str(values.get("annotation_id") or "").strip() or None
    if requested_annotation_id is not None:
        existing_annotation = next(
            (
                item
                for item in (
                    scene.get("payload", {})
                    .get("videoAsset", {})
                    .get("reconstruction", {})
                    .get("frameAnnotations", [])
                )
                if str(item.get("id") or "") == requested_annotation_id
            ),
            None,
        )
        if (
            existing_annotation is not None
            and existing_annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
        ):
            raise ReconstructionError(
                "Canonical roster corrections can only be changed through Bind / Unbind"
            )
    if requested_annotation_id is not None and requested_annotation_id != annotation_id:
        raise ReconstructionError("Annotation identity changed while planning the correction")
    action = annotation_action(values)
    if values.get("external_player_id") is not None:
        raise ReconstructionError(
            "Roster identity must be changed through the canonical Bind / Unbind endpoint"
        )
    if action != "exclude" and values.get("kind") == "ignore":
        raise ReconstructionError(
            "Choose a person role when confirming or merging an excluded detection"
        )
    explicit_action = values.get("action") is not None
    requested_scope = str(values.get("scope") or "").strip().lower()
    scope = (
        "range"
        if action == "split"
        else requested_scope
        if explicit_action and requested_scope in {"observation", "identity"}
        else "identity"
        if explicit_action
        else "observation"
    )
    merge_target_id = (
        str(values.get("merge_target_id") or "").strip() or None
        if action == "merge"
        else None
    )
    source_track_id = str(values.get("source_track_id") or "").strip() or None
    canonical_person_id = (
        str(values.get("canonical_person_id") or "").strip() or None
    )
    canonical_subject: dict | None = None
    if canonical_person_id is not None:
        canonical_subject = next(
            (
                subject
                for subject in canonical_analysis_subjects(scene)
                if str(subject.get("canonicalPersonId") or "")
                == canonical_person_id
            ),
            None,
        )
        if canonical_subject is None:
            raise ReconstructionError("The canonical person no longer exists")
    if (
        action == "confirm"
        and canonical_subject is not None
        and canonical_subject.get("externalPlayerId")
    ):
        expected_team = str(canonical_subject.get("teamId") or "").strip()
        requested_team = annotation_team(str(values.get("kind") or ""))
        requested_role = annotation_role(str(values.get("kind") or ""))
        if (
            requested_team != expected_team
            or requested_role in {"referee", "other", None}
        ):
            raise ReconstructionError(
                "Unbind the roster player before changing this person to another team or non-player role"
            )
    target_observation_id = (
        str(values.get("target_observation_id") or "").strip() or None
        if action == "split"
        else None
    )
    target_observation: dict | None = None
    split_canonical_person_id: str | None = None
    range_start: float | None = None
    range_end: float | None = None
    affected_preview: dict | None = None
    if action == "split":
        if canonical_person_id is None:
            raise ReconstructionError("Choose the canonical identity before splitting it")
        if target_observation_id is None:
            raise ReconstructionError(
                "Split requires one immutable tracked observation; rebuild or select another frame"
            )
        subject, target_observation = split_target_snapshot(
            scene,
            canonical_person_id,
            target_observation_id,
        )
        if int(target_observation["frameIndex"]) != target.frame_index:
            raise ReconstructionError("The split target does not belong to the selected frame")
        range_start = float(
            values.get("range_start")
            if values.get("range_start") is not None
            else target_observation["sceneTime"]
        )
        range_end = float(
            values.get("range_end")
            if values.get("range_end") is not None
            else scene.get("duration")
        )
        if (
            not np.isfinite([range_start, range_end]).all()
            or range_start < 0.0
            or range_end > float(scene.get("duration") or 0.0) + 1e-6
            or range_end <= range_start
        ):
            raise ReconstructionError("Split range must be inside the scene and have a valid end")
        target_time = float(target_observation["sceneTime"])
        if not range_start <= target_time < range_end:
            raise ReconstructionError("The target observation must be inside the split range")
        subject_observations = [
            observation
            for observation in subject.get("observations") or []
            if observation.get("sceneTime") is not None
        ]
        affected_count = sum(
            range_start <= float(observation["sceneTime"]) < range_end
            for observation in subject_observations
        )
        remaining_count = len(subject_observations) - affected_count
        if affected_count <= 0 or remaining_count <= 0:
            raise ReconstructionError(
                "Split must leave at least one detector observation on both identities"
            )
        split_seed = f"{annotation_id}:{target_observation_id}"
        split_canonical_person_id = (
            f"canonical-split-{sha256(split_seed.encode('utf-8')).hexdigest()[:12]}"
        )
        affected_preview = {
            "canonicalPersonId": canonical_person_id,
            "splitCanonicalPersonId": split_canonical_person_id,
            "rangeStart": round(range_start, 3),
            "rangeEnd": round(range_end, 3),
            "affectedObservationCount": affected_count,
            "remainingObservationCount": remaining_count,
        }
    if (
        action == "exclude"
        and scope == "identity"
        and canonical_person_id is None
        and source_track_id is None
    ):
        raise ReconstructionError(
            "Choose the tracked identity before excluding the whole trajectory"
        )
    annotation = {
        "id": annotation_id,
        "sceneTime": round(target.scene_time, 3),
        "sourceTime": round(
            float(scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0)
            + target.scene_time,
            3,
        ),
        "frameIndex": target.frame_index,
        "bbox": {
            "x": round(target.x, 2),
            "y": round(target.y, 2),
            "width": round(target.width, 2),
            "height": round(target.height, 2),
        },
        "kind": "ignore" if action == "exclude" else values["kind"],
        "label": (
            None
            if action in {"exclude", "split"}
            else (values.get("label") or "").strip() or None
        ),
        "externalPlayerId": None,
        "action": action,
        "scope": scope,
        "mergeTargetId": merge_target_id,
        "sourceTrackId": source_track_id,
        "canonicalPersonId": canonical_person_id,
        "targetObservationId": target_observation_id,
        "targetObservation": target_observation,
        "rangeStart": round(range_start, 3) if range_start is not None else None,
        "rangeEnd": round(range_end, 3) if range_end is not None else None,
        "splitCanonicalPersonId": split_canonical_person_id,
        "affectedPreview": affected_preview,
        "previewState": {
            "confirm": "confirmed",
            "exclude": "excluded",
            "merge": "merged",
            "split": "split",
        }[action],
        "updatedAt": updated_at,
    }
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    if action == "split":
        # Snapshot only decisions that existed before this split.  Undo can
        # then restore their original owner/id instead of silently deleting a
        # durable Unbind after the correction has moved to the range child.
        pre_split_unbinds = [
            deepcopy(item)
            for item in dedicated_roster_corrections_by_owner(
                scene, annotations
            ).get(str(canonical_person_id or ""), [])
            if item.get("rosterBindingState") == "unbound"
        ]
        for item in pre_split_unbinds:
            item[ROSTER_DECISION_ORIGIN_FIELD] = roster_decision_origin_id(item)
        annotation[SPLIT_ROSTER_UNDO_FIELD] = pre_split_unbinds
    existing_index = next(
        (index for index, item in enumerate(annotations) if item.get("id") == annotation_id),
        None,
    )
    if existing_index is None:
        annotations.append(annotation)
    else:
        annotations[existing_index] = annotation
    consolidated_roster_corrections: list[dict] = []
    if action == "merge":
        annotations, consolidated_roster_corrections = (
            consolidate_compatible_merge_roster_corrections(
                scene, annotations, annotation
            )
        )
        if consolidated_roster_corrections:
            consolidated_roster_ids = sorted(
                str(item["id"])
                for item in consolidated_roster_corrections
                if item.get("id")
            )
            annotation["consolidatedRosterCorrectionIds"] = consolidated_roster_ids
            annotation["consolidatedRosterCorrections"] = deepcopy(
                consolidated_roster_corrections
            )
    consolidated_roster_ids = {
        str(item["id"])
        for item in consolidated_roster_corrections
        if item.get("id")
    }
    validate_identity_corrections(scene, annotations)
    if action == "merge" and merge_target_id is not None:
        kind, label, external_player_id = identity_target_defaults(
            scene, annotations, merge_target_id
        )
        annotation.update(
            {
                "kind": kind,
                "label": label,
                # A merge points at the live target identity. Persisting its
                # roster ID here creates a stale independent claim after a
                # later Bind/Unbind/Rebind.
                "externalPlayerId": None,
            }
        )
    reconstruction["frameAnnotations"] = sorted(
        annotations,
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or "")),
    )
    video["reconstruction"] = reconstruction
    remove_annotation_references(scene, consolidated_roster_ids)
    return annotation
