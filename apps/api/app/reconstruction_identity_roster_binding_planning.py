from __future__ import annotations

"""Pure in-memory Set/Unbind planning for canonical roster decisions."""

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite
from typing import Mapping

from .reconstruction_errors import ReconstructionError
from .reconstruction_identity_contract import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    ROSTER_DECISION_ORIGIN_FIELD,
    ROSTER_IDENTITY_DEPENDENCIES_FIELD,
)
from .reconstruction_identity_match_roster import (
    match_snapshot_player,
    validate_canonical_roster_person,
    validate_canonical_roster_team,
)
from .reconstruction_identity_response import track_annotation_kind
from .reconstruction_identity_roster_baseline import (
    partition_local_identity_baseline,
)
from .reconstruction_identity_roster_corrections import (
    canonical_roster_binding_annotation_id,
    roster_decision_origin_id,
)
from .reconstruction_identity_roster_lineage import active_merge_dependencies
from .reconstruction_identity_roster_observations import (
    replace_roster_annotation_references,
    saved_detector_observation_for_binding,
)
from .reconstruction_identity_roster_ownership import (
    canonical_person_for_binding,
    ensure_external_player_is_available,
    roster_correction_index_for_set,
)
from .reconstruction_identity_semantics import annotation_action
from .reconstruction_identity_validation import validate_identity_corrections


def plan_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str | None,
    *,
    match_snapshot: Mapping[str, object] | None = None,
) -> dict:
    """Plan Set/Unbind against a hydrated scene without performing I/O."""
    canonical_person_id = str(canonical_person_id or "").strip()
    if not canonical_person_id:
        raise ReconstructionError("The canonical person no longer exists")
    normalized_external_id = (
        str(external_player_id).strip() if external_player_id is not None else None
    )
    if normalized_external_id == "":
        raise ReconstructionError("The external player id cannot be empty")

    person = canonical_person_for_binding(scene, canonical_person_id)
    if normalized_external_id is not None:
        validate_canonical_roster_person(person)

    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = deepcopy(list(reconstruction.get("frameAnnotations") or []))
    annotation_id = canonical_roster_binding_annotation_id(canonical_person_id)
    player: dict | None = None
    if normalized_external_id is not None:
        player = match_snapshot_player(match_snapshot, normalized_external_id)
        validate_canonical_roster_team(match_snapshot, person, player)
        ensure_external_player_is_available(
            scene,
            annotations,
            canonical_person_id,
            normalized_external_id,
        )

    existing_index = roster_correction_index_for_set(
        scene,
        annotations,
        canonical_person_id,
        annotation_id,
    )
    existing = annotations[existing_index] if existing_index is not None else None
    previous_annotation_id = str((existing or {}).get("id") or "") or None
    same_decision = (
        existing is not None
        and existing.get("externalPlayerId") == normalized_external_id
    )
    owner_changed = bool(
        existing is not None
        and str(existing.get("canonicalPersonId") or "") != canonical_person_id
    )
    existing_dependencies = (existing or {}).get(
        ROSTER_IDENTITY_DEPENDENCIES_FIELD
    ) or []
    if not isinstance(existing_dependencies, list):
        raise ReconstructionError(
            "The roster correction has invalid identity provenance"
        )
    identity_dependencies = {
        str(value).strip()
        for value in existing_dependencies
        if str(value or "").strip()
    }
    if existing is None or not same_decision or owner_changed:
        identity_dependencies.update(
            active_merge_dependencies(scene, person, annotations)
        )
    observation = saved_detector_observation_for_binding(
        person,
        existing,
        float(scene.get("duration") or 0.0),
        preserve_existing=same_decision,
    )
    observation["canonicalPersonId"] = canonical_person_id
    observation["annotationId"] = annotation_id

    baseline = _binding_baseline(
        person,
        annotations,
        existing,
        owner_changed=owner_changed,
        is_unbind=normalized_external_id is None,
    )
    annotation = _roster_binding_annotation(
        video=video,
        person=person,
        player=player,
        existing=existing,
        observation=observation,
        annotation_id=annotation_id,
        canonical_person_id=canonical_person_id,
        external_player_id=normalized_external_id,
        identity_dependencies=identity_dependencies,
        baseline=baseline,
        same_decision=same_decision,
    )
    if existing_index is None:
        annotations.append(annotation)
    else:
        annotations[existing_index] = annotation
    validate_identity_corrections(scene, annotations)
    reconstruction["frameAnnotations"] = sorted(
        annotations,
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or "")),
    )
    video["reconstruction"] = reconstruction
    if previous_annotation_id and previous_annotation_id != annotation_id:
        replace_roster_annotation_references(
            scene,
            canonical_person_id,
            previous_annotation_id,
            annotation_id,
        )

    _publish_binding_optimistically(person, scene, annotation, baseline)
    return annotation


def _binding_baseline(
    person: dict,
    annotations: list[dict],
    existing: dict | None,
    *,
    owner_changed: bool,
    is_unbind: bool,
) -> tuple[str, object, object, object]:
    base_display_name = str(
        (existing or {}).get("baseDisplayName")
        or person.get("displayName")
        or person.get("label")
        or person.get("canonicalPersonId")
        or person.get("id")
        or ""
    )
    local_annotation_ids = {str(value) for value in person.get("annotationIds") or []}
    has_non_roster_manual_semantics = any(
        str(item.get("id") or "") in local_annotation_ids
        and item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
        and annotation_action(item) in {"confirm", "merge", "split"}
        and item.get("kind") != "ignore"
        for item in annotations
    )
    if owner_changed or (is_unbind and has_non_roster_manual_semantics):
        return partition_local_identity_baseline(person, annotations)
    if existing is not None and "baseIdentityStatus" in existing:
        return (
            base_display_name,
            existing.get("baseIdentityStatus"),
            existing.get("baseIdentityConfidence"),
            existing.get("baseIdentitySource"),
        )
    return (
        base_display_name,
        person.get("identityStatus") or "provisional",
        person.get("identityConfidence"),
        person.get("identitySource") or "tracker+trajectory",
    )


def _roster_binding_annotation(
    *,
    video: dict,
    person: dict,
    player: dict | None,
    existing: dict | None,
    observation: dict,
    annotation_id: str,
    canonical_person_id: str,
    external_player_id: str | None,
    identity_dependencies: set[str],
    baseline: tuple[str, object, object, object],
    same_decision: bool,
) -> dict:
    bbox = observation["bbox"]
    scene_time = float(observation["sceneTime"])
    source_start = float(video.get("sourceStart") or 0.0)
    try:
        source_time = float(observation.get("sourceTime"))
    except (TypeError, ValueError):
        source_time = source_start + scene_time
    if not isfinite(source_time):
        source_time = source_start + scene_time

    base_display_name, base_status, base_confidence, base_source = baseline
    display_name = (
        str((player or {}).get("name") or external_player_id)
        if external_player_id is not None
        else base_display_name
    )
    return {
        "id": annotation_id,
        "sceneTime": round(scene_time, 3),
        "sourceTime": round(source_time, 3),
        "frameIndex": int(observation["frameIndex"]),
        "bbox": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "width": round(float(bbox["width"]), 2),
            "height": round(float(bbox["height"]), 2),
        },
        "kind": track_annotation_kind(person),
        "label": display_name,
        "externalPlayerId": external_player_id,
        "action": "confirm",
        "scope": "identity",
        "mergeTargetId": None,
        "sourceTrackId": person.get("renderTrackId"),
        "canonicalPersonId": canonical_person_id,
        "targetObservationId": observation["observationId"],
        "targetObservation": observation,
        "rangeStart": None,
        "rangeEnd": None,
        "splitCanonicalPersonId": None,
        "affectedPreview": None,
        "previewState": "confirmed" if external_player_id is not None else "unbound",
        "correctionKind": CANONICAL_ROSTER_BINDING_CORRECTION,
        "rosterBindingState": "bound" if external_player_id is not None else "unbound",
        ROSTER_DECISION_ORIGIN_FIELD: (
            roster_decision_origin_id(existing or {}) or annotation_id
        ),
        ROSTER_IDENTITY_DEPENDENCIES_FIELD: sorted(identity_dependencies),
        "baseDisplayName": base_display_name,
        "baseIdentityStatus": base_status,
        "baseIdentityConfidence": base_confidence,
        "baseIdentitySource": base_source,
        "updatedAt": (
            existing.get("updatedAt")
            if same_decision and existing is not None and existing.get("updatedAt")
            else datetime.now(UTC).isoformat()
        ),
    }


def _publish_binding_optimistically(
    person: dict,
    scene: dict,
    annotation: dict,
    baseline: tuple[str, object, object, object],
) -> None:
    annotation_id = str(annotation["id"])
    external_player_id = annotation.get("externalPlayerId")
    display_name = str(annotation["label"])
    _, base_status, base_confidence, base_source = baseline

    person["externalPlayerId"] = external_player_id
    person["displayName"] = display_name
    if external_player_id is not None:
        person["identityStatus"] = "resolved"
        person["identityConfidence"] = 1.0
        person["identitySource"] = "manual"
    else:
        person["identityStatus"] = base_status
        person["identityConfidence"] = base_confidence
        person["identitySource"] = base_source
    person["annotationIds"] = sorted(
        {*list(person.get("annotationIds") or []), annotation_id}
    )

    canonical_person_id = str(annotation["canonicalPersonId"])
    for track in scene.get("payload", {}).get("tracks") or []:
        if (
            str(track.get("canonicalPersonId") or "") == canonical_person_id
            or person.get("renderTrackId")
            and str(track.get("id") or "") == str(person.get("renderTrackId"))
        ):
            track["externalPlayerId"] = external_player_id
            track["label"] = display_name
            track["annotationIds"] = sorted(
                {*list(track.get("annotationIds") or []), annotation_id}
            )
