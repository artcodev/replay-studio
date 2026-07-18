"""Apply persisted manual person corrections to one frame's detector evidence."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .person_appearance import appearance_feature
from .reconstruction_errors import ReconstructionError
from .reconstruction_person_detection_contract import Detection
from .reconstruction_identity_contract import CANONICAL_ROSTER_BINDING_CORRECTION
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_manual_semantic_key,
    annotation_scope,
    annotation_source_identity,
    is_identity_unbind_tombstone,
)
from .bounding_box_geometry import intersection_over_union


def _annotation_box(annotation: dict) -> tuple[float, float, float, float]:
    bbox = annotation["bbox"]
    return (
        float(bbox["x"]),
        float(bbox["y"]),
        float(bbox["x"]) + float(bbox["width"]),
        float(bbox["y"]) + float(bbox["height"]),
    )


def _detection_box(detection: Detection) -> tuple[float, float, float, float]:
    return (
        detection.x - detection.width / 2,
        detection.y - detection.height,
        detection.x + detection.width / 2,
        detection.y,
    )


def frame_annotations(scene: dict, frame_index: int) -> list[dict]:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    annotations = deepcopy(list(reconstruction.get("frameAnnotations") or []))
    split_owner_aliases = {
        str(annotation.get("splitCanonicalPersonId")): str(
            annotation_source_identity(annotation) or ""
        )
        for annotation in annotations
        if annotation_action(annotation) == "split"
        and annotation.get("splitCanonicalPersonId")
        and annotation_source_identity(annotation)
    }

    def pre_split_owner(owner: str) -> str:
        visited: set[str] = set()
        current = owner
        while current in split_owner_aliases and current not in visited:
            visited.add(current)
            current = split_owner_aliases[current]
        return current

    for annotation in annotations:
        owner = str(annotation.get("canonicalPersonId") or "").strip()
        if owner in split_owner_aliases:
            annotation["preSplitCanonicalOwnerId"] = pre_split_owner(owner)
    dedicated_owner_keys = {
        str(value)
        for annotation in annotations
        if annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION
        and annotation.get("rosterBindingState") in {"bound", "unbound"}
        for value in (
            annotation.get("canonicalPersonId"),
            annotation.get("sourceTrackId"),
        )
        if str(value or "").strip()
    }
    result: list[dict] = []
    for annotation in annotations:
        if (
            annotation.get("frameIndex") is None
            or int(annotation["frameIndex"]) != frame_index
        ):
            continue
        owner_keys = {
            str(value)
            for value in (
                annotation.get("canonicalPersonId"),
                annotation.get("sourceTrackId"),
            )
            if str(value or "").strip()
        }
        if (
            annotation.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("externalPlayerId") is not None
            and (
                annotation_action(annotation) == "merge"
                or bool(owner_keys & dedicated_owner_keys)
            )
        ):
            result.append(
                {
                    **annotation,
                    "externalPlayerId": None,
                    "rosterValueSupersededByDedicatedCorrection": True,
                }
            )
        else:
            result.append(annotation)
    return result


def _annotation_detection_index(
    detections: list[Detection],
    annotation: dict,
) -> int | None:
    target = _annotation_box(annotation)
    candidates: list[tuple[float, int]] = []
    for index, detection in enumerate(detections):
        box = _detection_box(detection)
        overlap = intersection_over_union(target, box)
        center_inside = (
            target[0] <= detection.x <= target[2]
            and target[1] <= detection.y <= target[3]
        )
        if overlap >= 0.12 or center_inside:
            candidates.append((overlap + (0.25 if center_inside else 0.0), index))
    return max(candidates)[1] if candidates else None


def apply_person_annotations(
    image: np.ndarray,
    detections: list[Detection],
    annotations: list[dict],
) -> list[Detection]:
    result = list(detections)
    for annotation in annotations:
        # A split edits the identity graph; it never synthesizes detector evidence.
        if annotation_action(annotation) == "split":
            continue
        detection_index = _annotation_detection_index(result, annotation)
        if annotation_action(annotation) == "exclude":
            if annotation_scope(annotation) == "identity":
                if detection_index is not None:
                    detection = result[detection_index]
                    detection.annotation_id = annotation["id"]
                    detection.annotation_kind = "ignore"
                    detection.annotation_label = None
                    detection.external_player_id = None
                continue
            if detection_index is not None:
                result.pop(detection_index)
            continue
        if detection_index is None:
            x1, y1, x2, y2 = _annotation_box(annotation)
            detection = Detection(
                x=(x1 + x2) / 2,
                y=y2,
                width=x2 - x1,
                height=y2 - y1,
                confidence=1.0,
                feature=appearance_feature(image, (x1, y1, x2, y2)),
            )
            result.append(detection)
        else:
            detection = result[detection_index]
            detection.confidence = max(detection.confidence, 0.98)
        annotation_id = str(annotation["id"])
        detection.annotation_ids.add(annotation_id)
        if (
            annotation_action(annotation) == "confirm"
            and annotation_scope(annotation) == "identity"
        ):
            manual_owner_id = str(
                annotation.get("preSplitCanonicalOwnerId")
                or annotation.get("canonicalPersonId")
                or annotation.get("sourceTrackId")
                or ""
            ).strip()
            if manual_owner_id:
                detection.manual_identity_owner_ids.add(manual_owner_id)
                if len(detection.manual_identity_owner_ids) > 1:
                    raise ReconstructionError(
                        "Conflicting explicit canonical identities target one observation"
                    )
        is_unbind_tombstone = is_identity_unbind_tombstone(annotation)
        is_roster_binding = (
            annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("rosterBindingState") in {"bound", "unbound"}
        )
        if is_roster_binding:
            incoming_state = str(annotation["rosterBindingState"])
            incoming_external_id = annotation.get("externalPlayerId")
            if (
                detection.roster_binding_state is not None
                and (
                    detection.roster_binding_state != incoming_state
                    or detection.external_player_id != incoming_external_id
                )
            ):
                raise ReconstructionError(
                    "Conflicting dedicated roster corrections target one observation"
                )
            detection.roster_binding_state = incoming_state
            detection.roster_binding_annotation_ids.add(annotation_id)
        if is_unbind_tombstone:
            detection.identity_tombstone_annotation_ids.add(annotation_id)
        semantic_key = annotation_manual_semantic_key(annotation)
        if (
            detection.manual_semantic_key is None
            or semantic_key >= detection.manual_semantic_key
        ):
            detection.annotation_id = annotation["id"]
            detection.annotation_kind = annotation["kind"]
            detection.annotation_label = annotation.get("label")
            detection.annotation_is_identity_evidence = not is_unbind_tombstone
            detection.manual_semantic_key = semantic_key
        incoming_external_id = annotation.get("externalPlayerId")
        if is_roster_binding:
            detection.external_player_id = incoming_external_id
        elif incoming_external_id is not None:
            raise ReconstructionError(
                "Roster identity requires a dedicated Bind / Unbind correction"
            )
    return result
