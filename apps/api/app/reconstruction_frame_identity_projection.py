from __future__ import annotations

"""Project current-frame detections onto persisted canonical identities."""

from copy import deepcopy
from dataclasses import dataclass

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_calibration_policy import METRIC_CALIBRATION_THRESHOLD
from .reconstruction_person_detection_contract import Detection
from .reconstruction_identity_correction_graph import terminal_identity_target
from .reconstruction_identity_read_model import (
    canonical_analysis_subjects,
    interpolate_scene_keyframes,
)
from .reconstruction_identity_response import identity_annotation_response
from .reconstruction_identity_review import (
    frame_track_observations,
    pair_detections_to_stored_observations,
    raw_person_bbox,
    track_observation_schema_version,
)
from .reconstruction_identity_semantics import (
    annotation_action,
    annotation_scope,
    annotation_source_identity,
    annotation_team,
    identity_annotations,
    observation_identifier,
)
from .reconstruction_team_classification import cluster_color


@dataclass(frozen=True)
class FrameIdentityProjection:
    detections: list[dict]
    annotations: list[dict]
    observation_schema_version: int | None

    @property
    def has_observation_schema(self) -> bool:
        return self.observation_schema_version is not None


def _manually_forced_track(
    correction: dict | None,
    *,
    subjects: list[dict],
    annotations_by_id: dict[str, dict],
) -> dict | None:
    if correction is None:
        return None
    action = annotation_action(correction)
    requested_id: str | None = None
    if action == "merge" and correction.get("mergeTargetId"):
        requested_id = terminal_identity_target(
            str(correction["mergeTargetId"]), annotations_by_id
        )
    elif (
        action == "confirm"
        and annotation_scope(correction) == "identity"
        and annotation_source_identity(correction)
    ):
        requested_id = str(annotation_source_identity(correction))
    if not requested_id:
        return None
    return next(
        (
            track
            for track in subjects
            if str(track.get("id") or "") == requested_id
            or str(track.get("canonicalPersonId") or "") == requested_id
            or requested_id in (track.get("annotationIds") or [])
        ),
        None,
    )


def _correction_fields(correction: dict | None) -> dict:
    action = annotation_action(correction) if correction else None
    return {
        "correctionAction": action,
        "correctionScope": annotation_scope(correction) if correction else None,
        "mergeTargetId": (
            correction.get("mergeTargetId") if action == "merge" else None
        ),
        "sourceTrackId": correction.get("sourceTrackId") if correction else None,
        "targetObservationId": (
            correction.get("targetObservationId") if correction else None
        ),
        "rangeStart": correction.get("rangeStart") if correction else None,
        "rangeEnd": correction.get("rangeEnd") if correction else None,
        "splitCanonicalPersonId": (
            correction.get("splitCanonicalPersonId") if correction else None
        ),
        "affectedPreview": (
            deepcopy(correction.get("affectedPreview")) if correction else None
        ),
        "previewState": (
            correction.get("previewState") if correction else "uncorrected"
        ),
    }


def _stored_observation_document(
    *,
    track: dict,
    observation: dict,
    item: Detection | None,
    correction: dict | None,
    annotation_id: str | None,
    forced_match: dict | None,
    frame_index: int,
    frame_time: float,
) -> dict:
    matched = forced_match or track
    observation_pitch = observation.get("pitch")
    metric_status = str(
        observation.get("metricStatus")
        or ("accepted" if observation_pitch else "unprojected")
    )
    metric_reason = observation.get("metricReason")
    if metric_status == "accepted" and not observation_pitch:
        metric_status = "unprojected"
        metric_reason = "metric-projection-unavailable"
    position = observation_pitch if metric_status == "accepted" else None
    position_source = "observation" if position else "track-inferred"
    if position is None:
        position = interpolate_scene_keyframes(
            track.get("keyframes") or [], frame_time
        )
    if position is None:
        # No accepted metric, no published keyframes: inventing a pitch
        # point (the old {0,0} placeholder put these people at the centre
        # circle) is worse than an explicit missing position.
        position_source = "unprojectable"
    annotation_kind = (
        item.annotation_kind
        if item is not None and item.annotation_kind
        else correction.get("kind")
        if correction
        else None
    )
    annotation_label = (
        item.annotation_label
        if item is not None and item.annotation_label
        else correction.get("label")
        if correction
        else None
    )
    return {
        "id": str(
            observation.get("observationId")
            or observation.get("id")
            or f"observation-{track.get('canonicalPersonId') or track.get('id')}-{frame_index}"
        ),
        "observationId": observation.get("observationId")
        or observation.get("id"),
        "confidence": round(float(observation.get("confidence") or 0.0), 3),
        "bbox": deepcopy(observation["bbox"]),
        "pitch": (
            {
                "x": round(float(position["x"]), 2),
                "z": round(float(position["z"]), 2),
            }
            if position is not None
            else None
        ),
        "jerseyColor": (
            cluster_color(item.feature)
            if item is not None
            else str(matched.get("color") or "#d7dce8")
        ),
        "annotationId": annotation_id,
        "kind": annotation_kind,
        "annotationLabel": annotation_label,
        "source": "manual" if annotation_id else "automatic",
        "matchedTrackId": matched.get("renderTrackId") or matched.get("id"),
        "matchedTrackLabel": annotation_label or matched.get("label"),
        "canonicalPersonId": matched.get("canonicalPersonId"),
        "identityStatus": matched.get("identityStatus"),
        "identityConfidence": matched.get("identityConfidence"),
        "identitySource": matched.get("identitySource"),
        "displayName": matched.get("displayName") or matched.get("label"),
        "jerseyNumber": matched.get("jerseyNumber"),
        "teamId": annotation_team(annotation_kind) or matched.get("teamId"),
        "matchDistance": None,
        "matchSource": (
            "manual-identity" if forced_match else "persisted-observation"
        ),
        "metricStatus": metric_status,
        "metricReason": metric_reason,
        "rawPitch": deepcopy(observation.get("rawPitch")),
        "projectionSource": observation.get("projectionSource"),
        "positionUncertaintyMetres": observation.get(
            "positionUncertaintyMetres"
        ),
        "positionSource": position_source,
        **_correction_fields(correction),
    }


def _fresh_detection_document(
    *,
    index: int,
    item: Detection,
    bbox: dict,
    position: tuple[float, float] | None,
    raw_position: tuple[float, float] | None,
    correction: dict | None,
    forced_match: dict | None,
    calibration: PitchCalibration | None,
) -> dict:
    metric_accepted = bool(
        position is not None
        and calibration is not None
        and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
    )
    matched = forced_match
    return {
        "id": f"person-{index + 1}",
        "observationId": None,
        "confidence": round(float(item.confidence), 3),
        "bbox": bbox,
        "pitch": (
            {"x": round(position[0], 2), "z": round(position[1], 2)}
            if position is not None
            else None
        ),
        "jerseyColor": cluster_color(item.feature),
        "annotationId": item.annotation_id,
        "kind": item.annotation_kind,
        "annotationLabel": item.annotation_label,
        "source": "manual" if item.annotation_id else "automatic",
        "matchedTrackId": matched.get("id") if matched else None,
        "matchedTrackLabel": item.annotation_label
        or (matched.get("label") if matched else None),
        "canonicalPersonId": matched.get("canonicalPersonId") if matched else None,
        "identityStatus": matched.get("identityStatus") if matched else None,
        "identityConfidence": (
            matched.get("identityConfidence") if matched else None
        ),
        "identitySource": matched.get("identitySource") if matched else None,
        "displayName": matched.get("displayName") if matched else None,
        "jerseyNumber": matched.get("jerseyNumber") if matched else None,
        "teamId": annotation_team(item.annotation_kind)
        or (matched.get("teamId") if matched else None),
        "matchDistance": None,
        "matchSource": "manual-identity" if matched else None,
        "metricStatus": "accepted" if metric_accepted else "unprojected",
        "metricReason": (
            None
            if metric_accepted
            else "metric-projection-outside-pitch"
            if raw_position is not None and calibration is not None
            else "metric-projection-unavailable"
        ),
        "rawPitch": (
            {"x": round(raw_position[0], 2), "z": round(raw_position[1], 2)}
            if raw_position is not None and not metric_accepted
            else None
        ),
        "projectionSource": (
            calibration.method
            if metric_accepted and calibration is not None
            else None
        ),
        "positionUncertaintyMetres": None,
        "positionSource": "observation",
        **_correction_fields(correction),
    }


def project_frame_people(
    scene: dict,
    *,
    people: list[Detection],
    projected_people: list[tuple[float, float] | None],
    raw_projected_people: list[tuple[float, float] | None],
    frame_index: int,
    frame_time: float,
    calibration: PitchCalibration | None,
) -> FrameIdentityProjection:
    annotations = [
        identity_annotation_response(annotation)
        for annotation in frame_annotations(scene, frame_index)
    ]
    annotations_by_id = {str(item["id"]): item for item in annotations}
    all_annotations = {
        str(item["id"]): item
        for item in identity_annotations(scene)
        if item.get("id")
    }
    subjects = canonical_analysis_subjects(scene)
    schema_version = track_observation_schema_version(scene)
    detection_boxes = [
        raw_person_bbox(
            {"x": item.x, "y": item.y, "width": item.width, "height": item.height}
        )
        for item in people
    ]
    stored = frame_track_observations(scene, frame_index) if schema_version else []
    pairs, consumed = (
        pair_detections_to_stored_observations(detection_boxes, stored)
        if stored
        else ({}, set())
    )
    detections: list[dict] = []
    for observation_index, (track, observation) in enumerate(stored):
        detection_index = pairs.get(observation_index)
        item = people[detection_index] if detection_index is not None else None
        annotation_id = (
            item.annotation_id
            if item is not None and item.annotation_id
            else observation.get("annotationId")
        )
        correction = annotations_by_id.get(str(annotation_id or ""))
        if correction is None:
            observation_id = observation_identifier(observation)
            split_matches = [
                candidate
                for candidate in annotations
                if candidate.get("action") == "split"
                and candidate.get("targetObservationId") == observation_id
            ]
            if len(split_matches) == 1:
                correction = split_matches[0]
                annotation_id = correction["id"]
        forced = _manually_forced_track(
            correction,
            subjects=subjects,
            annotations_by_id=all_annotations,
        )
        detections.append(
            _stored_observation_document(
                track=track,
                observation=observation,
                item=item,
                correction=correction,
                annotation_id=annotation_id,
                forced_match=forced,
                frame_index=frame_index,
                frame_time=frame_time,
            )
        )
    raw_positions = raw_projected_people or [None] * len(projected_people)
    for index, (item, position, raw_position) in enumerate(
        zip(people, projected_people, raw_positions)
    ):
        if index in consumed:
            continue
        correction = annotations_by_id.get(str(item.annotation_id or ""))
        forced = _manually_forced_track(
            correction,
            subjects=subjects,
            annotations_by_id=all_annotations,
        )
        detections.append(
            _fresh_detection_document(
                index=index,
                item=item,
                bbox=detection_boxes[index],
                position=position,
                raw_position=raw_position,
                correction=correction,
                forced_match=forced,
                calibration=calibration,
            )
        )
    return FrameIdentityProjection(detections, annotations, schema_version)


# Imported late to keep the projection module's dependency direction explicit:
# annotation retrieval is input, while all identity decisions remain pure above.
from .reconstruction_person_annotations import frame_annotations
