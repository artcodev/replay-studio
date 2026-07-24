from __future__ import annotations

"""Validated mutation boundary for appending a detection to a person track."""

from .reconstruction_errors import ReconstructionError
from .reconstruction_person_detection_contract import Detection
from .reconstruction_track_state import TrackState
from .track_reid_accumulator import accumulate_track_reid_observation


def _validate_manual_identity_observation(
    track: TrackState,
    detection: Detection,
) -> None:
    """Reject manual-identity conflicts before mutating any track evidence."""

    combined_manual_owners = (
        set(track.manual_identity_owner_ids) | set(detection.manual_identity_owner_ids)
    )
    if len(combined_manual_owners) > 1:
        raise ReconstructionError(
            "Conflicting explicit canonical identities reached one raw track"
        )
    if not detection.annotation_id:
        return
    if detection.roster_binding_state in {"bound", "unbound"}:
        if (
            track.roster_binding_state is not None
            and (
                track.roster_binding_state != detection.roster_binding_state
                or track.manual_external_player_id != detection.external_player_id
            )
        ):
            raise ReconstructionError(
                "Conflicting dedicated roster corrections reached one raw track"
            )
    elif detection.external_player_id is not None:
        raise ReconstructionError(
            "Roster identity requires a dedicated Bind / Unbind correction"
        )


def append_track_observation(
    track: TrackState,
    detection: Detection,
    frame_index: int,
    time: float,
) -> None:
    _validate_manual_identity_observation(track, detection)

    track.source_tracklet_ids.add(track.local_tracklet_id)
    image_x = detection.image_x if detection.image_x is not None else detection.x
    image_y = detection.image_y if detection.image_y is not None else detection.y
    source_frame_index = (
        detection.source_frame_index
        if detection.source_frame_index is not None
        else frame_index
    )
    observation_id = detection.observation_id or (
        f"{track.local_tracklet_id}:{source_frame_index}"
    )
    point = {
        "t": time,
        "px": detection.x,
        "py": detection.y,
        "confidence": detection.confidence,
        "frameIndex": source_frame_index,
        "observationId": observation_id,
        "sourceTrackletId": track.local_tracklet_id,
        "bbox": {
            "x": image_x - detection.width / 2,
            "y": image_y - detection.height,
            "width": detection.width,
            "height": detection.height,
        },
        "annotationId": detection.annotation_id,
        **(
            {"annotationIds": sorted(detection.annotation_ids)}
            if detection.annotation_ids
            else {}
        ),
        "_appearanceFeature": detection.feature.copy(),
        **(
            {"annotationIsIdentityEvidence": detection.annotation_is_identity_evidence}
            if detection.annotation_id
            else {}
        ),
    }
    if detection.pitch_x is not None and detection.pitch_z is not None:
        point["pitchX"] = detection.pitch_x
        point["pitchZ"] = detection.pitch_z
        point["projectionSource"] = detection.projection_source or "direct"
        point["calibrationFrameIndex"] = detection.calibration_frame_index
        point["positionUncertaintyMetres"] = detection.position_uncertainty_metres
    if detection.association_cost is not None:
        point["associationCost"] = round(detection.association_cost, 4)
        point["associationMargin"] = (
            round(detection.association_margin, 4)
            if detection.association_margin is not None
            else None
        )
    if detection.association_diagnostics is not None:
        point["associationDiagnostics"] = dict(
            detection.association_diagnostics
        )
    if detection.tracking_decision is not None:
        point["trackingDecision"] = detection.tracking_decision
    track.points.append(point)
    if track.feature_sum is None:
        track.feature_sum = detection.feature.copy()
    else:
        track.feature_sum += detection.feature
    track.feature_count += 1

    accumulate_track_reid_observation(
        track,
        detection,
        point,
        observation_id=observation_id,
        frame_index=frame_index,
        time=time,
    )
    track.last_frame = frame_index
    track.last_height = detection.height
    if not detection.annotation_id:
        return

    detection_annotation_ids = set(detection.annotation_ids) | {detection.annotation_id}
    track.annotation_ids.update(detection_annotation_ids)
    tombstone_ids = set(detection.identity_tombstone_annotation_ids)
    if not detection.annotation_is_identity_evidence:
        tombstone_ids.add(detection.annotation_id)
    track.identity_tombstone_ids.update(tombstone_ids)
    track.identity_tombstone_ids.intersection_update(track.annotation_ids)
    track.manual_identity_owner_ids.update(detection.manual_identity_owner_ids)
    if (
        detection.manual_semantic_key is not None
        and (
            track.manual_semantic_key is None
            or detection.manual_semantic_key >= track.manual_semantic_key
        )
    ):
        track.manual_kind = detection.annotation_kind
        track.manual_label = detection.annotation_label
        track.manual_semantic_key = detection.manual_semantic_key

    if detection.roster_binding_state in {"bound", "unbound"}:
        incoming_external_id = detection.external_player_id
        track.roster_binding_state = detection.roster_binding_state
        track.roster_binding_annotation_ids.update(
            detection.roster_binding_annotation_ids or {detection.annotation_id}
        )
        track.manual_external_player_id = incoming_external_id


__all__ = ["append_track_observation"]
