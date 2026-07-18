from __future__ import annotations

"""Short-horizon association of sampled person detections into local tracks."""

from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from .reconstruction_person_detection_contract import Detection
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_team
from .track_observation_accumulator import append_track_observation


NEW_TRACK_CONFIDENCE = 0.12


def _predicted_track_point(track: TrackState, time: float) -> tuple[float, float]:
    last = track.points[-1]
    predicted_x, predicted_y = float(last["px"]), float(last["py"])
    if len(track.points) < 2:
        return predicted_x, predicted_y
    previous = track.points[-2]
    sample_elapsed = float(last["t"]) - float(previous["t"])
    prediction_elapsed = time - float(last["t"])
    if sample_elapsed <= 1e-4 or prediction_elapsed <= 0.0:
        return predicted_x, predicted_y
    # Do not extrapolate a noisy two-point velocity indefinitely through an
    # occlusion. The association gate still grows with elapsed time.
    horizon = min(prediction_elapsed, 0.35)
    predicted_x += (float(last["px"]) - float(previous["px"])) / sample_elapsed * horizon
    predicted_y += (float(last["py"]) - float(previous["py"])) / sample_elapsed * horizon
    return predicted_x, predicted_y


def _association_cost(track: TrackState, detection: Detection, time: float) -> float:
    last = track.points[-1]
    elapsed = time - float(last["t"])
    if elapsed <= 0.0 or elapsed > 0.65:
        return float("inf")

    track_team = annotation_team(track.manual_kind)
    detection_team = annotation_team(detection.annotation_kind)
    if track_team and detection_team and track_team != detection_team:
        return float("inf")
    if (
        track.manual_identity_owner_ids
        and detection.manual_identity_owner_ids
        and track.manual_identity_owner_ids.isdisjoint(
            detection.manual_identity_owner_ids
        )
    ):
        return float("inf")
    if (
        track.roster_binding_state is not None
        and detection.roster_binding_state is not None
        and (
            track.roster_binding_state != detection.roster_binding_state
            or track.manual_external_player_id != detection.external_player_id
        )
    ):
        return float("inf")
    if (
        track.manual_external_player_id
        and detection.external_player_id
        and track.manual_external_player_id != detection.external_player_id
        and (
            track.roster_binding_state is None
            and detection.roster_binding_state is None
            or track.roster_binding_state is not None
            and detection.roster_binding_state is not None
        )
    ):
        return float("inf")

    predicted_x, predicted_y = _predicted_track_point(track, time)
    pixel_distance = hypot(detection.x - predicted_x, detection.y - predicted_y)
    elapsed_scale = 1.0 + min(2.2, max(0.0, elapsed / 0.1 - 1.0) * 0.45)
    pixel_gate = max(48.0, detection.height * 2.4, track.last_height * 2.4) * elapsed_scale
    if pixel_distance > pixel_gate:
        return float("inf")
    pixel_cost = pixel_distance / max(1.0, pixel_gate)

    appearance_distance = float(np.linalg.norm(detection.feature - track.feature))
    appearance_cost = min(2.0, appearance_distance / 0.9)
    reid_cost: float | None = None
    track_reid = track.reid_feature
    if detection.reid_feature is not None and track_reid is not None:
        detection_reid = np.asarray(detection.reid_feature, dtype=np.float32)
        detection_norm = float(np.linalg.norm(detection_reid))
        if detection_norm > 1e-8 and np.isfinite(detection_reid).all():
            detection_reid = detection_reid / detection_norm
            cosine_distance = max(
                0.0,
                min(2.0, 1.0 - float(np.dot(track_reid, detection_reid))),
            )
            # This gate only affects the short-horizon tracker. Long-gap
            # identity decisions are made later by the audited resolver.
            if cosine_distance > 0.72 and track.reid_feature_count >= 2:
                return float("inf")
            reid_cost = min(2.0, cosine_distance / 0.38)
    height_ratio = max(detection.height, track.last_height) / max(
        1.0, min(detection.height, track.last_height)
    )
    size_cost = min(1.0, abs(float(np.log(height_ratio))) / 0.7)

    pitch_cost: float | None = None
    if (
        last.get("pitchX") is not None
        and last.get("pitchZ") is not None
        and detection.pitch_x is not None
        and detection.pitch_z is not None
    ):
        predicted_pitch_x = float(last["pitchX"])
        predicted_pitch_z = float(last["pitchZ"])
        if len(track.points) > 1:
            previous = track.points[-2]
            pitch_elapsed = float(last["t"]) - float(previous["t"])
            if (
                pitch_elapsed > 1e-4
                and previous.get("pitchX") is not None
                and previous.get("pitchZ") is not None
            ):
                horizon = min(elapsed, 0.35)
                predicted_pitch_x += (
                    float(last["pitchX"]) - float(previous["pitchX"])
                ) / pitch_elapsed * horizon
                predicted_pitch_z += (
                    float(last["pitchZ"]) - float(previous["pitchZ"])
                ) / pitch_elapsed * horizon
        pitch_distance = hypot(
            float(detection.pitch_x) - predicted_pitch_x,
            float(detection.pitch_z) - predicted_pitch_z,
        )
        uncertainty = float(detection.position_uncertainty_metres or 0.0) + float(
            last.get("positionUncertaintyMetres") or 0.0
        )
        pitch_gate = 2.2 + 16.0 * elapsed + min(5.0, uncertainty)
        if pitch_distance > pitch_gate:
            return float("inf")
        pitch_cost = pitch_distance / max(0.5, pitch_gate)

    if pitch_cost is None and reid_cost is None:
        cost = pixel_cost * 0.58 + appearance_cost * 0.34 + size_cost * 0.08
    elif pitch_cost is None:
        cost = (
            pixel_cost * 0.51
            + appearance_cost * 0.18
            + float(reid_cost) * 0.25
            + size_cost * 0.06
        )
    elif reid_cost is None:
        cost = (
            pixel_cost * 0.30
            + appearance_cost * 0.24
            + pitch_cost * 0.40
            + size_cost * 0.06
        )
    else:
        cost = (
            pixel_cost * 0.25
            + appearance_cost * 0.12
            + float(reid_cost) * 0.20
            + pitch_cost * 0.38
            + size_cost * 0.05
        )
    if detection.annotation_id and detection.annotation_id in track.annotation_ids:
        cost *= 0.2
    elif detection.external_player_id and (
        detection.external_player_id == track.manual_external_player_id
    ):
        cost *= 0.35
    return float(cost)


def track_people(frames: list[tuple[list[Detection], float]]) -> list[TrackState]:
    tracks: list[TrackState] = []
    next_id = 1
    for frame_index, (detections, time) in enumerate(frames):
        active = [
            track
            for track in tracks
            if track.points and time - float(track.points[-1]["t"]) <= 0.65
        ]
        assigned_track_ids: set[int] = set()
        assigned_detections: set[int] = set()
        primary_detection_indices = [
            index
            for index, detection in enumerate(detections)
            if detection.confidence >= NEW_TRACK_CONFIDENCE or detection.annotation_id
        ]
        secondary_detection_indices = [
            index
            for index in range(len(detections))
            if index not in primary_detection_indices
        ]

        def assign(
            candidate_tracks: list[TrackState],
            detection_indices: list[int],
            maximum_cost: float,
        ) -> None:
            if not candidate_tracks or not detection_indices:
                return
            costs = np.full(
                (len(candidate_tracks), len(detection_indices)),
                np.inf,
                dtype=np.float64,
            )
            for track_index, track in enumerate(candidate_tracks):
                for column, detection_index in enumerate(detection_indices):
                    costs[track_index, column] = _association_cost(
                        track, detections[detection_index], time
                    )
            finite = np.isfinite(costs)
            assignment_costs = np.where(finite, costs, 1e6)
            rows, columns = linear_sum_assignment(assignment_costs)
            for track_index, column in zip(rows.tolist(), columns.tolist()):
                cost = float(costs[track_index, column])
                if not np.isfinite(cost) or cost > maximum_cost:
                    continue
                detection_index = detection_indices[column]
                alternatives = [
                    float(value)
                    for index, value in enumerate(costs[track_index])
                    if index != column and np.isfinite(value)
                ]
                margin = max(0.0, min(alternatives) - cost) if alternatives else None
                detection = detections[detection_index]
                detection.association_cost = cost
                detection.association_margin = margin
                track = candidate_tracks[track_index]
                append_track_observation(track, detection, frame_index, time)
                assigned_track_ids.add(track.id)
                assigned_detections.add(detection_index)

        # ByteTrack-style two-stage association: reliable observations claim
        # identities first; low-confidence detections may continue an existing
        # track but never create a new ghost track by themselves.
        assign(active, primary_detection_indices, 1.05)
        remaining_tracks = [
            track for track in active if track.id not in assigned_track_ids
        ]
        assign(remaining_tracks, secondary_detection_indices, 0.92)

        for detection_index in primary_detection_indices:
            detection = detections[detection_index]
            if detection_index in assigned_detections:
                continue
            track = TrackState(id=next_id)
            append_track_observation(track, detection, frame_index, time)
            tracks.append(track)
            next_id += 1
    return tracks


__all__ = ["track_people"]
