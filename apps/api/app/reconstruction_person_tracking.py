from __future__ import annotations

"""Short-horizon association of sampled person detections into local tracks."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from .reconstruction_person_detection_contract import Detection
from .reconstruction_coordinate_policy import (
    EXPLICIT_IMAGE_FALLBACK,
    METRIC_REQUIRED,
    TRACKING_COORDINATE_POLICIES,
)
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_team
from .track_observation_accumulator import append_track_observation


NEW_TRACK_CONFIDENCE = 0.12
PRIMARY_MAXIMUM_COST = 0.80
SECONDARY_MAXIMUM_COST = 0.72
AMBIGUOUS_COST_FLOOR = 0.72
MINIMUM_ASSOCIATION_MARGIN = 0.015
ASSOCIATION_MODEL = "metric-first-ambiguity-guarded-v2"


@dataclass(frozen=True)
class AssociationDecision:
    cost: float
    rejection_reason: str | None
    metrics: dict

    @property
    def accepted_by_gates(self) -> bool:
        return np.isfinite(self.cost)

    def payload(self) -> dict:
        return {
            "model": ASSOCIATION_MODEL,
            "cost": round(float(self.cost), 4)
            if np.isfinite(self.cost)
            else None,
            "rejectionReason": self.rejection_reason,
            **self.metrics,
        }


def _rejected(reason: str, **metrics) -> AssociationDecision:
    return AssociationDecision(float("inf"), reason, metrics)


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


def _association_decision(
    track: TrackState,
    detection: Detection,
    time: float,
    image_fallback_authorized: bool,
) -> AssociationDecision:
    last = track.points[-1]
    elapsed = time - float(last["t"])
    if elapsed <= 0.0:
        return _rejected("non-positive-time-gap", elapsedSeconds=round(elapsed, 4))
    if elapsed > 0.65:
        return _rejected("time-gap-too-large", elapsedSeconds=round(elapsed, 4))

    track_team = annotation_team(track.manual_kind)
    detection_team = annotation_team(detection.annotation_kind)
    if track_team and detection_team and track_team != detection_team:
        return _rejected(
            "manual-team-conflict",
            elapsedSeconds=round(elapsed, 4),
        )
    if (
        track.manual_identity_owner_ids
        and detection.manual_identity_owner_ids
        and track.manual_identity_owner_ids.isdisjoint(
            detection.manual_identity_owner_ids
        )
    ):
        return _rejected("manual-identity-conflict")
    if (
        track.roster_binding_state is not None
        and detection.roster_binding_state is not None
        and (
            track.roster_binding_state != detection.roster_binding_state
            or track.manual_external_player_id != detection.external_player_id
        )
    ):
        return _rejected("roster-binding-conflict")
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
        return _rejected("external-player-conflict")

    metric_available = (
        last.get("pitchX") is not None
        and last.get("pitchZ") is not None
        and detection.pitch_x is not None
        and detection.pitch_z is not None
    )
    if not metric_available and not image_fallback_authorized:
        return _rejected(
            "metric-position-unavailable",
            elapsedSeconds=round(elapsed, 4),
        )

    predicted_x, predicted_y = _predicted_track_point(track, time)
    pixel_distance = hypot(detection.x - predicted_x, detection.y - predicted_y)
    elapsed_scale = 1.0 + min(2.2, max(0.0, elapsed / 0.1 - 1.0) * 0.45)
    pixel_gate = max(48.0, detection.height * 2.4, track.last_height * 2.4) * elapsed_scale
    # A calibrated metric observation must never be vetoed solely by screen
    # motion. Camera motion, zoom and direction changes are already represented
    # in the per-frame homography. Pixel distance is a soft tie-breaker there;
    # it remains a hard gate only for explicitly authorized image fallback.
    if not metric_available and pixel_distance > pixel_gate:
        return _rejected(
            "image-fallback-pixel-gate",
            elapsedSeconds=round(elapsed, 4),
            pixelDistance=round(pixel_distance, 3),
            pixelGate=round(pixel_gate, 3),
        )
    pixel_cost = min(2.0, pixel_distance / max(1.0, pixel_gate))

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
                return _rejected(
                    "reid-distance-gate",
                    elapsedSeconds=round(elapsed, 4),
                    reidCosineDistance=round(cosine_distance, 4),
                    reidGate=0.72,
                )
            reid_cost = min(2.0, cosine_distance / 0.38)
    height_ratio = max(detection.height, track.last_height) / max(
        1.0, min(detection.height, track.last_height)
    )
    size_cost = min(1.0, abs(float(np.log(height_ratio))) / 0.7)

    pitch_cost: float | None = None
    if metric_available:
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
        predicted_pitch_distance = hypot(
            float(detection.pitch_x) - predicted_pitch_x,
            float(detection.pitch_z) - predicted_pitch_z,
        )
        last_pitch_distance = hypot(
            float(detection.pitch_x) - float(last["pitchX"]),
            float(detection.pitch_z) - float(last["pitchZ"]),
        )
        # Constant velocity is useful while a player runs straight, but a hard
        # extrapolated gate loses a clearly visible player on an abrupt turn.
        # Accept the better of constant-velocity and last-position hypotheses;
        # the common physical gate still rejects impossible displacement.
        pitch_distance = min(predicted_pitch_distance, last_pitch_distance)
        pitch_hypothesis = (
            "constant-velocity"
            if predicted_pitch_distance <= last_pitch_distance
            else "last-position-direction-change"
        )
        uncertainty = float(detection.position_uncertainty_metres or 0.0) + float(
            last.get("positionUncertaintyMetres") or 0.0
        )
        pitch_gate = 2.2 + 16.0 * elapsed + min(5.0, uncertainty)
        if pitch_distance > pitch_gate:
            return _rejected(
                "metric-distance-gate",
                elapsedSeconds=round(elapsed, 4),
                pitchDistanceMetres=round(pitch_distance, 4),
                predictedPitchDistanceMetres=round(
                    predicted_pitch_distance,
                    4,
                ),
                lastPitchDistanceMetres=round(last_pitch_distance, 4),
                pitchGateMetres=round(pitch_gate, 4),
                pitchHypothesis=pitch_hypothesis,
                pixelDistance=round(pixel_distance, 3),
                pixelGate=round(pixel_gate, 3),
            )
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
            pixel_cost * 0.10
            + appearance_cost * 0.18
            + pitch_cost * 0.62
            + size_cost * 0.10
        )
    else:
        cost = (
            pixel_cost * 0.08
            + appearance_cost * 0.08
            + float(reid_cost) * 0.22
            + pitch_cost * 0.57
            + size_cost * 0.05
        )
    if detection.annotation_id and detection.annotation_id in track.annotation_ids:
        cost *= 0.2
    elif detection.external_player_id and (
        detection.external_player_id == track.manual_external_player_id
    ):
        cost *= 0.35
    return AssociationDecision(
        float(cost),
        None,
        {
            "coordinateMode": (
                "metric" if metric_available else "explicit-image-fallback"
            ),
            "elapsedSeconds": round(elapsed, 4),
            "pixelDistance": round(pixel_distance, 3),
            "pixelGate": round(pixel_gate, 3),
            "pixelCost": round(pixel_cost, 4),
            "appearanceCost": round(appearance_cost, 4),
            "reidCost": (
                round(float(reid_cost), 4) if reid_cost is not None else None
            ),
            "pitchCost": (
                round(float(pitch_cost), 4) if pitch_cost is not None else None
            ),
            "pitchDistanceMetres": (
                round(float(pitch_distance), 4)
                if pitch_cost is not None
                else None
            ),
            "pitchGateMetres": (
                round(float(pitch_gate), 4)
                if pitch_cost is not None
                else None
            ),
            "pitchHypothesis": (
                pitch_hypothesis if pitch_cost is not None else None
            ),
            "sizeCost": round(size_cost, 4),
        },
    )


def _association_cost(
    track: TrackState,
    detection: Detection,
    time: float,
    image_fallback_authorized: bool,
) -> float:
    """Compatibility wrapper for numerical callers and focused tests."""

    return _association_decision(
        track,
        detection,
        time,
        image_fallback_authorized,
    ).cost


def track_people(
    frames: list[tuple[list[Detection], float]],
    *,
    coordinate_policy: str = METRIC_REQUIRED,
    image_fallback_sample_indices: Sequence[int] = (),
    diagnostics: dict | None = None,
) -> list[TrackState]:
    if coordinate_policy not in TRACKING_COORDINATE_POLICIES:
        raise ValueError("Unknown tracking coordinate policy")
    tracks: list[TrackState] = []
    fallback_samples = {int(value) for value in image_fallback_sample_indices}
    track_last_sample_index: dict[int, int] = {}
    frame_decisions: list[dict] = []
    outcome_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    next_id = 1
    for frame_index, (detections, time) in enumerate(frames):
        active = [
            track
            for track in tracks
            if track.points and time - float(track.points[-1]["t"]) <= 0.65
        ]
        assigned_track_ids: set[int] = set()
        assigned_detections: set[int] = set()
        accepted_track_by_detection: dict[int, int] = {}
        pair_decisions: dict[tuple[int, int], AssociationDecision] = {}
        assignment_rejections: dict[tuple[int, int], str] = {}
        image_fallback_authorized = (
            coordinate_policy == EXPLICIT_IMAGE_FALLBACK
            and frame_index in fallback_samples
        )
        primary_detection_indices = [
            index
            for index, detection in enumerate(detections)
            if (detection.confidence >= NEW_TRACK_CONFIDENCE or detection.annotation_id)
            and (
                image_fallback_authorized
                or detection.pitch_x is not None
                and detection.pitch_z is not None
            )
        ]
        secondary_detection_indices = [
            index
            for index in range(len(detections))
            if index not in primary_detection_indices
            and (
                image_fallback_authorized
                or detections[index].pitch_x is not None
                and detections[index].pitch_z is not None
            )
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
                    decision = _association_decision(
                        track,
                        detections[detection_index],
                        time,
                        image_fallback_authorized=(
                            image_fallback_authorized
                            or coordinate_policy == EXPLICIT_IMAGE_FALLBACK
                            and track_last_sample_index.get(track.id)
                            in fallback_samples
                        ),
                    )
                    pair_decisions[(track.id, detection_index)] = decision
                    costs[track_index, column] = decision.cost
            finite = np.isfinite(costs)
            assignment_costs = np.where(finite, costs, 1e6)
            rows, columns = linear_sum_assignment(assignment_costs)
            for track_index, column in zip(rows.tolist(), columns.tolist()):
                cost = float(costs[track_index, column])
                track = candidate_tracks[track_index]
                detection_index = detection_indices[column]
                if not np.isfinite(cost):
                    continue
                if cost > maximum_cost:
                    assignment_rejections[(track.id, detection_index)] = (
                        "cost-above-assignment-threshold"
                    )
                    continue
                row_alternatives = [
                    float(value)
                    for index, value in enumerate(costs[track_index])
                    if index != column and np.isfinite(value)
                ]
                column_alternatives = [
                    float(value)
                    for index, value in enumerate(costs[:, column])
                    if index != track_index and np.isfinite(value)
                ]
                alternatives = [*row_alternatives, *column_alternatives]
                margin = (
                    max(0.0, min(alternatives) - cost)
                    if alternatives
                    else None
                )
                # In a crowded frame, a high-cost Hungarian result with no
                # separation from the next candidate is an ID-switch risk, not
                # positive identity evidence. Starting a short new tracklet is
                # safer because the audited canonical resolver can merge it
                # later using accumulated ReID evidence.
                if (
                    cost >= AMBIGUOUS_COST_FLOOR
                    and margin is not None
                    and margin < MINIMUM_ASSOCIATION_MARGIN
                ):
                    assignment_rejections[(track.id, detection_index)] = (
                        "association-margin-too-small"
                    )
                    continue
                detection = detections[detection_index]
                detection.association_cost = cost
                detection.association_margin = margin
                decision = pair_decisions[(track.id, detection_index)]
                detection.association_diagnostics = decision.payload()
                detection.tracking_decision = "matched-existing-track"
                append_track_observation(track, detection, frame_index, time)
                track_last_sample_index[track.id] = frame_index
                assigned_track_ids.add(track.id)
                assigned_detections.add(detection_index)
                accepted_track_by_detection[detection_index] = track.id

        # ByteTrack-style two-stage association: reliable observations claim
        # identities first; low-confidence detections may continue an existing
        # track but never create a new false-positive track by themselves.
        assign(active, primary_detection_indices, PRIMARY_MAXIMUM_COST)
        remaining_tracks = [
            track for track in active if track.id not in assigned_track_ids
        ]
        assign(
            remaining_tracks,
            secondary_detection_indices,
            SECONDARY_MAXIMUM_COST,
        )

        for detection_index in primary_detection_indices:
            detection = detections[detection_index]
            if detection_index in assigned_detections:
                continue
            track = TrackState(id=next_id)
            detection.tracking_decision = "created-new-track"
            append_track_observation(track, detection, frame_index, time)
            track_last_sample_index[track.id] = frame_index
            tracks.append(track)
            accepted_track_by_detection[detection_index] = track.id
            assigned_detections.add(detection_index)
            next_id += 1
        frame_rows: list[dict] = []
        for detection_index, detection in enumerate(detections):
            candidate_rows = [
                {
                    "trackId": track_id,
                    **decision.payload(),
                    **(
                        {
                            "assignmentRejectionReason": assignment_rejections[
                                (track_id, detection_index)
                            ]
                        }
                        if (track_id, detection_index) in assignment_rejections
                        else {}
                    ),
                }
                for (track_id, candidate_index), decision in pair_decisions.items()
                if candidate_index == detection_index
            ]
            candidate_rows.sort(
                key=lambda item: (
                    item.get("cost") is None,
                    (
                        float(item["cost"])
                        if item.get("cost") is not None
                        else 1e9
                    ),
                    str(item.get("rejectionReason") or ""),
                    int(item["trackId"]),
                )
            )
            if detection_index in accepted_track_by_detection:
                status = str(
                    detection.tracking_decision or "matched-existing-track"
                )
                reason = None
                track_id = accepted_track_by_detection[detection_index]
            elif (
                (
                    detection.pitch_x is None
                    or detection.pitch_z is None
                )
                and not image_fallback_authorized
            ):
                status = "untracked"
                reason = detection.metric_projection_reason or (
                    "metric-position-unavailable"
                )
                track_id = None
            elif detection.confidence < NEW_TRACK_CONFIDENCE and not detection.annotation_id:
                status = "untracked"
                reason = "confidence-below-new-track-threshold"
                track_id = None
            else:
                status = "untracked"
                reason = (
                    candidate_rows[0].get("rejectionReason")
                    if candidate_rows
                    else "no-active-association-candidate"
                )
                track_id = None
            outcome_counts[status] += 1
            if reason:
                rejection_counts[str(reason)] += 1
            frame_rows.append(
                {
                    "detectionIndex": detection_index,
                    "observationId": detection.observation_id,
                    "confidence": round(float(detection.confidence), 4),
                    "metricAvailable": (
                        detection.pitch_x is not None
                        and detection.pitch_z is not None
                    ),
                    "rawPitch": (
                        {
                            "x": round(float(detection.raw_pitch_x), 3),
                            "z": round(float(detection.raw_pitch_z), 3),
                        }
                        if detection.raw_pitch_x is not None
                        and detection.raw_pitch_z is not None
                        else None
                    ),
                    "status": status,
                    "reason": reason,
                    "trackId": track_id,
                    "candidates": candidate_rows[:3],
                }
            )
        frame_decisions.append(
            {
                "sampleIndex": frame_index,
                "sceneTime": round(float(time), 4),
                "imageFallbackAuthorized": image_fallback_authorized,
                "detections": frame_rows,
            }
        )
    if diagnostics is not None:
        diagnostics.update(
            {
                "schemaVersion": 1,
                "model": ASSOCIATION_MODEL,
                "coordinatePolicy": coordinate_policy,
                "primaryMaximumCost": PRIMARY_MAXIMUM_COST,
                "secondaryMaximumCost": SECONDARY_MAXIMUM_COST,
                "ambiguousCostFloor": AMBIGUOUS_COST_FLOOR,
                "minimumAssociationMargin": MINIMUM_ASSOCIATION_MARGIN,
                "frameCount": len(frames),
                "detectionCount": sum(
                    len(items) for items, _ in frames
                ),
                "trackletCount": len(tracks),
                "outcomeCounts": dict(sorted(outcome_counts.items())),
                "rejectionReasonCounts": dict(
                    sorted(rejection_counts.items())
                ),
                "frames": frame_decisions,
            }
        )
    return tracks


__all__ = ["track_people"]
