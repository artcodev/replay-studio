from __future__ import annotations

"""Materialize authoritative video observations alongside trajectory QA."""

from copy import deepcopy
from dataclasses import dataclass

from .reconstruction_track_state import TrackState
from .reconstruction_track_trajectory import TrackTrajectory


def _observation_priority(observation: dict) -> tuple[int, float]:
    return (
        1 if observation.get("annotationId") else 0,
        float(observation.get("confidence") or 0.0),
    )


def merge_track_observations(*collections: list[dict]) -> list[dict]:
    """Return at most one authoritative video observation per source frame."""

    by_frame: dict[int, dict] = {}
    for observation in (item for collection in collections for item in collection):
        if observation.get("frameIndex") is None or not observation.get("bbox"):
            continue
        frame_index = int(observation["frameIndex"])
        previous = by_frame.get(frame_index)
        if previous is None or _observation_priority(observation) > _observation_priority(
            previous
        ):
            by_frame[frame_index] = deepcopy(observation)
    return [by_frame[frame_index] for frame_index in sorted(by_frame)]


@dataclass(frozen=True)
class TrackObservationMaterialization:
    observations: list[dict]
    quality: dict
    image_observation_count: int
    accepted_count: int
    rejected_count: int
    unprojected_count: int


def materialize_track_observations(
    track: TrackState,
    trajectory: TrackTrajectory,
    source_start: float,
) -> TrackObservationMaterialization:
    rows: list[dict] = []
    for point in track.points:
        if point.get("frameIndex") is None or not point.get("bbox"):
            continue
        observation = {
            "frameIndex": int(point["frameIndex"]),
            "sceneTime": round(float(point["t"]), 3),
            "bbox": {
                "x": round(float(point["bbox"]["x"]), 2),
                "y": round(float(point["bbox"]["y"]), 2),
                "width": round(float(point["bbox"]["width"]), 2),
                "height": round(float(point["bbox"]["height"]), 2),
            },
            "confidence": round(float(point.get("confidence") or 0.0), 3),
            "annotationId": point.get("annotationId"),
        }
        if track.canonical_person_id:
            observation.update(
                {
                    "id": point.get("observationId"),
                    "observationId": point.get("observationId"),
                    "sourceFrameIndex": int(point["frameIndex"]),
                    "sourceTime": round(source_start + float(point["t"]), 3),
                    "sourceTrackletId": point.get("sourceTrackletId")
                    or track.local_tracklet_id,
                    "canonicalPersonId": track.canonical_person_id,
                }
            )

        retained = trajectory.retained_by_source.get(id(point))
        projected = trajectory.projected_by_source.get(id(point))
        if retained is not None:
            source = retained.projection_source
            calibration_frame_index = retained.calibration_frame_index
            uncertainty = retained.uncertainty_metres
            observation.update(
                {
                    "metricStatus": "accepted",
                    "metricReason": None,
                    "pitch": {
                        "x": round(float(retained.x), 2),
                        "z": round(float(retained.z), 2),
                    },
                }
            )
        elif projected is not None:
            source = projected.projection_source
            calibration_frame_index = projected.calibration_frame_index
            uncertainty = projected.uncertainty_metres
            observation.update(
                {
                    "metricStatus": "rejected",
                    "metricReason": "trajectory-fragment-rejected",
                    "rawPitch": {
                        "x": round(float(projected.x), 2),
                        "z": round(float(projected.z), 2),
                    },
                }
            )
        else:
            source = point.get("projectionSource")
            calibration_frame_index = point.get("calibrationFrameIndex")
            uncertainty = point.get("positionUncertaintyMetres")
            observation.update(
                {
                    "metricStatus": "unprojected",
                    "metricReason": "metric-projection-unavailable",
                }
            )
        if source:
            observation["projectionSource"] = str(source)
        if calibration_frame_index is not None:
            observation["calibrationFrameIndex"] = int(calibration_frame_index)
        if uncertainty is not None:
            observation["positionUncertaintyMetres"] = round(
                float(uncertainty),
                3,
            )
        rows.append(observation)

    observations = merge_track_observations(rows)
    status_counts = {
        status: sum(item.get("metricStatus") == status for item in observations)
        for status in ("accepted", "rejected", "unprojected")
    }
    image_observation_count = sum(
        point.get("frameIndex") is not None and bool(point.get("bbox"))
        for point in track.points
    )
    quality = {
        "imageObservationCount": image_observation_count,
        "publishedIdentityObservationCount": len(observations),
        "metricAcceptedObservationCount": status_counts["accepted"],
        "metricRejectedObservationCount": status_counts["rejected"],
        "metricUnprojectedObservationCount": status_counts["unprojected"],
        "identityObservationCoverage": round(
            len(observations) / max(1, image_observation_count),
            3,
        ),
        "metricObservationCoverage": round(
            status_counts["accepted"] / max(1, len(observations)),
            3,
        ),
    }
    return TrackObservationMaterialization(
        observations=observations,
        quality=quality,
        image_observation_count=image_observation_count,
        accepted_count=status_counts["accepted"],
        rejected_count=status_counts["rejected"],
        unprojected_count=status_counts["unprojected"],
    )


__all__ = [
    "TrackObservationMaterialization",
    "materialize_track_observations",
    "merge_track_observations",
]
