from __future__ import annotations

"""Project, filter, and materialize one accepted player trajectory."""

from dataclasses import dataclass
from math import hypot

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_calibration_policy import METRIC_CALIBRATION_THRESHOLD
from .reconstruction_track_state import TrackState
from .reconstruction_latent_presence import materialize_continuous_presence
from .reconstruction_pitch_projection import project_pitch_point


MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND = 14.0


def _smooth_axis(values: list[float]) -> list[float]:
    if len(values) < 3:
        return values
    return [
        values[index]
        if index in {0, len(values) - 1}
        else (values[index - 1] + values[index] * 2 + values[index + 1]) / 4
        for index in range(len(values))
    ]


@dataclass(frozen=True)
class ProjectedTrajectoryPoint:
    source_point: dict
    x: float
    z: float
    projection_source: str
    calibration_frame_index: int | None
    uncertainty_metres: float | None


@dataclass(frozen=True)
class TrackTrajectory:
    projected_by_source: dict[int, ProjectedTrajectoryPoint]
    retained_by_source: dict[int, ProjectedTrajectoryPoint]
    observed_keyframes: list[dict]
    quality: dict
    maximum_raw_speed: float | None

    @property
    def retained_count(self) -> int:
        return len(self.observed_keyframes)


def resolve_track_coordinate_mode(
    calibration: PitchCalibration | None,
    requested_mode: str | None,
) -> str:
    return requested_mode or (
        "metric"
        if calibration is not None
        and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
        else "approximate"
    )


def project_track_trajectory(
    track: TrackState,
    frame_size: tuple[int, int],
    pitch: dict,
    coordinate_mode: str,
) -> TrackTrajectory | None:
    width, height = frame_size
    projected: list[ProjectedTrajectoryPoint] = []
    for point in track.points:
        if coordinate_mode == "metric":
            if point.get("pitchX") is None or point.get("pitchZ") is None:
                # Missing metric observations stay missing. Screen coordinates
                # must never masquerade as metric evidence.
                continue
            position = (float(point["pitchX"]), float(point["pitchZ"]))
            source = str(point.get("projectionSource") or "direct")
            calibration_frame_index = point.get("calibrationFrameIndex")
            uncertainty = point.get("positionUncertaintyMetres")
        else:
            position = project_pitch_point(
                point["px"],
                point["py"],
                width,
                height,
                pitch,
                None,
            )
            source = "screen-approximate"
            calibration_frame_index = None
            uncertainty = 12.0
        projected.append(
            ProjectedTrajectoryPoint(
                source_point=point,
                x=position[0],
                z=position[1],
                projection_source=source,
                calibration_frame_index=calibration_frame_index,
                uncertainty_metres=uncertainty,
            )
        )
    if not projected:
        return None

    segments: list[list[ProjectedTrajectoryPoint]] = [[]]
    raw_speeds: list[float] = []
    for item in projected:
        segment = segments[-1]
        if segment:
            previous = segment[-1]
            elapsed = max(
                0.001,
                item.source_point["t"] - previous.source_point["t"],
            )
            speed = hypot(item.x - previous.x, item.z - previous.z) / elapsed
            raw_speeds.append(speed)
            if speed > MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND:
                segments.append([])
                segment = segments[-1]
        segment.append(item)

    retained = max(segments, key=len)
    non_empty_segments = [segment for segment in segments if segment]
    impossible_speed_count = sum(
        speed > MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND for speed in raw_speeds
    )
    discarded_observations = len(projected) - len(retained)
    quality = {
        "rawObservationCount": len(projected),
        "retainedObservationCount": len(retained),
        "discardedObservationCount": discarded_observations,
        "fragmentCount": len(non_empty_segments),
        "discardedFragmentCount": max(0, len(non_empty_segments) - 1),
        "rawSpeedSampleCount": len(raw_speeds),
        "impossibleSpeedSegmentCount": impossible_speed_count,
        "maximumRawSpeedMetresPerSecond": (
            round(max(raw_speeds), 3) if raw_speeds else None
        ),
    }

    xs = _smooth_axis([item.x for item in retained])
    zs = _smooth_axis([item.z for item in retained])
    retained_smoothed = [
        ProjectedTrajectoryPoint(
            source_point=item.source_point,
            x=x,
            z=z,
            projection_source=item.projection_source,
            calibration_frame_index=item.calibration_frame_index,
            uncertainty_metres=item.uncertainty_metres,
        )
        for item, x, z in zip(retained, xs, zs)
    ]
    observed_keyframes = [
        {
            "t": round(item.source_point["t"], 3),
            "x": round(item.x, 2),
            "z": round(item.z, 2),
            "confidence": round(
                0.35 + min(1.0, item.source_point["confidence"]) * 0.62,
                3,
            ),
            "observed": True,
            "presenceState": "observed",
            "projectionSource": item.projection_source,
            "calibrationFrameIndex": item.calibration_frame_index,
            "positionUncertaintyMetres": item.uncertainty_metres,
            "projection": {
                "source": item.projection_source,
                "calibrationFrameIndex": item.calibration_frame_index,
                "uncertaintyMetres": item.uncertainty_metres,
            },
        }
        for item in retained_smoothed
    ]
    return TrackTrajectory(
        projected_by_source={id(item.source_point): item for item in projected},
        retained_by_source={
            id(item.source_point): item for item in retained_smoothed
        },
        observed_keyframes=observed_keyframes,
        quality=quality,
        maximum_raw_speed=max(raw_speeds) if raw_speeds else None,
    )


def eligible_track_candidates(
    tracks: list[TrackState],
    mapping: dict[int, str],
    minimum_observations: int,
) -> list[TrackState]:
    accepted = [
        track
        for track in tracks
        if track.identity_status != "excluded"
        and track.id in mapping
        and (
            len(track.points) >= minimum_observations
            or track.positive_annotation_ids
        )
    ]
    accepted.sort(
        key=lambda track: (
            mapping[track.id],
            0 if track.positive_annotation_ids else 1,
            0 if track.role == "goalkeeper" else 1,
            -len(track.points),
        )
    )
    return accepted


def trajectory_passes_acceptance(
    track: TrackState,
    trajectory: TrackTrajectory,
    *,
    minimum_observations: int,
    published_team_count: int,
    maximum_team_count: int,
) -> bool:
    if track.positive_annotation_ids:
        return True
    return (
        trajectory.retained_count >= minimum_observations
        and published_team_count < maximum_team_count
    )


def materialize_continuous_trajectory(
    trajectory: TrackTrajectory,
    duration: float,
    pitch: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    """Fill latent presence without converting inferred points to observations."""

    return materialize_continuous_presence(
        trajectory.observed_keyframes,
        duration,
        pitch,
        seed,
    )


__all__ = [
    "ProjectedTrajectoryPoint",
    "TrackTrajectory",
    "eligible_track_candidates",
    "materialize_continuous_trajectory",
    "project_track_trajectory",
    "resolve_track_coordinate_mode",
    "trajectory_passes_acceptance",
]
