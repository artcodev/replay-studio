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
# Above this implied speed two neighbouring observations cannot belong to
# one person even under projection noise: such a splice is an identity
# boundary. Between the two thresholds a splice is measurement noise — the
# fragments on both sides still belong to the same person and are retained
# as one chain instead of discarding everything but the longest fragment.
TRACK_IDENTITY_SPLIT_SPEED_METRES_PER_SECOND = 25.0


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
    adjusted_speeds: list[float] = []
    splice_speeds: list[float] = []
    splice_events: list[dict] = []
    for item in projected:
        segment = segments[-1]
        if segment:
            previous = segment[-1]
            elapsed = max(
                0.001,
                item.source_point["t"] - previous.source_point["t"],
            )
            distance = hypot(item.x - previous.x, item.z - previous.z)
            speed = distance / elapsed
            uncertainty_allowance = min(
                2.5,
                (
                    float(previous.uncertainty_metres or 0.0)
                    + float(item.uncertainty_metres or 0.0)
                )
                * 0.5,
            )
            adjusted_speed = max(0.0, distance - uncertainty_allowance) / elapsed
            raw_speeds.append(speed)
            adjusted_speeds.append(adjusted_speed)
            if adjusted_speed > MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND:
                splice_speeds.append(adjusted_speed)
                splice_events.append(
                    {
                        "leftTime": round(
                            float(previous.source_point["t"]),
                            3,
                        ),
                        "rightTime": round(
                            float(item.source_point["t"]),
                            3,
                        ),
                        "leftFrameIndex": previous.source_point.get(
                            "frameIndex"
                        ),
                        "rightFrameIndex": item.source_point.get(
                            "frameIndex"
                        ),
                        "rawDistanceMetres": round(distance, 3),
                        "uncertaintyAllowanceMetres": round(
                            uncertainty_allowance,
                            3,
                        ),
                        "rawSpeedMetresPerSecond": round(speed, 3),
                        "adjustedSpeedMetresPerSecond": round(
                            adjusted_speed,
                            3,
                        ),
                        "classification": (
                            "identity-boundary"
                            if adjusted_speed
                            > TRACK_IDENTITY_SPLIT_SPEED_METRES_PER_SECOND
                            else "measurement-noise"
                        ),
                    }
                )
                segments.append([])
                segment = segments[-1]
        segment.append(item)

    # Fragments joined by measurement-noise splices form one chain; only an
    # identity-grade splice breaks the chain. The longest chain is retained,
    # so a clearly visible early span is no longer sacrificed to one noisy
    # step, while observations beyond an identity switch stay quarantined.
    chains: list[list[list[ProjectedTrajectoryPoint]]] = [[segments[0]]]
    for boundary_speed, segment in zip(splice_speeds, segments[1:]):
        if boundary_speed > TRACK_IDENTITY_SPLIT_SPEED_METRES_PER_SECOND:
            chains.append([segment])
        else:
            chains[-1].append(segment)
    retained_chain_index = max(
        range(len(chains)),
        key=lambda index: sum(
            len(segment) for segment in chains[index]
        ),
    )
    retained_chain = chains[retained_chain_index]
    retained = [item for segment in retained_chain for item in segment]
    impossible_speed_count = sum(
        speed > MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND
        for speed in adjusted_speeds
    )
    discarded_ranges = [
        {
            "chainIndex": index,
            "startTime": round(
                float(chain[0][0].source_point["t"]),
                3,
            ),
            "endTime": round(
                float(chain[-1][-1].source_point["t"]),
                3,
            ),
            "startFrameIndex": chain[0][0].source_point.get("frameIndex"),
            "endFrameIndex": chain[-1][-1].source_point.get("frameIndex"),
            "observationCount": sum(len(segment) for segment in chain),
            "reason": "identity-grade-speed-boundary",
        }
        for index, chain in enumerate(chains)
        if index != retained_chain_index
    ]
    for event in splice_events:
        event["retainedAcrossBoundary"] = (
            event["classification"] == "measurement-noise"
        )
    discarded_observations = len(projected) - len(retained)
    quality = {
        "rawObservationCount": len(projected),
        "retainedObservationCount": len(retained),
        "discardedObservationCount": discarded_observations,
        "fragmentCount": len(segments),
        "retainedFragmentCount": len(retained_chain),
        "discardedFragmentCount": len(segments) - len(retained_chain),
        "softSpliceBridgedCount": len(retained_chain) - 1,
        "identitySpliceCount": sum(
            speed > TRACK_IDENTITY_SPLIT_SPEED_METRES_PER_SECOND
            for speed in splice_speeds
        ),
        "retentionPolicy": "soft-splice-chains-v1",
        "rawSpeedSampleCount": len(raw_speeds),
        "impossibleSpeedSegmentCount": impossible_speed_count,
        "rawImpossibleSpeedSegmentCount": sum(
            speed > MAXIMUM_PLAYER_SPEED_METRES_PER_SECOND
            for speed in raw_speeds
        ),
        "maximumRawSpeedMetresPerSecond": (
            round(max(raw_speeds), 3) if raw_speeds else None
        ),
        "maximumUncertaintyAdjustedSpeedMetresPerSecond": (
            round(max(adjusted_speeds), 3)
            if adjusted_speeds
            else None
        ),
        "retainedChainIndex": retained_chain_index,
        "spliceEvents": splice_events,
        "discardedRanges": discarded_ranges,
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
