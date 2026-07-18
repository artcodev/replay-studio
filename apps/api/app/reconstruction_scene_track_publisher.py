from __future__ import annotations

"""Publish accepted reconstruction tracks as scene documents."""

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_role
from .reconstruction_inputs import frame_paths
from .reconstruction_team_classification import cluster_color
from .reconstruction_track_observations import (
    TrackObservationMaterialization,
    materialize_track_observations,
)
from .reconstruction_track_trajectory import (
    TrackTrajectory,
    eligible_track_candidates,
    materialize_continuous_trajectory,
    project_track_trajectory,
    resolve_track_coordinate_mode,
    trajectory_passes_acceptance,
)


def _trajectory_diagnostics() -> dict:
    return {
        "rawProjectedObservationCount": 0,
        "retainedProjectedObservationCount": 0,
        "discardedProjectedObservationCount": 0,
        "preFilterSpeedSampleCount": 0,
        "preFilterSpeedViolationCount": 0,
        "preFilterMaximumSpeedMetresPerSecond": None,
        "splitTrajectoryCount": 0,
        "discardedTrajectoryFragmentCount": 0,
        "acceptedIdentityImageObservationCount": 0,
        "publishedIdentityObservationCount": 0,
        "metricAcceptedIdentityObservationCount": 0,
        "metricRejectedIdentityObservationCount": 0,
        "metricUnprojectedIdentityObservationCount": 0,
    }


def _record_trajectory_quality(diagnostics: dict, trajectory: TrackTrajectory) -> None:
    quality = trajectory.quality
    diagnostics["rawProjectedObservationCount"] += quality["rawObservationCount"]
    diagnostics["retainedProjectedObservationCount"] += quality[
        "retainedObservationCount"
    ]
    diagnostics["discardedProjectedObservationCount"] += quality[
        "discardedObservationCount"
    ]
    diagnostics["preFilterSpeedSampleCount"] += quality["rawSpeedSampleCount"]
    diagnostics["preFilterSpeedViolationCount"] += quality[
        "impossibleSpeedSegmentCount"
    ]
    if quality["fragmentCount"] > 1:
        diagnostics["splitTrajectoryCount"] += 1
        diagnostics["discardedTrajectoryFragmentCount"] += quality[
            "discardedFragmentCount"
        ]
    if trajectory.maximum_raw_speed is not None:
        diagnostics["preFilterMaximumSpeedMetresPerSecond"] = max(
            float(diagnostics["preFilterMaximumSpeedMetresPerSecond"] or 0.0),
            trajectory.maximum_raw_speed,
        )


def _record_observation_quality(
    diagnostics: dict,
    materialized: TrackObservationMaterialization,
) -> None:
    diagnostics[
        "acceptedIdentityImageObservationCount"
    ] += materialized.image_observation_count
    diagnostics["publishedIdentityObservationCount"] += len(
        materialized.observations
    )
    diagnostics[
        "metricAcceptedIdentityObservationCount"
    ] += materialized.accepted_count
    diagnostics[
        "metricRejectedIdentityObservationCount"
    ] += materialized.rejected_count
    diagnostics[
        "metricUnprojectedIdentityObservationCount"
    ] += materialized.unprojected_count


def _publish_trajectory_diagnostics(target: dict, values: dict) -> None:
    maximum = values["preFilterMaximumSpeedMetresPerSecond"]
    if maximum is not None:
        values["preFilterMaximumSpeedMetresPerSecond"] = round(
            float(maximum),
            3,
        )
    accepted = int(values["acceptedIdentityImageObservationCount"])
    published = int(values["publishedIdentityObservationCount"])
    metric_accepted = int(values["metricAcceptedIdentityObservationCount"])
    values["identityObservationCoverage"] = round(
        published / max(1, accepted),
        3,
    )
    values["metricObservationCoverage"] = round(
        metric_accepted / max(1, published),
        3,
    )
    target.update(values)


def _track_label(
    track: TrackState,
    team: str,
    team_number: int,
) -> tuple[str, str | None]:
    role = annotation_role(track.manual_kind) or track.role
    default_label = (
        f"{team.title()} goalkeeper"
        if role == "goalkeeper"
        else "Referee"
        if role == "referee"
        else "Other person"
        if role == "other"
        else f"{team.title()} track {team_number:02d}"
    )
    return track.manual_label or default_label, role


def _track_color(
    track: TrackState,
    team: str,
    role: str | None,
    colors: dict[str, str],
) -> str:
    return (
        "#f1c84c"
        if role == "referee"
        else "#a78bfa"
        if role == "other"
        else cluster_color(track.feature)
        if role == "goalkeeper"
        else colors.get(team, "#d7dce8")
    )


def _scene_track_document(
    track: TrackState,
    *,
    team: str,
    team_number: int,
    colors: dict[str, str],
    coordinate_mode: str,
    trajectory: TrackTrajectory,
    materialized: TrackObservationMaterialization,
    duration: float,
    pitch: dict,
) -> dict:
    keyframes, presence = materialize_continuous_trajectory(
        trajectory,
        duration,
        pitch,
        track.id,
    )
    label, role = _track_label(track, team, team_number)
    return {
        "id": f"auto-{team}-{team_number:02d}",
        "label": label,
        "teamId": team,
        "color": _track_color(track, team, role, colors),
        "number": team_number if team in {"home", "away"} else 0,
        "externalPlayerId": track.manual_external_player_id,
        "source": "manual-anchor" if track.positive_annotation_ids else "automatic",
        "coordinateMode": coordinate_mode,
        **({"role": role} if role else {}),
        **(
            {"annotationIds": sorted(track.annotation_ids)}
            if track.annotation_ids
            else {}
        ),
        "trajectoryQa": {**trajectory.quality, **materialized.quality},
        "presence": presence,
        "observations": materialized.observations,
        "keyframes": keyframes,
        **(
            {
                "canonicalPersonId": track.canonical_person_id,
                "sourceTrackletIds": sorted(
                    track.source_tracklet_ids or {track.local_tracklet_id}
                ),
            }
            if track.canonical_person_id
            else {}
        ),
        **(
            {"identitySplitPartitions": dict(track.identity_split_partitions)}
            if track.identity_split_partitions
            else {}
        ),
    }


def publish_scene_tracks(
    tracks: list[TrackState],
    mapping: dict[int, str],
    colors: dict[str, str],
    frame_size: tuple[int, int],
    scene: dict,
    calibration: PitchCalibration | None = None,
    coordinate_mode: str | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    minimum = max(5, round(len(frame_paths(scene)) * 0.24))
    candidates = eligible_track_candidates(tracks, mapping, minimum)
    counts = {"home": 0, "away": 0, "officials": 0, "unknown": 0}
    result: list[dict] = []
    aggregate_quality = _trajectory_diagnostics()
    resolved_mode = resolve_track_coordinate_mode(calibration, coordinate_mode)
    pitch = scene["payload"]["pitch"]
    source_start = float(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("sourceStart")
        or 0.0
    )

    for track in candidates:
        team = mapping[track.id]
        trajectory = project_track_trajectory(
            track,
            frame_size,
            pitch,
            resolved_mode,
        )
        if trajectory is None:
            continue
        _record_trajectory_quality(aggregate_quality, trajectory)
        maximum = 11 if team in {"home", "away"} else 6
        if not trajectory_passes_acceptance(
            track,
            trajectory,
            minimum_observations=minimum,
            published_team_count=counts[team],
            maximum_team_count=maximum,
        ):
            continue
        counts[team] += 1
        materialized = materialize_track_observations(
            track,
            trajectory,
            source_start,
        )
        _record_observation_quality(aggregate_quality, materialized)
        result.append(
            _scene_track_document(
                track,
                team=team,
                team_number=counts[team],
                colors=colors,
                coordinate_mode=resolved_mode,
                trajectory=trajectory,
                materialized=materialized,
                duration=float(scene["duration"]),
                pitch=pitch,
            )
        )

    if diagnostics is not None:
        _publish_trajectory_diagnostics(diagnostics, aggregate_quality)
    return result


__all__ = ["publish_scene_tracks"]
