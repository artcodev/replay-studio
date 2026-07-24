from __future__ import annotations

"""Classify local tracks by kit appearance and recover goalkeeper candidates."""

import cv2
import numpy as np

from .reconstruction_identity_semantics import annotation_team
from .reconstruction_track_state import TrackState


DEFAULT_TEAM_COLORS = {"home": "#e74a3b", "away": "#e8edf2"}
GOALKEEPER_PENALTY_AREA_X = 36.0
NEAREST_TEAM_PROTOTYPE_MAX_DISTANCE = 0.85
MINIMUM_TEAM_PROTOTYPE_MARGIN = 0.08


def _metric_center_x(track: TrackState) -> float | None:
    values = [
        float(point["pitchX"])
        for point in track.points
        if point.get("pitchX") is not None
    ]
    return float(np.median(values)) if values else None


def _robust_track_feature(track: TrackState) -> tuple[np.ndarray, dict]:
    samples = [
        np.asarray(point["_appearanceFeature"], dtype=np.float32)
        for point in track.points
        if point.get("_appearanceFeature") is not None
    ]
    samples = [
        item
        for item in samples
        if item.shape == (12,) and np.isfinite(item).all()
    ]
    if len(samples) < 3:
        return np.asarray(track.feature, dtype=np.float32), {
            "sampleCount": len(samples),
            "retainedSampleCount": len(samples),
            "aggregation": "mean-fallback",
        }
    values = np.stack(samples)
    median = np.median(values, axis=0)
    distances = np.linalg.norm(values - median, axis=1)
    keep_count = max(3, int(round(len(values) * 0.75)))
    kept = values[np.argsort(distances)[:keep_count]]
    feature = np.mean(kept, axis=0).astype(np.float32)
    hue_mass = float(feature[:8].sum())
    if hue_mass > 1e-6:
        feature[:8] /= hue_mass
    return feature, {
        "sampleCount": len(samples),
        "retainedSampleCount": keep_count,
        "aggregation": "trimmed-frame-feature-mean",
        "discardedOutlierCount": len(samples) - keep_count,
    }


def include_goalkeeper_candidates(
    tracks: list[TrackState],
    mapping: dict[int, str],
    frame_width: int,
) -> dict[int, str]:
    """Recover long-lived keepers whose distinct kit forms a third color cluster."""

    if not tracks or not mapping:
        return mapping
    longest = max(len(track.points) for track in tracks)
    minimum = max(5, round(longest * 0.70))
    result = dict(mapping)

    for side in ("left", "right"):
        candidates = [
            track
            for track in tracks
            if track.id not in result
            and len(track.points) >= minimum
            and _metric_center_x(track) is not None
            and (
                float(_metric_center_x(track)) <= -GOALKEEPER_PENALTY_AREA_X
                if side == "left"
                else float(_metric_center_x(track)) >= GOALKEEPER_PENALTY_AREA_X
            )
        ]
        if not candidates:
            continue
        candidate = max(
            candidates,
            key=lambda track: (len(track.points), track.feature_count),
        )
        nearby = [
            track
            for track in tracks
            if track.id in result
            and _metric_center_x(track) is not None
            and (
                float(_metric_center_x(track)) < 0.0
                if side == "left"
                else float(_metric_center_x(track)) >= 0.0
            )
        ]
        if not nearby:
            nearby = [track for track in tracks if track.id in result]
        support = {
            team: sum(
                len(track.points)
                for track in nearby
                if result[track.id] == team
            )
            for team in ("home", "away")
        }
        team = max(
            ("home", "away"),
            key=lambda item: (support[item], item == "away"),
        )
        if sum(value == team for value in result.values()) >= 11:
            continue
        candidate.role = "goalkeeper"
        result[candidate.id] = team
    return result


def include_reid_official_candidates(
    tracks: list[TrackState],
    mapping: dict[int, str],
) -> list[str]:
    """Map referee-voted tracks outside both kit clusters to ``officials``.

    The third kit cluster used to be dropped silently even when the PRTReID
    role votes had already elected "referee" for a long-lived track. Manual
    assignments and existing team memberships are never overridden.
    """

    official_tracklets: list[str] = []
    for track in tracks:
        if track.id in mapping or track.manual_kind:
            continue
        if track.role == "referee":
            mapping[track.id] = "officials"
            official_tracklets.append(track.local_tracklet_id)
    return official_tracklets


def team_clusters(
    tracks: list[TrackState],
    frame_width: int | None = None,
    diagnostics: dict | None = None,
) -> tuple[dict[int, str], dict[str, str]]:
    if len(tracks) < 2:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "schemaVersion": 1,
                    "status": "insufficient-tracks",
                    "trackCount": len(tracks),
                }
            )
        return {}, dict(DEFAULT_TEAM_COLORS)
    robust = [_robust_track_feature(track) for track in tracks]
    features = np.float32([item[0] for item in robust])
    cluster_count = 3 if len(tracks) >= 6 else 2
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(
        features,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01),
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    median_length = float(np.median([len(track.points) for track in tracks]))
    # Cap one track's vote so an ID-switched long track cannot select a
    # contaminated third cluster as an entire team.
    weights = {
        cluster: sum(
            min(len(track.points), max(5.0, median_length * 1.5))
            for track, label in zip(tracks, labels.ravel())
            if int(label) == cluster
        )
        for cluster in range(cluster_count)
    }
    selected = sorted(weights, key=weights.get, reverse=True)[:2]
    if len(selected) < 2:
        return {}, dict(DEFAULT_TEAM_COLORS)

    first, second = selected
    manual_votes = {
        cluster: {
            team: sum(
                1
                for track, label in zip(tracks, labels.ravel())
                if int(label) == cluster
                and annotation_team(track.manual_kind) == team
            )
            for team in ("home", "away")
        }
        for cluster in selected
    }
    first_manual = max(
        ("home", "away"),
        key=lambda team: manual_votes[first][team],
    )
    second_manual = max(
        ("home", "away"),
        key=lambda team: manual_votes[second][team],
    )
    manual_pair_is_authoritative = (
        manual_votes[first][first_manual] > 0
        and manual_votes[second][second_manual] > 0
        and first_manual != second_manual
    )
    if manual_pair_is_authoritative:
        team_by_cluster = {
            first: first_manual,
            second: second_manual,
        }
        if team_by_cluster[first] == "away":
            first, second = second, first
    else:
        # Deterministic fallback only establishes home/away naming. It does
        # not claim semantic confidence and is exposed in diagnostics.
        if centers[first][8] > centers[second][8]:
            first, second = second, first
        team_by_cluster = {first: "home", second: "away"}
    assignment_rows: list[dict] = []
    mapping: dict[int, str] = {}
    for track, label, feature, feature_diagnostics in zip(
        tracks,
        labels.ravel(),
        features,
        (item[1] for item in robust),
    ):
        cluster = int(label)
        distances = {
            selected_cluster: float(
                np.linalg.norm(feature - centers[selected_cluster])
            )
            for selected_cluster in selected
        }
        assigned_prototype = min(distances, key=distances.get)
        own_distance = distances[assigned_prototype]
        alternative_distances = [
            value
            for candidate, value in distances.items()
            if candidate != assigned_prototype
        ]
        alternative = min(alternative_distances) if alternative_distances else None
        margin = (
            max(0.0, alternative - own_distance) / max(1e-6, alternative)
            if own_distance is not None and alternative is not None
            else None
        )
        manual_team = annotation_team(track.manual_kind)
        team = manual_team or team_by_cluster.get(assigned_prototype)
        is_selected_cluster = cluster in selected
        accepted = bool(
            team
            and (
                manual_team is not None
                or (
                    margin is None
                    or margin >= MINIMUM_TEAM_PROTOTYPE_MARGIN
                )
                and (
                    is_selected_cluster
                    or own_distance <= NEAREST_TEAM_PROTOTYPE_MAX_DISTANCE
                )
            )
        )
        if accepted:
            mapping[track.id] = team
        status = (
            "accepted-manual"
            if accepted and manual_team
            else "accepted"
            if accepted and is_selected_cluster
            else "accepted-nearest-team-prototype"
            if accepted
            else "ambiguous"
        )
        assignment_rows.append(
            {
                "trackletId": track.local_tracklet_id,
                "cluster": cluster,
                "assignedPrototypeCluster": assigned_prototype,
                "team": team if accepted else None,
                "status": status,
                "distance": (
                    round(own_distance, 5)
                    if own_distance is not None
                    else None
                ),
                "margin": round(margin, 5) if margin is not None else None,
                **feature_diagnostics,
            }
        )
    before_goalkeeper_recovery = set(mapping)
    if frame_width is not None:
        mapping = include_goalkeeper_candidates(tracks, mapping, frame_width)
    recovered_goalkeepers = set(mapping) - before_goalkeeper_recovery
    if recovered_goalkeepers:
        by_tracklet = {
            track.local_tracklet_id: track for track in tracks
        }
        for row in assignment_rows:
            track = by_tracklet[row["trackletId"]]
            if track.id not in recovered_goalkeepers:
                continue
            row["team"] = mapping[track.id]
            row["status"] = "accepted-goalkeeper-position"
    colors = {
        "home": cluster_color(centers[first]),
        "away": cluster_color(centers[second]),
    }
    if diagnostics is not None:
        diagnostics.update(
            {
                "schemaVersion": 1,
                "status": "ready",
                "method": "robust-track-hsv-kmeans",
                "trackCount": len(tracks),
                "clusterCount": cluster_count,
                "selectedClusters": [first, second],
                "clusterWeights": {
                    str(key): round(float(value), 3)
                    for key, value in sorted(weights.items())
                },
                "labelSource": (
                    "manual-team-anchors"
                    if manual_pair_is_authoritative
                    else "white-ratio-ordering"
                ),
                "colors": dict(colors),
                "assignments": assignment_rows,
                "acceptedTrackCount": len(mapping),
                "ambiguousTrackCount": sum(
                    item["status"] == "ambiguous"
                    for item in assignment_rows
                ),
                "nearestPrototypeMaximumDistance": (
                    NEAREST_TEAM_PROTOTYPE_MAX_DISTANCE
                ),
                "minimumPrototypeMargin": MINIMUM_TEAM_PROTOTYPE_MARGIN,
                "goalkeeperPenaltyAreaThresholdMetres": (
                    GOALKEEPER_PENALTY_AREA_X
                ),
                "goalkeeperRecoveredTrackCount": len(recovered_goalkeepers),
            }
        )
    return mapping, colors


def cluster_color(center: np.ndarray) -> str:
    white_ratio, dark_ratio = float(center[8]), float(center[9])
    if white_ratio > 0.28:
        return "#e8edf2"
    if dark_ratio > 0.58:
        return "#30363d"
    hue_bin = int(np.argmax(center[:8]))
    hue = 2 if hue_bin == 0 else int(hue_bin * 22.5 + 11.25)
    pixel = np.uint8([[[hue, 205, 225]]])
    blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


__all__ = ["cluster_color", "include_goalkeeper_candidates", "team_clusters"]
