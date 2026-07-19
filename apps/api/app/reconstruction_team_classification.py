from __future__ import annotations

"""Classify local tracks by kit appearance and recover goalkeeper candidates."""

import cv2
import numpy as np

from .reconstruction_track_state import TrackState


def include_goalkeeper_candidates(
    tracks: list[TrackState],
    mapping: dict[int, str],
    frame_width: int,
) -> dict[int, str]:
    """Recover long-lived keepers whose distinct kit forms a third color cluster."""

    if not tracks or not mapping or frame_width <= 0:
        return mapping
    longest = max(len(track.points) for track in tracks)
    minimum = max(5, round(longest * 0.70))
    result = dict(mapping)

    def center_x(track: TrackState) -> float:
        return float(np.mean([point["px"] for point in track.points]))

    for side in ("left", "right"):
        candidates = [
            track
            for track in tracks
            if track.id not in result
            and len(track.points) >= minimum
            and -frame_width * 0.05 <= center_x(track) <= frame_width * 1.05
            and (
                center_x(track) <= frame_width * 0.12
                if side == "left"
                else center_x(track) >= frame_width * 0.88
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
            and (
                center_x(track) <= frame_width * 0.45
                if side == "left"
                else center_x(track) >= frame_width * 0.55
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
) -> tuple[dict[int, str], dict[str, str]]:
    if len(tracks) < 2:
        return {}, {"home": "#e74a3b", "away": "#e8edf2"}
    features = np.float32([track.feature for track in tracks])
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
    weights = {
        cluster: sum(
            len(track.points)
            for track, label in zip(tracks, labels.ravel())
            if int(label) == cluster
        )
        for cluster in range(cluster_count)
    }
    selected = sorted(weights, key=weights.get, reverse=True)[:2]
    if len(selected) < 2:
        return {}, {"home": "#e74a3b", "away": "#e8edf2"}

    first, second = selected
    if centers[first][8] > centers[second][8]:
        first, second = second, first
    team_by_cluster = {first: "home", second: "away"}
    mapping = {
        track.id: team_by_cluster[int(label)]
        for track, label in zip(tracks, labels.ravel())
        if int(label) in team_by_cluster
    }
    if frame_width is not None:
        mapping = include_goalkeeper_candidates(tracks, mapping, frame_width)
    colors = {
        "home": cluster_color(centers[first]),
        "away": cluster_color(centers[second]),
    }
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
