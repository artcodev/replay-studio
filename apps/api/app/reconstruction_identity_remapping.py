from __future__ import annotations

"""Remap persisted canonical identities onto freshly rebuilt raw tracks."""

from math import hypot

import numpy as np

from .reconstruction_errors import IdentityCorrectionError
from .reconstruction_track_state import TrackState
from .reconstruction_identity_semantics import annotation_role, annotation_team
from .bounding_box_geometry import intersection_over_union

def raw_track_match_score(track: TrackState, target: dict) -> dict | None:
    target_observations = {
        int(item["frameIndex"]): item
        for item in target.get("observations") or []
        if item.get("frameIndex") is not None and item.get("bbox")
    }
    image_costs: list[float] = []
    image_times: list[float] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox")
        if frame_index is None or not bbox or int(frame_index) not in target_observations:
            continue
        target_bbox = target_observations[int(frame_index)]["bbox"]
        overlap = intersection_over_union(
            (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            ),
            (
                float(target_bbox["x"]),
                float(target_bbox["y"]),
                float(target_bbox["x"]) + float(target_bbox["width"]),
                float(target_bbox["y"]) + float(target_bbox["height"]),
            ),
        )
        if overlap >= 0.25:
            image_costs.append((1.0 - overlap) * 2.0)
            image_times.append(float(point["t"]))
    if image_costs:
        return {
            "median": float(np.median(image_costs)),
            "p90": float(np.percentile(image_costs, 90)),
            "normalizedMedian": float(np.median(image_costs)) / 2.0,
            "overlap": len(image_costs),
            "span": max(image_times) - min(image_times),
            "source": "image-observation-overlap",
        }

    keyframes = [
        keyframe
        for keyframe in target.get("keyframes") or []
        if keyframe.get("observed") is not False
    ]
    distances: list[float] = []
    normalized_distances: list[float] = []
    shared_times: list[float] = []
    for point in track.points:
        if point.get("pitchX") is None or point.get("pitchZ") is None or not keyframes:
            continue
        nearest = min(keyframes, key=lambda item: abs(float(item["t"]) - float(point["t"])))
        if abs(float(nearest["t"]) - float(point["t"])) > 0.16:
            continue
        distance = hypot(
            float(nearest["x"]) - float(point["pitchX"]),
            float(nearest["z"]) - float(point["pitchZ"]),
        )
        uncertainty = max(
            0.5,
            float(nearest.get("positionUncertaintyMetres") or 0.0)
            + float(point.get("positionUncertaintyMetres") or 0.0),
        )
        distances.append(distance)
        normalized_distances.append(distance / uncertainty)
        shared_times.append(float(point["t"]))
    if not distances:
        return None
    return {
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "normalizedMedian": float(np.median(normalized_distances)),
        "overlap": len(distances),
        "span": max(shared_times) - min(shared_times),
    }


def raw_track_matches_identity_metadata(track: TrackState, target: dict) -> bool:
    target_team = str(target.get("teamId") or "") or None
    target_role = str(target.get("role") or "") or None
    track_team = annotation_team(track.manual_kind)
    track_role = annotation_role(track.manual_kind)
    if target_team and track_team and target_team != track_team:
        return False
    if target_role and track_role and target_role != track_role:
        return False
    target_player = str(target.get("externalPlayerId") or "") or None
    track_player = str(track.manual_external_player_id or "") or None
    return not (target_player and track_player and target_player != track_player)


def resolve_previous_identity_track(
    tracks: list[TrackState],
    target: dict,
    *,
    correction_id: str,
    action: str,
    source_track_id: str | None = None,
    target_id: str | None = None,
    exclude: TrackState | None = None,
) -> TrackState:
    candidates = [
        track
        for track in tracks
        if track is not exclude and raw_track_matches_identity_metadata(track, target)
    ]
    target_player = str(target.get("externalPlayerId") or "") or None
    if target_player:
        exact_roster = [
            track
            for track in candidates
            if track.manual_external_player_id == target_player
        ]
        if len(exact_roster) == 1:
            return exact_roster[0]
        if len(exact_roster) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} is ambiguous across roster-matched tracks",
                correction_id=correction_id,
                action=action,
                status="ambiguous",
                reason="multiple-roster-matches",
                source_track_id=source_track_id,
                target_id=target_id,
                candidates=[{"rawTrackId": track.id} for track in exact_roster],
            )

    scored: list[tuple[dict, TrackState]] = []
    for track in candidates:
        score = raw_track_match_score(track, target)
        if score is None or score["overlap"] < 3 or score["span"] < 0.4:
            continue
        scored.append((score, track))
    scored.sort(
        key=lambda item: (
            item[0]["median"],
            item[0]["p90"],
            -item[0]["overlap"],
        )
    )
    candidate_diagnostics = [
        {
            "rawTrackId": track.id,
            "medianDistanceMetres": round(float(score["median"]), 3),
            "p90DistanceMetres": round(float(score["p90"]), 3),
            "normalizedMedian": round(float(score["normalizedMedian"]), 3),
            "overlapSamples": int(score["overlap"]),
            "overlapSpanSeconds": round(float(score["span"]), 3),
        }
        for score, track in scored
    ]
    if not scored:
        raise IdentityCorrectionError(
            f"Identity correction {correction_id} could not resolve its previous trajectory",
            correction_id=correction_id,
            action=action,
            status="unresolved",
            reason="insufficient-observation-overlap",
            source_track_id=source_track_id,
            target_id=target_id,
            candidates=[{"rawTrackId": track.id} for track in candidates],
        )
    best_score, best_track = scored[0]
    if (
        best_score["median"] > 4.0
        or best_score["p90"] > 6.0
        or best_score["normalizedMedian"] > 2.0
    ):
        raise IdentityCorrectionError(
            f"Identity correction {correction_id} no longer matches the rebuilt trajectory",
            correction_id=correction_id,
            action=action,
            status="unresolved",
            reason="trajectory-outside-remap-threshold",
            source_track_id=source_track_id,
            target_id=target_id,
            candidates=candidate_diagnostics,
        )
    if len(scored) > 1:
        runner_up = scored[1][0]
        absolute_margin = runner_up["median"] - best_score["median"]
        relative_margin = runner_up["median"] / max(0.25, best_score["median"])
        if absolute_margin < 2.0 and relative_margin < 1.5:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} is ambiguous between nearby trajectories",
                correction_id=correction_id,
                action=action,
                status="ambiguous",
                reason="nearby-trajectories",
                source_track_id=source_track_id,
                target_id=target_id,
                candidates=candidate_diagnostics,
            )
    return best_track
