from __future__ import annotations

"""Read-only projections over persisted scene reconstruction data."""

from copy import deepcopy

import numpy as np

from .pitch_calibration_contract import PitchCalibration

def interpolate_scene_keyframes(keyframes: list[dict], time: float) -> dict | None:
    if not keyframes:
        return None
    if time <= float(keyframes[0]["t"]):
        return keyframes[0]
    if time >= float(keyframes[-1]["t"]):
        return keyframes[-1]
    for index in range(1, len(keyframes)):
        right = keyframes[index]
        if float(right["t"]) < time:
            continue
        left = keyframes[index - 1]
        span = max(0.0001, float(right["t"]) - float(left["t"]))
        progress = (time - float(left["t"])) / span
        return {
            "t": time,
            "x": float(left["x"]) + (float(right["x"]) - float(left["x"])) * progress,
            "z": float(left["z"]) + (float(right["z"]) - float(left["z"])) * progress,
            "confidence": float(left.get("confidence") or 0.0)
            + (float(right.get("confidence") or 0.0) - float(left.get("confidence") or 0.0))
            * progress,
        }
    return keyframes[-1]


def saved_pitch_calibration(scene: dict) -> PitchCalibration | None:
    metadata = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("pitchCalibration")
        or {}
    )
    if metadata.get("status") not in {"ready", "review", "approximate"}:
        return None
    matrix = metadata.get("imageToPitch") or np.eye(3, dtype=np.float64)
    return PitchCalibration(
        image_to_pitch=np.asarray(matrix, dtype=np.float64),
        confidence=float(metadata.get("confidence") or 0.0),
        supported_lines=int(metadata.get("supportedLines") or 0),
        mean_line_score=float(metadata.get("meanLineScore") or 0.0),
        rectangle=str(metadata.get("rectangle") or ""),
        matched_curves=int(metadata.get("matchedCurves") or 0),
    )


def bbox_payload_box(bbox: dict) -> tuple[float, float, float, float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    return x, y, x + float(bbox["width"]), y + float(bbox["height"])


def canonical_analysis_subjects(scene: dict) -> list[dict]:
    payload = scene.get("payload", {})
    render_tracks = payload.get("tracks") or []
    canonical_people = payload.get("canonicalPeople") or []
    if not canonical_people:
        return [deepcopy(track) for track in render_tracks]
    by_id = {str(track.get("id")): track for track in render_tracks if track.get("id")}
    by_canonical = {
        str(track.get("canonicalPersonId")): track
        for track in render_tracks
        if track.get("canonicalPersonId")
    }
    result = []
    for person in canonical_people:
        canonical_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        render = by_id.get(str(person.get("renderTrackId") or "")) or by_canonical.get(
            canonical_id
        )
        result.append(
            {
                **(deepcopy(render) if render is not None else {}),
                "id": render.get("id") if render is not None else None,
                "canonicalPersonId": canonical_id,
                "label": person.get("displayName")
                or (render.get("label") if render is not None else canonical_id),
                "displayName": person.get("displayName"),
                "identityStatus": person.get("identityStatus"),
                "identityConfidence": person.get("identityConfidence"),
                "identitySource": person.get("identitySource"),
                "jerseyNumber": person.get("jerseyNumber"),
                "teamId": person.get("teamId")
                or (render.get("teamId") if render is not None else None),
                "role": person.get("role")
                or (render.get("role") if render is not None else None),
                "externalPlayerId": person.get("externalPlayerId"),
                "annotationIds": person.get("annotationIds")
                or (render.get("annotationIds") if render is not None else []),
                "observations": deepcopy(person.get("observations") or []),
                "keyframes": deepcopy(render.get("keyframes") or [])
                if render is not None
                else [],
                "renderTrackId": render.get("id") if render is not None else None,
            }
        )
    return result
