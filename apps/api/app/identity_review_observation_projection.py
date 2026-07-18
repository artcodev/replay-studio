"""Representative observation selection for identity review."""

from __future__ import annotations

from copy import deepcopy
from math import sqrt
from typing import Any, Mapping


def crop_evidence_by_observation(
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reid: dict[str, dict[str, Any]] = {}
    jersey: dict[str, dict[str, Any]] = {}
    for item in (diagnostics.get("reid") or {}).get("crops") or []:
        observation_id = str(item.get("observationId") or "")
        if observation_id:
            reid[observation_id] = deepcopy(item)
    for item in (diagnostics.get("jerseyOcr") or {}).get("crops") or []:
        observation_id = str(item.get("observationId") or "")
        if observation_id:
            jersey[observation_id] = deepcopy(item)
    return reid, jersey


def _crop_diagnostic(value: Mapping[str, Any] | None) -> dict | None:
    if value is None:
        return None
    return {
        "status": value.get("status"),
        "usable": value.get("usable"),
        "rejectionReasons": [
            str(reason) for reason in value.get("rejectionReasons") or []
        ],
        "number": value.get("number"),
        "confidence": value.get("confidence"),
    }


def _observation_score(
    observation: Mapping[str, Any],
    reid: Mapping[str, Any] | None,
    jersey: Mapping[str, Any] | None,
) -> float:
    bbox = observation.get("bbox") or {}
    area = max(0.0, float(bbox.get("width") or 0.0)) * max(
        0.0, float(bbox.get("height") or 0.0)
    )
    confidence = max(0.0, min(1.0, float(observation.get("confidence") or 0.0)))
    score = confidence * min(1.0, sqrt(area) / 180.0)
    if reid and reid.get("status") in {"usable", "ready"}:
        score += 0.25
    if reid and reid.get("usable") is True:
        score += 0.25
    if jersey and jersey.get("status") == "recognized":
        score += 0.45
    elif jersey and jersey.get("status") in {"low-confidence", "ambiguous"}:
        score += 0.18
    return score


def representative_observations(
    person: Mapping[str, Any],
    reid_by_observation: Mapping[str, dict[str, Any]],
    jersey_by_observation: Mapping[str, dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for observation in person.get("observations") or []:
        observation_id = str(
            observation.get("observationId") or observation.get("id") or ""
        )
        frame_index = observation.get("frameIndex")
        bbox = observation.get("bbox")
        if (
            not observation_id
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or not isinstance(bbox, Mapping)
            or float(bbox.get("width") or 0.0) <= 0.0
            or float(bbox.get("height") or 0.0) <= 0.0
        ):
            continue
        scene_time = float(observation.get("sceneTime") or 0.0)
        reid = reid_by_observation.get(observation_id)
        jersey = jersey_by_observation.get(observation_id)
        ranked.append(
            (
                _observation_score(observation, reid, jersey),
                scene_time,
                observation_id,
                observation,
            )
        )

    selected: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for candidate in sorted(ranked, key=lambda item: (-item[0], item[1], item[2])):
        if any(abs(candidate[1] - previous[1]) < 0.35 for previous in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (item[1], item[2]))

    result: list[dict[str, Any]] = []
    for quality, scene_time, observation_id, observation in selected:
        reid = reid_by_observation.get(observation_id)
        jersey = jersey_by_observation.get(observation_id)
        source_frame_index = observation.get("sourceFrameIndex")
        if source_frame_index is None:
            source_frame_index = observation["frameIndex"]
        result.append(
            {
                "observationId": observation_id,
                "frameIndex": int(observation["frameIndex"]),
                "sourceFrameIndex": int(source_frame_index),
                "sceneTime": scene_time,
                "sourceTime": observation.get("sourceTime"),
                "bbox": deepcopy(observation["bbox"]),
                "confidence": observation.get("confidence"),
                "reviewQuality": round(quality, 6),
                "rejectionReasons": list(
                    dict.fromkeys(
                        [
                            *(reid or {}).get("rejectionReasons", []),
                            *(jersey or {}).get("rejectionReasons", []),
                        ]
                    )
                ),
                "reid": _crop_diagnostic(reid),
                "jerseyOcr": _crop_diagnostic(jersey),
            }
        )
    return result


__all__ = ("crop_evidence_by_observation", "representative_observations")
