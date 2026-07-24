from __future__ import annotations

"""Canonical scene-frame exclusions shared by every analysis consumer."""

from copy import deepcopy
from typing import Mapping

from .reconstruction_errors import ReconstructionError


def scene_frame_exclusions(scene: Mapping) -> list[dict]:
    """Return validated exclusions ordered by their immutable source frame."""

    payload = scene.get("payload")
    video = payload.get("videoAsset") if isinstance(payload, Mapping) else None
    raw = video.get("frameExclusions") if isinstance(video, Mapping) else None
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ReconstructionError("Scene frame exclusions are malformed")

    result: list[dict] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReconstructionError("Scene frame exclusions are malformed")
        try:
            source_frame_index = int(item["sourceFrameIndex"])
            scene_time = float(item["sceneTime"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionError("Scene frame exclusions are malformed") from exc
        if source_frame_index <= 0 or scene_time < 0:
            raise ReconstructionError("Scene frame exclusions are malformed")
        if source_frame_index in seen:
            continue
        result.append(
            {
                "sourceFrameIndex": source_frame_index,
                "sceneTime": scene_time,
                **(
                    {"excludedAt": str(item["excludedAt"])}
                    if item.get("excludedAt")
                    else {}
                ),
            }
        )
        seen.add(source_frame_index)
    result.sort(key=lambda item: int(item["sourceFrameIndex"]))
    return deepcopy(result)


def excluded_source_frame_indices(scene: Mapping) -> set[int]:
    return {
        int(item["sourceFrameIndex"])
        for item in scene_frame_exclusions(scene)
    }


def frame_exclusion_fingerprint_input(scene: Mapping) -> list[dict]:
    return [
        {
            "sourceFrameIndex": int(item["sourceFrameIndex"]),
            "sceneTime": round(float(item["sceneTime"]), 9),
        }
        for item in scene_frame_exclusions(scene)
    ]


__all__ = (
    "excluded_source_frame_indices",
    "frame_exclusion_fingerprint_input",
    "scene_frame_exclusions",
)
