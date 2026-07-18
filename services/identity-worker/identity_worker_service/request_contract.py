from __future__ import annotations

import json
from typing import Any


class IdentityRequestError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def parse_manifest(raw: str, frame_count: int) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IdentityRequestError("manifest is not valid JSON") from exc
    frame_items = value.get("frames") if isinstance(value, dict) else None
    if not isinstance(frame_items, list):
        raise IdentityRequestError("manifest.frames must be an array")
    seen: set[str] = set()
    for frame in frame_items:
        if not isinstance(frame, dict):
            raise IdentityRequestError("Each manifest frame must be an object")
        file_index = frame.get("fileIndex")
        if not isinstance(file_index, int) or not 0 <= file_index < frame_count:
            raise IdentityRequestError("manifest fileIndex is out of range")
        observations = frame.get("observations")
        if not isinstance(observations, list):
            raise IdentityRequestError("frame.observations must be an array")
        for observation in observations:
            observation_id = (
                observation.get("observationId")
                if isinstance(observation, dict)
                else None
            )
            bbox = observation.get("bbox") if isinstance(observation, dict) else None
            if not isinstance(observation_id, str) or not observation_id.strip():
                raise IdentityRequestError("observationId is required")
            if observation_id in seen:
                raise IdentityRequestError(f"Duplicate observationId: {observation_id}")
            seen.add(observation_id)
            if not isinstance(bbox, dict) or any(
                not isinstance(bbox.get(key), (int, float))
                for key in ("x", "y", "width", "height")
            ):
                raise IdentityRequestError(f"Invalid bbox for {observation_id}")
            if float(bbox["width"]) <= 0 or float(bbox["height"]) <= 0:
                raise IdentityRequestError(f"Empty bbox for {observation_id}")
    return frame_items

