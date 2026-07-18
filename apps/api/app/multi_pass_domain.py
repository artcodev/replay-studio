from __future__ import annotations

"""Domain invariants and state transitions for multi-angle compositions."""

from copy import deepcopy
from datetime import UTC, datetime


class MultiPassError(RuntimeError):
    pass


def source_segments(scene: dict) -> list[dict]:
    """Return the explicit immutable dependency list of a composition."""

    multi_pass = scene.get("payload", {}).get("videoAsset", {}).get("multiPass") or {}
    result: list[dict] = []
    for item in multi_pass.get("sourcePasses") or []:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("sceneId") or "")
        segment_id = str(item.get("segmentId") or item.get("id") or "")
        if not scene_id or not segment_id:
            continue
        result.append(
            {
                **deepcopy(item),
                "id": segment_id,
                "segmentId": segment_id,
                "sceneId": scene_id,
                "label": item.get("label") or segment_id,
            }
        )
    return result


def mark_multi_pass_failed(
    scene: dict,
    message: str,
    passes: list[dict] | None = None,
) -> None:
    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    multi_pass.update(
        {"status": "failed", "passes": passes or multi_pass.get("passes") or []}
    )
    reconstruction = video.get("reconstruction") or {}
    reconstruction.update(
        {
            "status": "failed",
            "error": message,
            "completedAt": datetime.now(UTC).isoformat(),
        }
    )
    video["processingState"] = "multi-pass-failed"
