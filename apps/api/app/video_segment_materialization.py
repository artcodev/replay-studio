from __future__ import annotations

from .project_resource_repository import project_resources
from .project_segment_contract import SegmentUpsert
from .scene_repository import scenes
from .video_segment_planning import build_segment_scene


def materialize_segment_scene(parent: dict, segment: dict) -> dict:
    existing = scenes.find_segment_scene(parent["id"], segment["id"])
    if existing:
        segment["sceneId"] = existing["id"]
        _link_materialized_segment(parent, existing, segment)
        return existing

    child = scenes.put(build_segment_scene(parent, segment))
    _link_materialized_segment(parent, child, segment)
    return child


def _link_materialized_segment(parent: dict, child: dict, segment: dict) -> None:
    project_id = project_resources.scene_owner(str(parent["id"]))
    if project_id is None:
        return
    project_resources.link_scene(project_id, str(child["id"]), role="segment")
    video = parent.get("payload", {}).get("videoAsset") or {}
    project_resources.upsert_segment(
        project_id,
        SegmentUpsert(
            video_asset_id=str(video.get("id")) if video.get("id") else None,
            scene_id=str(child["id"]),
            source_segment_id=str(segment["id"]),
            label=str(segment.get("label") or segment["id"]),
            start_seconds=float(segment.get("start") or 0.0),
            end_seconds=float(segment.get("end") or child["duration"]),
            ordinal=max(0, int(segment.get("ordinal") or 0)),
            payload=dict(segment),
        ),
    )
