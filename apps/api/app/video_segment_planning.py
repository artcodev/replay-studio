from __future__ import annotations

from .project_identifiers import stable_identifier
from .sample import make_video_scene


def rank_reconstruction_shots(
    segments: list[dict],
    limit: int = 5,
) -> list[dict]:
    eligible = [segment for segment in segments if segment["duration"] >= 4.0]
    ranked = sorted(
        eligible,
        key=lambda item: (item["score"], item["duration"]),
        reverse=True,
    )[:limit]
    recommended_ids = {item["id"] for item in ranked}
    for segment in segments:
        segment["recommended"] = segment["id"] in recommended_ids
    return ranked


def build_segment_scene(parent: dict, segment: dict) -> dict:
    """Build a deterministic child Scene without performing any writes."""

    video = parent["payload"]["videoAsset"]
    scene_id = f"moment-{video['id'].removeprefix('asset-')}-{segment['id']}"
    child_video = {
        **video,
        "sourceStart": segment["start"],
        "sourceEnd": segment["end"],
        "parentSceneId": parent["id"],
        "selectedSegmentId": segment["id"],
        "segments": [],
    }
    child = make_video_scene(
        scene_id=scene_id,
        title=f"{parent['title']} · {segment['label']}",
        duration=segment["duration"],
        video_asset=child_video,
    )
    segment["sceneId"] = child["id"]
    return child


def build_recommended_segment_scenes(
    parent: dict,
    segments: list[dict],
) -> list[dict]:
    ranked = rank_reconstruction_shots(segments)
    children = [build_segment_scene(parent, segment) for segment in ranked]
    parent["payload"]["videoAsset"]["segments"] = segments
    if children:
        parent["payload"]["videoAsset"]["primarySceneId"] = children[0]["id"]
    return children
