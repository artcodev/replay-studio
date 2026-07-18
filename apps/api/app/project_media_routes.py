from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .multi_pass_composition import create_project_multi_pass_scene
from .multi_pass_domain import MultiPassError
from .project_http_contracts import (
    ProjectCompositionRequest,
    PublicProjectAsset,
    PublicProjectSegment,
)
from .project_http_errors import project_http_error
from .project_resource_repository import (
    ProjectResourceNotFound,
    project_resources,
)
from .project_segment_contract import SegmentDocument
from .project_store import project_store
from .scene_contracts import SceneDocument, SceneSummary
from .reconstruction_job_queries import reconstruction_jobs
from .scene_repository import scenes
from .video_segment_materialization import materialize_segment_scene
from .video_store import video_store


router = APIRouter(prefix="/api/projects/{project_id}", tags=["media"])


@router.get("/assets", response_model=list[PublicProjectAsset])
def list_project_assets(project_id: str) -> list[PublicProjectAsset]:
    try:
        links = project_resources.list_video_asset_links(project_id)
    except ProjectResourceNotFound as exc:
        raise project_http_error(exc) from exc
    assets = {
        item["id"]: item
        for item in video_store.list_by_ids(
            [link.video_asset_id for link in links]
        )
    }
    result: list[PublicProjectAsset] = []
    for link in links:
        asset = assets.get(link.video_asset_id)
        if asset is None:
            continue
        raw_status = str(asset.get("status") or "processing")
        status = "uploading" if raw_status == "queued" else raw_status
        result.append(
            PublicProjectAsset(
                id=str(asset["id"]),
                project_id=project_id,
                timeline_scene_id=(
                    str(asset["scene_id"]) if asset.get("scene_id") else None
                ),
                filename=str(
                    asset.get("original_name") or asset.get("filename") or "video"
                ),
                duration=asset.get("duration"),
                status=status,
                media_url=(
                    f"/api/projects/{project_id}/videos/{asset['id']}/media"
                    if status == "ready"
                    else None
                ),
                poster_url=(
                    f"/api/projects/{project_id}/videos/{asset['id']}/poster"
                    if status == "ready"
                    else None
                ),
                created_at=str(asset.get("created_at") or ""),
            )
        )
    return result


@router.get("/segments", response_model=list[PublicProjectSegment])
def list_project_segments(project_id: str) -> list[PublicProjectSegment]:
    try:
        segments = project_resources.list_segments(project_id)
    except ProjectResourceNotFound as exc:
        raise project_http_error(exc) from exc
    statuses = reconstruction_jobs.statuses(
        [segment.scene_id for segment in segments if segment.scene_id]
    )
    result: list[PublicProjectSegment] = []
    for segment in segments:
        status = "pending"
        if segment.scene_id:
            raw_status = statuses.get(segment.scene_id)
            status = (
                "analyzing"
                if raw_status in {"queued", "processing"}
                else "failed"
                if raw_status in {"failed", "cancelled"}
                else "ready"
            )
        result.append(
            PublicProjectSegment(
                id=segment.id,
                project_id=project_id,
                asset_id=segment.video_asset_id or "",
                source_segment_id=segment.source_segment_id,
                scene_id=segment.scene_id,
                label=segment.label or segment.source_segment_id,
                start=segment.start_seconds,
                end=segment.end_seconds,
                status=status,
            )
        )
    return result


@router.post("/compositions", response_model=SceneDocument, status_code=202)
def create_project_composition(
    project_id: str,
    request: ProjectCompositionRequest,
) -> dict[str, Any]:
    if project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    segment_by_id = {
        segment.id: segment
        for segment in project_resources.list_segments(project_id)
    }
    missing = [value for value in request.segment_ids if value not in segment_by_id]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Project segments not found: {', '.join(missing)}",
        )

    sources: list[dict[str, Any]] = []
    parent_scene_id: str | None = None
    for segment_id in request.segment_ids:
        segment = segment_by_id[segment_id]
        child = scenes.get(segment.scene_id) if segment.scene_id else None
        if child is None:
            child = _materialize_project_segment(segment, segment_id)
        if project_resources.scene_owner(str(child["id"])) != project_id:
            raise HTTPException(
                status_code=409,
                detail=f"Segment {segment_id} belongs to another project",
            )
        video = child.get("payload", {}).get("videoAsset") or {}
        parent_scene_id = parent_scene_id or str(
            video.get("parentSceneId") or child["id"]
        )
        sources.append(
            {
                "id": segment.id,
                "segmentId": segment.id,
                "sourceSegmentId": segment.source_segment_id,
                "sceneId": str(child["id"]),
                "assetId": segment.video_asset_id or video.get("id"),
                "label": segment.label or segment.source_segment_id,
                "start": segment.start_seconds,
                "end": segment.end_seconds,
                "duration": segment.end_seconds - segment.start_seconds,
                "score": (segment.payload or {}).get("score", 0.0),
            }
        )
    try:
        return create_project_multi_pass_scene(
            parent_scene_id or str(sources[0]["sceneId"]),
            sources,
            project_id=project_id,
            title=request.title,
            manual_alignment_anchors=request.manual_alignment_anchors,
        )
    except MultiPassError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _materialize_project_segment(
    segment: SegmentDocument,
    segment_id: str,
) -> dict:
    asset = video_store.get(segment.video_asset_id) if segment.video_asset_id else None
    root = (
        scenes.get(str(asset.get("scene_id")))
        if asset and asset.get("scene_id")
        else None
    )
    if root is None:
        raise HTTPException(
            status_code=409,
            detail=f"Segment {segment_id} has no materialized source scene",
        )
    raw = next(
        (
            item
            for item in root.get("payload", {})
            .get("videoAsset", {})
            .get("segments", [])
            if item.get("id") == segment.source_segment_id
        ),
        None,
    )
    if raw is None:
        raise HTTPException(
            status_code=409,
            detail=f"Source range for segment {segment_id} was not found",
        )
    return materialize_segment_scene(root, raw)


@router.get("/scenes", response_model=list[SceneSummary])
def list_project_scenes(project_id: str) -> list[dict[str, Any]]:
    try:
        links = project_resources.list_scene_links(project_id)
    except ProjectResourceNotFound as exc:
        raise project_http_error(exc) from exc
    role_by_id = {link.scene_id: link.role for link in links}
    summaries = {
        item["id"]: item for item in scenes.list_by_ids(list(role_by_id))
    }
    return [
        summaries[scene_id]
        for scene_id, _role in sorted(
            role_by_id.items(),
            key=lambda item: (item[1] != "root", item[1], item[0]),
        )
        if scene_id in summaries
    ]
