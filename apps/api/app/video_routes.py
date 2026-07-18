from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import get_settings
from .pipeline_domain import PipelineJobConflict
from .project_store import project_store
from . import project_resource_access
from .scene_contracts import SceneDocument, VideoAsset
from .segment_layout import propose_segment_layout
from .scene_repository import scenes
from .video_media_paths import asset_directory, published_video_directory
from .video_processing_contract import VideoProcessingError, video_processing_run_id
from .video_segment_materialization import materialize_segment_scene
from .video_pipeline import video_pipeline
from .video_store import video_store
from .video_http_views import project_video_view


router = APIRouter(prefix="/api/projects/{project_id}/videos", tags=["videos"])
settings = get_settings()


@router.post("", response_model=VideoAsset, status_code=202)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    if not project_store.project_exists(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(
            status_code=415,
            detail="Supported formats: MP4, MOV, MKV, WebM, M4V",
        )
    asset_id = f"asset-{uuid4().hex[:12]}"
    directory = asset_directory(asset_id)
    directory.mkdir(parents=True, exist_ok=False)
    stored_name = f"source{suffix}"
    destination = directory / stored_name
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_video_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Video is larger than the 250 MB upload limit",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        await file.close()

    run_id = video_processing_run_id(project_id, asset_id)
    try:
        video_pipeline.enqueue_upload(
            job_id=run_id,
            project_id=project_id,
            asset_id=asset_id,
            filename=stored_name,
            original_name=Path(file.filename or "clip.mp4").name[:240],
            content_type=(file.content_type or "application/octet-stream")[:120],
            title=title,
        )
    except PipelineJobConflict as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    asset = video_store.get(asset_id)
    if asset is None:
        raise RuntimeError("Atomic video upload committed without its asset row")
    return project_video_view(project_id, asset)


@router.get("/{asset_id}", response_model=VideoAsset)
def get_video(project_id: str, asset_id: str):
    return project_video_view(
        project_id,
        project_resource_access.project_video_or_404(project_id, asset_id),
    )


@router.get("/{asset_id}/media")
def video_media(project_id: str, asset_id: str):
    asset = project_resource_access.project_video_or_404(project_id, asset_id)
    try:
        path = published_video_directory(asset) / "proxy.mp4"
    except VideoProcessingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=409, detail="Browser proxy is not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{asset_id}.mp4")


@router.get("/{asset_id}/poster")
def video_poster(project_id: str, asset_id: str):
    asset = project_resource_access.project_video_or_404(project_id, asset_id)
    try:
        path = published_video_directory(asset) / "poster.jpg"
    except VideoProcessingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=409, detail="Poster is not ready")
    return FileResponse(path, media_type="image/jpeg")


@router.post(
    "/{asset_id}/segments/{segment_id}/scene",
    response_model=SceneDocument,
    status_code=201,
)
def create_segment_scene(project_id: str, asset_id: str, segment_id: str):
    asset = project_resource_access.project_video_or_404(project_id, asset_id)
    if not asset.get("scene_id"):
        raise HTTPException(status_code=404, detail="Processed video asset not found")
    parent = project_resource_access.project_scene_or_404(
        project_id,
        str(asset["scene_id"]),
    )
    video = parent.get("payload", {}).get("videoAsset") or {}
    segment = next(
        (item for item in video.get("segments", []) if item.get("id") == segment_id),
        None,
    )
    if segment is None:
        raise HTTPException(status_code=404, detail="Video segment not found")

    scene = materialize_segment_scene(parent, segment)
    scenes.put(parent)
    return scene


@router.post("/{asset_id}/segment-layout/propose", response_model=SceneDocument)
def propose_video_segment_layout(project_id: str, asset_id: str):
    asset = project_resource_access.project_video_or_404(project_id, asset_id)
    if not asset.get("scene_id"):
        raise HTTPException(status_code=404, detail="Processed video asset not found")
    parent = project_resource_access.project_scene_or_404(
        project_id,
        str(asset["scene_id"]),
    )
    video = parent.get("payload", {}).get("videoAsset") or {}
    segments = video.get("segments") or []
    if not segments:
        raise HTTPException(status_code=409, detail="No continuous shots were detected")
    try:
        source = published_video_directory(asset) / "proxy.mp4"
    except VideoProcessingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not source.exists():
        raise HTTPException(
            status_code=409,
            detail="Published browser proxy is unavailable",
        )
    video["segmentLayout"] = propose_segment_layout(
        source,
        segments,
        float(parent["duration"]),
    )
    return scenes.put(parent)
