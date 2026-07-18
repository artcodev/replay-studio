from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.identity_decision_routes as identity_decision_routes
import app.identity_review_routes as identity_review_routes
import app.main as main
import app.project_analysis_routes as project_analysis_routes
import app.project_core_routes as project_core_routes
import app.project_media_routes as project_media_routes
import app.project_routes as project_routes
import app.project_resource_access as resource_access
from app.analysis_cancellation import AnalysisCancellationService
from app.analysis_run_repository import AnalysisRunRepository
from app.database import Base, VideoAssetRow
from app.analysis_run_contract import AnalysisRunCreate
from app.project_lifecycle_contract import ProjectCreate
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.sample import make_video_scene
from app.scene_repository import SceneRepository


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


def _workspace(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    projects = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    matches = ProjectMatchRepository(sessions)
    runs = AnalysisRunRepository(sessions)
    cancellation = AnalysisCancellationService(sessions)
    scenes = SceneRepository(sessions)
    projects.create_project(ProjectCreate(id="project-a", title="A"))
    projects.create_project(ProjectCreate(id="project-b", title="B"))

    scene = make_video_scene(
        "scene-a",
        "1-A",
        5.0,
        {
            "id": "asset-a",
            "filename": "source.mp4",
            "generationKey": "generation-a",
            "selectedSegmentId": "shot-01",
            "reconstruction": {"status": "ready", "model": "yolo26m.pt"},
        },
    )
    scenes.put(scene)
    with sessions.begin() as session:
        session.add(
            VideoAssetRow(
                id="asset-a",
                filename="source.mp4",
                original_name="source.mp4",
                content_type="video/mp4",
                status="ready",
                stage="Ready",
                progress=100,
                generation_key="generation-a",
                scene_id="scene-a",
            )
        )
    resources.link_scene("project-a", "scene-a", role="segment")
    resources.link_video_asset("project-a", "asset-a")

    asset = {
        "id": "asset-a",
        "filename": "source.mp4",
        "original_name": "source.mp4",
        "content_type": "video/mp4",
        "status": "ready",
        "stage": "Ready",
        "progress": 100,
        "duration": 5.0,
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "frame_count": 50,
        "generation_key": "generation-a",
        "scene_id": "scene-a",
        "media_url": None,
        "poster_url": None,
        "error": None,
        "created_at": "2026-07-18T12:00:00+00:00",
    }
    videos = SimpleNamespace(
        get=lambda asset_id: asset if asset_id == "asset-a" else None,
        list_by_ids=lambda asset_ids: [asset] if "asset-a" in asset_ids else [],
    )

    monkeypatch.setattr(resource_access, "project_resources", resources)
    monkeypatch.setattr(resource_access, "project_matches", matches)
    monkeypatch.setattr(resource_access, "scenes", scenes)
    monkeypatch.setattr(resource_access, "video_store", videos)
    monkeypatch.setattr(project_core_routes, "project_store", projects)
    monkeypatch.setattr(project_media_routes, "project_store", projects)
    monkeypatch.setattr(project_media_routes, "project_resources", resources)
    monkeypatch.setattr(project_media_routes, "scenes", scenes)
    monkeypatch.setattr(project_media_routes, "video_store", videos)
    monkeypatch.setattr(project_analysis_routes, "analysis_runs", runs)
    monkeypatch.setattr(
        project_analysis_routes,
        "analysis_cancellation",
        cancellation,
    )
    monkeypatch.setattr(identity_review_routes, "project_resources", resources)
    monkeypatch.setattr(identity_review_routes, "scenes", scenes)
    monkeypatch.setattr(identity_decision_routes, "project_resources", resources)
    monkeypatch.setattr(identity_decision_routes, "scenes", scenes)
    return engine, projects, runs


def test_scene_identity_series_and_media_are_project_owned(monkeypatch) -> None:
    engine, _projects, _runs = _workspace(monkeypatch)
    owned_scene = _request("GET", "/api/projects/project-a/scenes/scene-a")
    foreign_scene = _request("GET", "/api/projects/project-b/scenes/scene-a")
    foreign_reconstruction = _request(
        "POST",
        "/api/projects/project-b/scenes/scene-a/reconstruct"
    )
    foreign_series = _request(
        "GET",
        "/api/projects/project-b/scenes/scene-a/reconstruction-series?start=0&end=1"
    )
    foreign_identity = _request(
        "GET",
        "/api/projects/project-b/scenes/scene-a/identity-review"
    )
    owned_video = _request("GET", "/api/projects/project-a/videos/asset-a")
    foreign_video = _request("GET", "/api/projects/project-b/videos/asset-a")

    assert owned_scene.status_code == 200
    assert foreign_scene.status_code == 404
    assert foreign_reconstruction.status_code == 404
    assert foreign_series.status_code == 404
    assert foreign_identity.status_code == 404
    assert owned_video.status_code == 200
    assert owned_video.json()["media_url"] == (
        "/api/projects/project-a/videos/asset-a/media"
    )
    assert owned_video.json()["poster_url"] == (
        "/api/projects/project-a/videos/asset-a/poster"
    )
    assert foreign_video.status_code == 404

    # Removed global resources cannot bypass project ownership.
    assert _request("GET", "/api/scenes/scene-a").status_code == 404
    assert _request("GET", "/api/videos/asset-a").status_code == 404

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_analysis_cancel_cannot_cross_project_boundary(monkeypatch) -> None:
    engine, _projects, runs = _workspace(monkeypatch)
    runs.create(
        "project-a",
        AnalysisRunCreate(
            id="run-a",
            kind="calibration",
            status="queued",
            progress={"phase": "queued", "label": "Waiting"},
        ),
    )
    pipeline = SimpleNamespace(cancel=lambda _run_id: None)
    monkeypatch.setattr(project_analysis_routes, "pipeline_terminals", pipeline)
    foreign = _request(
        "POST",
        "/api/projects/project-b/analysis-runs/run-a/cancel"
    )
    owned = _request(
        "POST",
        "/api/projects/project-a/analysis-runs/run-a/cancel"
    )

    assert foreign.status_code == 404
    assert owned.status_code == 200
    assert owned.json()["status"] == "cancelled"
    assert _request("POST", "/api/analysis-runs/run-a/cancel").status_code == 404

    Base.metadata.drop_all(engine)
    engine.dispose()
