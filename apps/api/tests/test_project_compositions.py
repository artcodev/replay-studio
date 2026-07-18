from __future__ import annotations

import asyncio
import httpx
from fastapi import FastAPI
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.multi_pass_composition as multi_pass_composition
import app.project_routes as project_routes
import app.project_analysis_routes as project_analysis_routes
import app.project_media_routes as project_media_routes
from app.analysis_run_repository import AnalysisRunRepository
from app.database import Base, VideoAssetRow
from app.pipeline_terminal_service import PipelineTerminalService
from app.multi_pass_domain import MultiPassError
from app.multi_pass_pipeline_service import MultiPassPipelineService
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import ProjectResourceRepository
from app.project_lifecycle_contract import ProjectCreate
from app.project_segment_contract import SegmentUpsert
from app.project_store import ProjectStore
from app.sample import make_video_scene
from app.scene_repository import SceneRepository


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    return asyncio.run(_async_request(application, method, path, **kwargs))


def _source_scene(scene_id: str, asset_id: str, label: str) -> dict:
    return make_video_scene(
        scene_id=scene_id,
        title=label,
        duration=8.0,
        video_asset={
            "id": asset_id,
            "filename": f"{asset_id}.mp4",
            "fps": 25.0,
            "analysisFps": 10.0,
            "frameCount": 80,
            "processingState": "frames-ready",
            "sourceStart": 2.0,
            "sourceEnd": 10.0,
            # Both assets deliberately use the same detector-local shot id.
            # The composition must retain the distinct canonical segment ids.
            "selectedSegmentId": "shot-01",
            "reconstruction": {"status": "ready", "model": "yolo26m.pt"},
        },
    )


def _cross_asset_project(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    runs = AnalysisRunRepository(sessions)
    scenes = SceneRepository(sessions)

    with sessions.begin() as session:
        for asset_id in ("asset-a", "asset-b"):
            session.add(
                VideoAssetRow(
                    id=asset_id,
                    filename="source.mp4",
                    original_name=f"{asset_id}.mp4",
                    content_type="video/mp4",
                    status="ready",
                    stage="Ready",
                    progress=100,
                    frame_count=80,
                )
            )

    store.create_project(ProjectCreate(id="project-angles", title="Two broadcasts"))
    source_a = scenes.put(_source_scene("scene-angle-a", "asset-a", "Angle A"))
    source_b = scenes.put(_source_scene("scene-angle-b", "asset-b", "Angle B"))
    for source, asset_id in ((source_a, "asset-a"), (source_b, "asset-b")):
        resources.link_video_asset("project-angles", asset_id)
        resources.link_scene("project-angles", source["id"], role="segment")

    segment_a, _ = resources.upsert_segment(
        "project-angles",
        SegmentUpsert(
            id="segment-angle-a",
            video_asset_id="asset-a",
            scene_id=source_a["id"],
            source_segment_id="shot-01",
            label="1-A",
            start_seconds=2.0,
            end_seconds=10.0,
            payload={"score": 0.91},
        ),
    )
    segment_b, _ = resources.upsert_segment(
        "project-angles",
        SegmentUpsert(
            id="segment-angle-b",
            video_asset_id="asset-b",
            scene_id=source_b["id"],
            source_segment_id="shot-01",
            label="1-B",
            start_seconds=14.0,
            end_seconds=22.0,
            payload={"score": 0.88},
        ),
    )

    monkeypatch.setattr(
        multi_pass_composition,
        "project_resources",
        resources,
    )
    monkeypatch.setattr(
        multi_pass_composition,
        "project_matches",
        ProjectMatchRepository(sessions),
    )
    monkeypatch.setattr(multi_pass_composition, "scenes", scenes)
    monkeypatch.setattr(project_media_routes, "project_store", store)
    monkeypatch.setattr(project_media_routes, "project_resources", resources)
    monkeypatch.setattr(project_media_routes, "scenes", scenes)
    monkeypatch.setattr(project_analysis_routes, "analysis_runs", runs)
    monkeypatch.setattr(
        multi_pass_composition,
        "multi_pass_pipeline",
        MultiPassPipelineService(sessions),
    )
    monkeypatch.setattr(
        project_analysis_routes,
        "pipeline_terminals",
        PipelineTerminalService(sessions),
    )
    return (
        engine,
        store,
        resources,
        runs,
        scenes,
        source_a,
        source_b,
        segment_a,
        segment_b,
    )


def _source_pass(segment, source: dict) -> dict:
    return {
        "id": segment.id,
        "segmentId": segment.id,
        "sourceSegmentId": segment.source_segment_id,
        "sceneId": source["id"],
        "assetId": segment.video_asset_id,
        "label": segment.label,
        "start": segment.start_seconds,
        "end": segment.end_seconds,
        "duration": segment.end_seconds - segment.start_seconds,
        "score": segment.payload.get("score", 0.0),
    }


def test_create_project_multi_pass_scene_keeps_cross_asset_sources_distinct(
    monkeypatch,
) -> None:
    engine, _store, resources, runs, _scenes, source_a, source_b, segment_a, segment_b = (
        _cross_asset_project(monkeypatch)
    )
    passes = [_source_pass(segment_a, source_a), _source_pass(segment_b, source_b)]

    composition = multi_pass_composition.create_project_multi_pass_scene(
        source_a["id"],
        passes,
        project_id="project-angles",
        title="Same moment, two assets",
        source_scenes={source_a["id"]: source_a, source_b["id"]: source_b},
    )

    video = composition["payload"]["videoAsset"]
    assert video["multiPass"]["selectedSegmentIds"] == [
        "segment-angle-a",
        "segment-angle-b",
    ]
    assert [item["sourceSegmentId"] for item in video["multiPass"]["sourcePasses"]] == [
        "shot-01",
        "shot-01",
    ]
    assert [item["assetId"] for item in video["multiPass"]["sourcePasses"]] == [
        "asset-a",
        "asset-b",
    ]
    assert resources.scene_owner(composition["id"]) == "project-angles"
    run = runs.get(video["reconstruction"]["runId"])
    assert run is not None
    assert run.kind == "multi-pass"
    assert run.scene_id == composition["id"]
    assert run.input_fingerprint == video["reconstruction"]["inputFingerprint"]
    assert run.input_fingerprint.startswith("sha256:")

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_multi_pass_requires_explicit_project_and_rejects_cross_project_sources(
    monkeypatch,
) -> None:
    engine, store, resources, runs, scenes, source_a, source_b, segment_a, segment_b = (
        _cross_asset_project(monkeypatch)
    )
    passes = [_source_pass(segment_a, source_a), _source_pass(segment_b, source_b)]

    with pytest.raises(TypeError):
        multi_pass_composition.create_project_multi_pass_scene(
            source_a["id"],
            passes,
            source_scenes={source_a["id"]: source_a, source_b["id"]: source_b},
        )
    with pytest.raises(MultiPassError, match="Project missing-project"):
        multi_pass_composition.create_project_multi_pass_scene(
            source_a["id"],
            passes,
            project_id="missing-project",
            source_scenes={source_a["id"]: source_a, source_b["id"]: source_b},
        )

    store.create_project(ProjectCreate(id="project-other", title="Other match"))
    other = scenes.put(_source_scene("scene-other", "asset-other", "Other match"))
    resources.link_scene("project-other", other["id"], role="segment")
    cross_project_passes = [
        _source_pass(segment_a, source_a),
        {
            "id": "segment-other",
            "segmentId": "segment-other",
            "sourceSegmentId": "shot-01",
            "sceneId": other["id"],
            "assetId": "asset-other",
            "label": "Other",
            "start": 2.0,
            "end": 10.0,
            "duration": 8.0,
            "score": 0.99,
        },
    ]
    with pytest.raises(
        MultiPassError,
        match="missing or owned by another project: scene-other",
    ):
        multi_pass_composition.create_project_multi_pass_scene(
            source_a["id"],
            cross_project_passes,
            project_id="project-angles",
            source_scenes={source_a["id"]: source_a, other["id"]: other},
        )

    assert runs.list_for_project("project-angles") == []
    assert runs.list_for_project("project-other") == []
    assert {link.scene_id for link in resources.list_scene_links("project-angles")} == {
        source_a["id"],
        source_b["id"],
    }

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_compositions_route_creates_cross_asset_job(monkeypatch) -> None:
    engine, _store, resources, _runs, _scenes, _source_a, _source_b, _segment_a, _segment_b = (
        _cross_asset_project(monkeypatch)
    )
    app = FastAPI()
    app.include_router(project_routes.router)
    response = _request(
        app,
        "POST",
        "/api/projects/project-angles/compositions",
        json={
            "segmentIds": ["segment-angle-a", "segment-angle-b"],
            "title": "API composition",
        },
    )

    assert response.status_code == 202
    composition = response.json()
    source_passes = composition["payload"]["videoAsset"]["multiPass"]["sourcePasses"]
    assert {item["assetId"] for item in source_passes} == {"asset-a", "asset-b"}
    assert {item["segmentId"] for item in source_passes} == {
        "segment-angle-a",
        "segment-angle-b",
    }
    jobs = _request(app, "GET", "/api/projects/project-angles/analysis-runs")
    assert jobs.status_code == 200
    assert [(item["kind"], item["status"]) for item in jobs.json()] == [
        ("multi-pass", "queued")
    ]
    assert resources.scene_owner(composition["id"]) == "project-angles"

    Base.metadata.drop_all(engine)
    engine.dispose()
