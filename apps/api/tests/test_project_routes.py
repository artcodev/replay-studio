from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.project_routes as project_routes
import app.project_analysis_routes as project_analysis_routes
import app.project_core_routes as project_core_routes
import app.project_match_routes as project_match_routes
import app.project_media_routes as project_media_routes
from app.analysis_cancellation import AnalysisCancellationService
from app.analysis_run_repository import AnalysisRunRepository
from app.database import Base, VideoAssetRow
from app.external_reference_repository import ExternalReferenceRepository
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import ProjectResourceRepository
from app.analysis_run_contract import AnalysisRunCreate
from app.project_lifecycle_contract import ProjectCreate
from app.project_match_persistence_contract import MatchSnapshotCreate, MatchUpsert
from app.project_segment_contract import SegmentUpsert
from app.project_store import ProjectStore
from app.sample import make_video_scene
from app.match_contracts import EventBundle, ExternalEvent, ExternalPlayer, ExternalTeam
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


def _bundle() -> EventBundle:
    return EventBundle(
        source="private-provider",
        event=ExternalEvent(
            id="upstream-99",
            provider="private-provider",
            name="Spain vs Belgium",
            date="2026-07-10",
            time="19:00",
            league="World Cup",
            season="2026",
            home=ExternalTeam(id="upstream-home", name="Spain"),
            away=ExternalTeam(id="upstream-away", name="Belgium"),
            home_score=2,
            away_score=1,
        ),
        players=[
            ExternalPlayer(
                id="upstream-player",
                name="Player Eight",
                team_id="upstream-home",
                position="G",
                number="8",
                lineup_role="starter",
            )
        ],
        fetched_at="2026-07-17T12:00:00+00:00",
    )


def test_public_project_match_flow_never_exposes_provider_ids(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    matches = ProjectMatchRepository(sessions)
    references = ExternalReferenceRepository(sessions)
    monkeypatch.setattr(project_core_routes, "project_store", store)
    monkeypatch.setattr(project_match_routes, "project_store", store)
    monkeypatch.setattr(project_match_routes, "project_matches", matches)
    monkeypatch.setattr(project_match_routes, "external_references", references)

    async def event_bundle_for(provider: str, event_id: str) -> EventBundle:
        assert provider == "private-provider"
        assert event_id == "upstream-99"
        return _bundle()

    monkeypatch.setattr(
        project_match_routes.sports_provider,
        "event_bundle_for",
        event_bundle_for,
    )
    app = FastAPI()
    app.include_router(project_routes.router)
    created = _request(app, "POST", "/api/projects", json={"title": "World Cup moment"})
    assert created.status_code == 201
    project = created.json()
    assert set(project) == {
        "id",
        "title",
        "revision",
        "matchId",
        "activeSegmentId",
        "createdAt",
        "updatedAt",
    }

    candidate = project_match_routes.remember_match_candidates(
        [_bundle().event],
        provider="private-provider",
        references=references,
    )[0]
    selected = _request(
        app,
        "PUT",
        f"/api/projects/{project['id']}/match",
        json={"matchId": candidate.id},
    )
    assert selected.status_code == 200
    match = selected.json()
    serialized = str(match)
    assert match["homeTeam"]["name"] == "Spain"
    assert match["awayTeam"]["name"] == "Belgium"
    assert match["roster"][0]["name"] == "Player Eight"
    assert match["roster"][0]["goalkeeper"] is True
    assert match["sync"]["state"] == "synced"
    assert "private-provider" not in serialized
    assert "upstream-99" not in serialized
    assert "upstream-player" not in serialized

    loaded = _request(app, "GET", f"/api/projects/{project['id']}/match")
    assert loaded.status_code == 200
    assert loaded.json() == match
    diagnostics = _request(
        app,
        "GET",
        f"/api/projects/{project['id']}/integration-diagnostics"
    )
    assert diagnostics.status_code == 200
    assert diagnostics.json()["currentMatchSnapshot"]["provider"] == "private-provider"

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_project_assets_expose_their_own_timeline_scene_for_multi_video_routing(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    with sessions.begin() as session:
        session.add_all(
            [
                VideoAssetRow(
                    id="asset-angle-a",
                    filename="source-a.mp4",
                    original_name="angle-a.mp4",
                    content_type="video/mp4",
                    status="ready",
                    stage="Ready",
                    progress=100,
                    duration=54.5,
                    scene_id="timeline-angle-a",
                ),
                VideoAssetRow(
                    id="asset-angle-b",
                    filename="source-b.mp4",
                    original_name="angle-b.mp4",
                    content_type="video/mp4",
                    status="queued",
                    stage="Waiting for FFmpeg",
                    progress=2,
                    scene_id=None,
                ),
            ]
        )
    store.create_project(ProjectCreate(id="project-angles", title="Two angles"))
    resources.link_video_asset("project-angles", "asset-angle-a")
    resources.link_video_asset("project-angles", "asset-angle-b")

    assets = {
        "asset-angle-a": {
            "id": "asset-angle-a",
            "original_name": "angle-a.mp4",
            "status": "ready",
            "duration": 54.5,
            "scene_id": "timeline-angle-a",
            "media_url": None,
            "poster_url": None,
            "created_at": "2026-07-18T08:00:00+00:00",
        },
        "asset-angle-b": {
            "id": "asset-angle-b",
            "original_name": "angle-b.mp4",
            "status": "queued",
            "duration": None,
            "scene_id": None,
            "media_url": None,
            "poster_url": None,
            "created_at": "2026-07-18T08:01:00+00:00",
        },
    }
    monkeypatch.setattr(project_media_routes, "project_store", store)
    monkeypatch.setattr(project_media_routes, "project_resources", resources)
    monkeypatch.setattr(
        project_media_routes,
        "video_store",
        SimpleNamespace(
            get=lambda asset_id: assets.get(asset_id),
            list_by_ids=lambda asset_ids: [
                assets[asset_id]
                for asset_id in asset_ids
                if asset_id in assets
            ],
        ),
    )
    app = FastAPI()
    app.include_router(project_routes.router)
    response = _request(app, "GET", "/api/projects/project-angles/assets")

    assert response.status_code == 200
    by_id = {asset["id"]: asset for asset in response.json()}
    assert by_id["asset-angle-a"]["timelineSceneId"] == "timeline-angle-a"
    assert by_id["asset-angle-b"]["timelineSceneId"] is None
    assert by_id["asset-angle-b"]["status"] == "uploading"
    assert "provider" not in str(response.json()).lower()

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_project_navigation_reads_compact_scene_metadata_only(monkeypatch) -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    scenes = SceneRepository(sessions)
    store.create_project(ProjectCreate(id="project-nav", title="Navigation"))
    for index in range(2):
        scene_id = f"scene-{index}"
        scene = make_video_scene(
            scene_id,
            f"Moment {index}",
            5.0,
            {
                "id": "asset-nav",
                "filename": "match.mp4",
                "selectedSegmentId": f"shot-{index}",
            },
        )
        scenes.put(scene)
        resources.link_scene("project-nav", scene_id, role="segment")
        resources.upsert_segment(
            "project-nav",
            SegmentUpsert(
                id=f"segment-{index}",
                scene_id=scene_id,
                source_segment_id=f"shot-{index}",
                label=f"Moment {index}",
                start_seconds=float(index * 5),
                end_seconds=float(index * 5 + 5),
                ordinal=index,
            ),
        )
    scenes.put(
        make_video_scene(
            "other-project-scene",
            "Must not be scanned",
            9.0,
            {"id": "asset-other", "filename": "other.mp4"},
        )
    )

    monkeypatch.setattr(project_core_routes, "project_store", store)
    monkeypatch.setattr(project_media_routes, "project_store", store)
    monkeypatch.setattr(project_media_routes, "project_resources", resources)
    monkeypatch.setattr(project_media_routes, "scenes", scenes)
    monkeypatch.setattr(
        project_media_routes,
        "reconstruction_jobs",
        SimpleNamespace(statuses=lambda _scene_ids: {}),
    )
    app = FastAPI()
    app.include_router(project_routes.router)
    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    project_response = _request(app, "GET", "/api/projects")
    segment_response = _request(app, "GET", "/api/projects/project-nav/segments")
    scene_response = _request(app, "GET", "/api/projects/project-nav/scenes")
    event.remove(engine, "before_cursor_execute", capture_statement)

    assert project_response.status_code == 200
    assert [item["id"] for item in project_response.json()] == ["project-nav"]
    assert segment_response.status_code == 200
    assert scene_response.status_code == 200
    assert len(segment_response.json()) == 2
    assert len(scene_response.json()) == 2
    assert not any("scenes.payload" in statement for statement in statements)
    scene_queries = [statement for statement in statements if " from scenes" in statement]
    assert len(scene_queries) == 1
    assert "scenes.duration" in scene_queries[0]
    assert "scenes.kind" in scene_queries[0]
    assert "where scenes.id in" in scene_queries[0]
    project_queries = [statement for statement in statements if " from projects" in statement]
    assert len(project_queries) == 3
    assert all("scenes.payload" not in statement for statement in project_queries)

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_project_header_and_analysis_existence_never_hydrate_project_graph(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = ProjectStore(sessions)
    resources = ProjectResourceRepository(sessions)
    matches = ProjectMatchRepository(sessions)
    runs = AnalysisRunRepository(sessions)
    cancellation = AnalysisCancellationService(sessions)
    store.create_project(ProjectCreate(id="project-compact", title="Compact"))
    matches.publish(
        "project-compact",
        MatchUpsert(id="match-compact", name="Compact match"),
        MatchSnapshotCreate(
            provider="private-provider",
            external_event_id="fixture-1",
            payload={"schemaVersion": 1, "largeRoster": [1, 2, 3]},
        ),
    )
    resources.upsert_segment(
        "project-compact",
        SegmentUpsert(
            id="segment-dense",
            source_segment_id="shot-01",
            label="1-A",
            start_seconds=0.0,
            end_seconds=8.0,
            payload={"dense": [1, 2, 3]},
        ),
    )
    runs.create(
        "project-compact",
        AnalysisRunCreate(
            id="analysis-compact",
            kind="video-processing",
            status="queued",
            progress={"phase": "queued", "overallPercent": 0},
        ),
    )
    monkeypatch.setattr(project_core_routes, "project_store", store)
    monkeypatch.setattr(project_analysis_routes, "analysis_runs", runs)
    monkeypatch.setattr(
        project_analysis_routes,
        "analysis_cancellation",
        cancellation,
    )
    monkeypatch.setattr(
        project_analysis_routes,
        "pipeline_terminals",
        SimpleNamespace(cancel=lambda _run_id: None),
    )
    app = FastAPI()
    app.include_router(project_routes.router)

    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    project_response = _request(app, "GET", "/api/projects/project-compact")
    project_statements = list(statements)
    statements.clear()
    analysis_response = _request(
        app,
        "GET",
        "/api/projects/project-compact/analysis-runs",
    )
    analysis_statements = list(statements)
    statements.clear()
    missing_response = _request(
        app,
        "GET",
        "/api/projects/missing/analysis-runs",
    )
    missing_statements = list(statements)
    statements.clear()
    cancel_response = _request(
        app,
        "POST",
        "/api/projects/project-compact/analysis-runs/analysis-compact/cancel",
    )
    cancel_statements = list(statements)
    event.remove(engine, "before_cursor_execute", capture_statement)

    assert project_response.status_code == 200
    assert project_response.json()["id"] == "project-compact"
    assert analysis_response.status_code == 200
    assert [row["id"] for row in analysis_response.json()] == ["analysis-compact"]
    assert missing_response.status_code == 404
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"

    assert not any("match_snapshots" in query for query in project_statements)
    assert not any(" from segments" in query for query in project_statements)
    assert not any(" from analysis_runs" in query for query in project_statements)

    assert sum(" from analysis_runs" in query for query in analysis_statements) == 1
    assert not any("match_snapshots.payload" in query for query in analysis_statements)
    assert not any(" from segments" in query for query in analysis_statements)
    assert not any(" from analysis_runs" in query for query in missing_statements)
    assert not any("match_snapshots.payload" in query for query in missing_statements)
    assert not any(" from segments" in query for query in missing_statements)
    assert not any("match_snapshots.payload" in query for query in cancel_statements)
    assert not any(" from segments" in query for query in cancel_statements)

    Base.metadata.drop_all(engine)
    engine.dispose()
