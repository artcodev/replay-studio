from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType

import pytest
from sqlalchemy import Index, MetaData, Table, create_engine, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.analysis_cancellation import AnalysisCancellationService
from app.analysis_run_repository import AnalysisRunConflict, AnalysisRunRepository
from app.database import Base, SceneRow, VideoAssetRow
from app.database import ReconstructionLeaseRow
from app.external_reference_repository import ExternalReferenceRepository
from app.project_integration_queries import ProjectIntegrationDiagnosticsQuery
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceRepository,
)
from app.analysis_run_contract import AnalysisRunCreate, AnalysisRunUpdate
from app.project_lifecycle_contract import ProjectCreate, ProjectUpdate
from app.project_match_persistence_contract import MatchSnapshotCreate, MatchUpsert
from app.project_segment_contract import SegmentUpsert
from app.project_store import ProjectConflict, ProjectStore
from app.project_models import MatchSnapshotRow, ProjectSceneRow


@pytest.fixture
def persistence():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    yield ProjectStore(sessions), ProjectResourceRepository(sessions), sessions
    Base.metadata.drop_all(engine)
    engine.dispose()


def _scene(scene_id: str, *, asset_id: str, video: dict | None = None) -> dict:
    video_asset = {"id": asset_id, **(video or {})}
    payload = {
        "id": scene_id,
        "title": scene_id,
        "version": 1,
        "revision": 0,
        "duration": 8.0,
        "payload": {
            "videoAsset": video_asset,
            "teams": [],
            "tracks": [],
        },
    }
    return payload


def _asset(asset_id: str, scene_id: str | None = None) -> VideoAssetRow:
    return VideoAssetRow(
        id=asset_id,
        filename="source.mp4",
        original_name=f"{asset_id}.mp4",
        content_type="video/mp4",
        status="ready",
        stage="Ready",
        progress=100,
        frame_count=80,
        scene_id=scene_id,
    )


def _concurrent_store(
    tmp_path: Path,
) -> tuple[ProjectStore, ProjectResourceRepository, sessionmaker, object]:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'concurrency.sqlite'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    return ProjectStore(sessions), ProjectResourceRepository(sessions), sessions, engine


def test_project_store_has_no_analysis_run_write_authority() -> None:
    obsolete_methods = {
        "create_analysis_run",
        "get_analysis_run",
        "list_analysis_runs",
        "request_analysis_run_cancel",
        "update_analysis_progress",
        "update_analysis_run",
        "update_analysis_status",
    }
    assert obsolete_methods.isdisjoint(vars(ProjectStore))


def test_project_store_has_no_match_or_external_reference_authority() -> None:
    obsolete_methods = {
        "upsert_match",
        "record_match_snapshot",
        "get_integration_diagnostics",
        "current_snapshot_payload",
        "current_match_snapshot",
        "current_match_snapshot_summary",
        "current_match_snapshot_source",
        "match_snapshot",
        "find_external_reference",
        "upsert_external_reference",
        "external_references_for_resource",
    }
    assert obsolete_methods.isdisjoint(vars(ProjectStore))


def test_project_store_has_no_resource_ownership_authority() -> None:
    obsolete_methods = {
        "get_project_graph",
        "link_scene",
        "link_scenes",
        "link_video_asset",
        "list_scene_links",
        "list_video_asset_links",
        "list_segments",
        "upsert_segment",
        "project_id_for_scene",
        "project_ids_for_scene",
        "project_id_for_video_asset",
        "project_ids_for_video_asset",
    }
    assert obsolete_methods.isdisjoint(vars(ProjectStore))


def test_project_revision_compare_and_swap_serializes_concurrent_writers(
    tmp_path: Path,
) -> None:
    store, _, _, engine = _concurrent_store(tmp_path)
    store.create_project(ProjectCreate(id="project-cas", title="Original"))
    barrier = Barrier(2)

    def update(title: str) -> str:
        barrier.wait()
        try:
            store.update_project(
                "project-cas",
                ProjectUpdate(title=title, expected_revision=1),
            )
            return "saved"
        except ProjectConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, ["Writer A", "Writer B"]))

    assert sorted(results) == ["conflict", "saved"]
    project = store.get_project("project-cas")
    assert project is not None
    assert project.revision == 2
    assert project.title in {"Writer A", "Writer B"}
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_project_resource_ownership_and_current_snapshot_are_race_safe(
    tmp_path: Path,
) -> None:
    store, resources, sessions, engine = _concurrent_store(tmp_path)
    matches = ProjectMatchRepository(sessions)
    with sessions.begin() as session:
        scene = _scene("scene-race", asset_id="asset-race")
        session.add(SceneRow(id="scene-race", title="Race", payload=scene))
    store.create_project(ProjectCreate(id="project-a", title="A"))
    store.create_project(ProjectCreate(id="project-b", title="B"))
    barrier = Barrier(2)

    def claim(project_id: str) -> str:
        barrier.wait()
        try:
            resources.link_scene(project_id, "scene-race")
            return project_id
        except ProjectResourceConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, ["project-a", "project-b"]))

    assert claims.count("conflict") == 1
    with sessions() as session:
        owners = session.scalars(
            select(ProjectSceneRow.project_id).where(
                ProjectSceneRow.scene_id == "scene-race"
            )
        ).all()
    assert owners == [next(value for value in claims if value != "conflict")]

    owner = owners[0]
    snapshot_barrier = Barrier(2)

    def publish_snapshot(value: int) -> str:
        snapshot_barrier.wait()
        publication = matches.publish(
            owner,
            MatchUpsert(id="match-race", name="A v B"),
            MatchSnapshotCreate(
                provider="test",
                external_event_id=f"event-{value}",
                payload={"schemaVersion": 1, "value": value},
            ),
        )
        return publication.snapshot.id

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_ids = list(executor.map(publish_snapshot, [1, 2]))

    with sessions() as session:
        current = session.scalars(
            select(MatchSnapshotRow).where(
                MatchSnapshotRow.project_id == owner,
                MatchSnapshotRow.is_current.is_(True),
            )
        ).all()
    assert len(current) == 1
    assert current[0].id in snapshot_ids
    project = store.get_project(owner)
    assert project is not None
    assert project.current_match_snapshot_id == current[0].id
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_resource_repository_owns_resources_and_project_hides_provenance(persistence) -> None:
    store, resources, sessions = persistence
    matches = ProjectMatchRepository(sessions)
    references = ExternalReferenceRepository(sessions)
    runs = AnalysisRunRepository(sessions)
    cancellation = AnalysisCancellationService(sessions)
    with sessions.begin() as session:
        scene = _scene("scene-1", asset_id="asset-1")
        session.add(SceneRow(id="scene-1", title="Scene 1", payload=scene))
        session.add(_asset("asset-1", "scene-1"))

    project = store.create_project(
        ProjectCreate(id="project-1", title="Spain vs Belgium")
    )
    assert project.revision == 1
    assert resources.link_scene("project-1", "scene-1", role="root") is True
    assert resources.link_video_asset("project-1", "asset-1") is True
    assert resources.scene_owner("scene-1") == "project-1"
    assert resources.video_asset_owner("asset-1") == "project-1"

    store.create_project(ProjectCreate(id="project-2", title="Other"))
    with pytest.raises(ProjectResourceConflict, match="already belongs"):
        resources.link_scene("project-2", "scene-1")
    with pytest.raises(ProjectResourceConflict, match="already belongs"):
        resources.link_video_asset("project-2", "asset-1")

    publication = matches.publish(
        "project-1",
        MatchUpsert(
            id="match-canonical",
            name="Spain vs Belgium",
            home_team_name="Spain",
            away_team_name="Belgium",
            metadata={
                "homeScore": 2,
                "awayScore": 1,
                "provider": "must-not-leak",
                "externalEventId": "fixture-secret",
            },
        ),
        MatchSnapshotCreate(
            provider="api-football",
            external_event_id="fixture-42",
            schema_version=1,
            payload={"schemaVersion": 1, "matchId": "match-canonical"},
        ),
    )
    assert publication.match_created is True
    assert publication.snapshot_created is True
    assert publication.match.metadata == {"homeScore": 2, "awayScore": 1}
    snapshot = publication.snapshot
    assert matches.current_payload("project-1") == {
        "schemaVersion": 1,
        "matchId": "match-canonical",
    }

    public = store.get_project("project-1")
    assert public is not None
    public_json = public.model_dump(by_alias=True)
    assert "currentMatchSnapshot" not in public_json
    assert "externalReferences" not in public_json
    summary_json = matches.current_summary("project-1").model_dump(by_alias=True)
    assert summary_json["id"] == snapshot.id
    assert "provider" not in summary_json
    assert "externalEventId" not in summary_json

    diagnostics = ProjectIntegrationDiagnosticsQuery(
        store,
        matches,
        references,
    ).get("project-1")
    assert diagnostics is not None
    assert diagnostics.current_match_snapshot is not None
    assert diagnostics.current_match_snapshot.provider == "api-football"
    assert diagnostics.current_match_snapshot.external_event_id == "fixture-42"

    queued, _ = runs.create(
        "project-1",
        AnalysisRunCreate(
            id="run-queued",
            scene_id="scene-1",
            kind="reconstruction",
            progress={"phase": "waiting", "overallPercent": 0, "largePayload": [1, 2, 3]},
        ),
    )
    assert queued.progress == {"phase": "waiting", "overallPercent": 0}
    assert cancellation.cancel("run-queued").status == "cancelled"

    running, _ = runs.create(
        "project-1",
        AnalysisRunCreate(
            id="run-running",
            scene_id="scene-1",
            kind="reconstruction",
            status="running",
        ),
    )
    assert running.status == "running"
    assert cancellation.cancel("run-running").status == "cancelled"
    with pytest.raises(AnalysisRunConflict):
        runs.update(
            "run-running",
            AnalysisRunUpdate(status="running"),
        )

    current = store.get_project("project-1")
    assert current is not None
    updated = store.update_project(
        "project-1",
        ProjectUpdate(title="World Cup replay", expected_revision=current.revision),
    )
    assert updated.title == "World Cup replay"
    assert updated.revision == current.revision + 1
    archived = store.archive_project("project-1", expected_revision=updated.revision)
    assert archived.status == "archived"


def test_resource_batch_link_is_atomic_and_reconstruction_context_is_exact(
    persistence,
) -> None:
    store, resources, sessions = persistence
    with sessions.begin() as session:
        session.add(
            SceneRow(
                id="scene-context",
                title="Context",
                payload=_scene("scene-context", asset_id="asset-context"),
            )
        )
    store.create_project(ProjectCreate(id="project-context", title="Context"))

    with pytest.raises(ProjectResourceConflict, match="Scene missing-scene"):
        resources.link_scenes(
            "project-context",
            [
                ("scene-context", "segment"),
                ("missing-scene", "segment"),
            ],
        )
    assert resources.scene_owner("scene-context") is None

    assert resources.link_scenes(
        "project-context",
        [("scene-context", "segment")],
    ) == 1
    resources.upsert_segment(
        "project-context",
        SegmentUpsert(
            id="segment-context",
            scene_id="scene-context",
            source_segment_id="shot-1",
            start_seconds=0.0,
            end_seconds=4.0,
        ),
    )
    with sessions() as session:
        context = resources.reconstruction_context_in_transaction(
            session,
            "scene-context",
        )
    assert context is not None
    assert context.project_id == "project-context"
    assert context.segment_id == "segment-context"

    resources.upsert_segment(
        "project-context",
        SegmentUpsert(
            id="segment-context-2",
            scene_id="scene-context",
            source_segment_id="shot-2",
            start_seconds=4.0,
            end_seconds=8.0,
        ),
    )
    with sessions() as session:
        with pytest.raises(ProjectResourceConflict, match="multiple segments"):
            resources.reconstruction_context_in_transaction(
                session,
                "scene-context",
            )


def test_alembic_baseline_adopts_legacy_sqlite_without_losing_rows(tmp_path) -> None:
    """Execute the revision DDL with a tiny Alembic-op compatibility shim.

    The repository dependency is installed in real API environments. The shim
    keeps this migration smoke test runnable in the existing offline developer
    venv while still executing the revision's actual ``upgrade`` function.
    """

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'migration.sqlite3'}")
    with engine.begin() as connection:
        SceneRow.__table__.create(connection)
        VideoAssetRow.__table__.create(connection)
        ReconstructionLeaseRow.__table__.create(connection)
        connection.execute(
            SceneRow.__table__.insert().values(
                id="legacy-scene",
                title="Legacy",
                payload=_scene("legacy-scene", asset_id="legacy-asset"),
            )
        )

        class FakeOperations:
            def get_bind(self):
                return connection

            def create_table(self, name, *elements):
                metadata = MetaData()
                metadata.reflect(bind=connection)
                table = Table(name, metadata, *elements)
                table.create(connection)
                return table

            def create_index(self, name, table_name, columns, unique=False):
                metadata = MetaData()
                metadata.reflect(bind=connection, only=[table_name])
                table = metadata.tables[table_name]
                Index(name, *(table.c[column] for column in columns), unique=unique).create(
                    connection
                )

            def drop_table(self, name):
                metadata = MetaData()
                metadata.reflect(bind=connection, only=[name])
                metadata.tables[name].drop(connection)

        fake_alembic = ModuleType("alembic")
        fake_alembic.op = FakeOperations()
        previous = sys.modules.get("alembic")
        sys.modules["alembic"] = fake_alembic
        try:
            revision_path = (
                Path(__file__).resolve().parents[1]
                / "alembic"
                / "versions"
                / "20260717_0001_project_foundation.py"
            )
            spec = importlib.util.spec_from_file_location(
                "project_foundation_migration",
                revision_path,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.upgrade()
            module.upgrade()  # adoption is deliberately restart/idempotency safe
        finally:
            if previous is None:
                sys.modules.pop("alembic", None)
            else:
                sys.modules["alembic"] = previous

        tables = set(inspect(connection).get_table_names())
        assert {
            "projects",
            "matches",
            "match_snapshots",
            "external_references",
            "project_scenes",
            "project_video_assets",
            "segments",
            "analysis_runs",
        } <= tables
        legacy = connection.execute(
            select(SceneRow.id).where(SceneRow.id == "legacy-scene")
        ).scalar_one()
        assert legacy == "legacy-scene"
    engine.dispose()
