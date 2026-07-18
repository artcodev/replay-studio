import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
import httpx
from sqlalchemy import JSON, MetaData, Table, create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

import app.project_routes as project_routes
import app.project_match_routes as project_match_routes
from app import database, schema_migrations
from app.project_identifiers import stable_identifier
from app.project_match_repository import ProjectMatchRepository, canonical_payload_hash
from app.project_store import ProjectStore


def test_init_database_uses_versioned_migrations(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        schema_migrations,
        "upgrade_database",
        lambda revision="head": calls.append(revision),
    )

    database.init_database()

    assert calls == ["head"]


def test_alembic_database_url_supports_percent_encoded_passwords() -> None:
    database_url = "postgresql+psycopg://user:p%40ss%25word@db/replay"
    config = Config()

    config.set_main_option(
        "sqlalchemy.url",
        schema_migrations.alembic_config_value(database_url),
    )

    assert config.get_main_option("sqlalchemy.url") == database_url


def test_explicit_alembic_database_url_wins_over_application_settings(
    monkeypatch,
) -> None:
    explicit = "sqlite+pysqlite:////tmp/isolated-migration.db"
    monkeypatch.setattr(
        schema_migrations,
        "get_settings",
        lambda: type("Settings", (), {"database_url": "sqlite:///wrong.db"})(),
    )

    assert schema_migrations.resolve_alembic_database_url(explicit) == explicit


def _migration_config(database_url: str) -> Config:
    package_root = Path(__file__).resolve().parents[1]
    config = Config(str(package_root / "alembic.ini"))
    config.set_main_option("script_location", str(package_root / "alembic"))
    config.set_main_option(
        "sqlalchemy.url",
        schema_migrations.alembic_config_value(database_url),
    )
    return config


def _request(application: FastAPI, method: str, path: str) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path)

    return asyncio.run(send())


def test_scene_index_migration_projects_existing_json_once(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'scene-index.db'}"
    config = _migration_config(database_url)
    command.upgrade(config, "20260718_0003")

    engine = create_engine(database_url)
    metadata = MetaData()
    scenes = Table("scenes", metadata, autoload_with=engine)
    segment = {
        "id": "segment-1",
        "title": "1-A",
        "duration": 7.25,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "filename": "match.mp4",
                "parentSceneId": "root-1",
                "selectedSegmentId": "shot-01",
            }
        },
    }
    composite = {
        "id": "composite-1",
        "title": "1 multi-pass",
        "duration": 7.25,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "filename": "match.mp4",
                "selectedSegmentId": "shot-01",
                "multiPass": {"parentSceneId": "root-1", "status": "ready"},
            }
        },
    }
    with engine.begin() as connection:
        connection.execute(
            scenes.insert(),
            [
                {"id": segment["id"], "title": segment["title"], "payload": segment},
                {
                    "id": composite["id"],
                    "title": composite["title"],
                    "payload": composite,
                },
            ],
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert {
        "duration",
        "kind",
        "parent_scene_id",
        "selected_segment_id",
    }.issubset({column["name"] for column in inspector.get_columns("scenes")})
    indexes = {index["name"]: index for index in inspector.get_indexes("scenes")}
    assert {
        "ix_scenes_kind",
        "ix_scenes_parent_segment",
        "ix_scenes_updated_at",
    }.issubset(indexes)
    assert indexes["ix_scenes_parent_segment"]["column_names"] == [
        "parent_scene_id",
        "selected_segment_id",
        "kind",
    ]

    metadata = MetaData()
    scenes = Table("scenes", metadata, autoload_with=engine)
    # Reflection on SQLite preserves JSON as a generic JSON column; selecting
    # only relational metadata also guards the migration assertion itself
    # against accidentally depending on the dense payload.
    assert isinstance(scenes.c.payload.type, JSON)
    with engine.connect() as connection:
        rows = {
            row.id: row
            for row in connection.execute(
                select(
                    scenes.c.id,
                    scenes.c.duration,
                    scenes.c.kind,
                    scenes.c.parent_scene_id,
                    scenes.c.selected_segment_id,
                )
            )
        }
    assert rows["segment-1"].duration == 7.25
    assert rows["segment-1"].kind == "segment"
    assert rows["segment-1"].parent_scene_id == "root-1"
    assert rows["segment-1"].selected_segment_id == "shot-01"
    assert rows["composite-1"].kind == "multi-pass"
    assert rows["composite-1"].parent_scene_id == "root-1"
    engine.dispose()


def test_scheduler_cutover_reconciles_only_non_runnable_telemetry(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'telemetry-cutover.db'}"
    config = _migration_config(database_url)
    command.upgrade(config, "20260718_0004")

    engine = create_engine(database_url)
    metadata = MetaData()
    projects = Table("projects", metadata, autoload_with=engine)
    scenes = Table("scenes", metadata, autoload_with=engine)
    jobs = Table("reconstruction_jobs", metadata, autoload_with=engine)
    runs = Table("analysis_runs", metadata, autoload_with=engine)
    with engine.begin() as connection:
        connection.execute(
            projects.insert(),
            {
                "id": "project-1",
                "title": "Telemetry cutover",
                "status": "active",
                "revision": 1,
                "metadata": {},
            },
        )
        connection.execute(
            scenes.insert(),
            [
                {"id": scene_id, "title": scene_id, "payload": {"id": scene_id}}
                for scene_id in ("scene-current", "scene-active", "scene-orphan")
            ],
        )
        connection.execute(
            jobs.insert(),
            [
                {
                    "scene_id": "scene-current",
                    "run_id": "run-new",
                    "input_fingerprint": "fingerprint-new",
                    "input_revision": 2,
                    "status": "ready",
                    "requested_at": 1.0,
                    "updated_at": 2.0,
                },
                {
                    "scene_id": "scene-active",
                    "run_id": "run-active",
                    "input_fingerprint": "fingerprint-active",
                    "input_revision": 1,
                    "status": "queued",
                    "requested_at": 3.0,
                    "updated_at": 3.0,
                },
            ],
        )
        connection.execute(
            runs.insert(),
            [
                {
                    "id": "analysis-current",
                    "project_id": "project-1",
                    "scene_id": "scene-current",
                    "kind": "reconstruction",
                    "status": "running",
                    "source_run_id": "run-new",
                    "progress": {"phase": "old"},
                    "diagnostics": {},
                },
                {
                    "id": "analysis-superseded",
                    "project_id": "project-1",
                    "scene_id": "scene-current",
                    "kind": "reconstruction",
                    "status": "running",
                    "source_run_id": "run-old",
                    "progress": {"phase": "old"},
                    "diagnostics": {},
                },
                {
                    "id": "analysis-active",
                    "project_id": "project-1",
                    "scene_id": "scene-active",
                    "kind": "reconstruction",
                    "status": "running",
                    "source_run_id": "run-active",
                    "progress": {"phase": "preparing"},
                    "diagnostics": {},
                },
                {
                    "id": "analysis-orphan",
                    "project_id": "project-1",
                    "scene_id": "scene-orphan",
                    "kind": "reconstruction",
                    "status": "queued",
                    "source_run_id": "run-missing",
                    "progress": {},
                    "diagnostics": {},
                },
            ],
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    metadata = MetaData()
    runs = Table("analysis_runs", metadata, autoload_with=engine)
    with engine.connect() as connection:
        rows = {
            row.id: row
            for row in connection.execute(
                select(runs.c.id, runs.c.status, runs.c.progress, runs.c.completed_at)
            )
        }
    assert rows["analysis-current"].status == "succeeded"
    assert rows["analysis-current"].progress["overallPercent"] == 100
    assert rows["analysis-current"].completed_at is not None
    assert rows["analysis-superseded"].status == "cancelled"
    assert "superseded" in rows["analysis-superseded"].progress["detail"]
    assert rows["analysis-orphan"].status == "cancelled"
    assert rows["analysis-active"].status == "running"
    assert rows["analysis-active"].progress == {"phase": "preparing"}
    assert rows["analysis-active"].completed_at is None
    engine.dispose()


def test_pipeline_scheduler_migration_adds_control_tables_and_media_pointer(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-schema.db'}"
    config = _migration_config(database_url)
    command.upgrade(config, "20260718_0005")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "generation_key" not in {
        column["name"] for column in inspector.get_columns("video_assets")
    }
    assert "pipeline_jobs" not in inspector.get_table_names()
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "generation_key" in {
        column["name"] for column in inspector.get_columns("video_assets")
    }
    assert {"pipeline_jobs", "pipeline_job_leases"}.issubset(
        inspector.get_table_names()
    )
    assert {
        "ix_pipeline_jobs_status_available",
        "ix_pipeline_jobs_project_id",
        "ix_pipeline_jobs_kind",
    }.issubset({index["name"] for index in inspector.get_indexes("pipeline_jobs")})
    unique_constraints = {
        constraint["name"]: constraint["column_names"]
        for constraint in inspector.get_unique_constraints("pipeline_jobs")
    }
    assert unique_constraints["uq_pipeline_job_kind_subject"] == [
        "kind",
        "subject_id",
    ]
    engine.dispose()

    command.downgrade(config, "20260718_0005")
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "pipeline_jobs" not in inspector.get_table_names()
    assert "generation_key" not in {
        column["name"] for column in inspector.get_columns("video_assets")
    }
    engine.dispose()


def test_scene_match_binding_cutover_removes_only_embedded_match_copy(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'scene-match-cutover.db'}"
    config = _migration_config(database_url)
    command.upgrade(config, "20260718_0006")

    engine = create_engine(database_url)
    metadata = MetaData()
    scenes = Table("scenes", metadata, autoload_with=engine)
    with engine.begin() as connection:
        connection.execute(
            scenes.insert(),
            [
                {
                    "id": "scene-with-copy",
                    "title": "Legacy match copy",
                    "payload": {
                        "id": "scene-with-copy",
                        "payload": {
                            "matchBinding": {
                                "snapshotId": "snapshot-old",
                                "roster": [{"id": "player-old"}],
                            },
                            "videoAsset": {"id": "asset-1"},
                        },
                        "matchBinding": {"unrelated": "top-level"},
                    },
                },
                {
                    "id": "scene-canonical",
                    "title": "Canonical scene",
                    "payload": {
                        "id": "scene-canonical",
                        "payload": {"videoAsset": {"id": "asset-2"}},
                    },
                },
            ],
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    metadata = MetaData()
    scenes = Table("scenes", metadata, autoload_with=engine)
    with engine.connect() as connection:
        documents = {
            row.id: row.payload
            for row in connection.execute(select(scenes.c.id, scenes.c.payload))
        }
    assert "matchBinding" not in documents["scene-with-copy"]["payload"]
    assert documents["scene-with-copy"]["payload"]["videoAsset"] == {
        "id": "asset-1"
    }
    assert documents["scene-with-copy"]["matchBinding"] == {
        "unrelated": "top-level"
    }
    assert documents["scene-canonical"]["payload"] == {
        "videoAsset": {"id": "asset-2"}
    }
    engine.dispose()

    # The data cutover is intentionally irreversible; downgrading the schema
    # must not fabricate the deleted match snapshot.
    command.downgrade(config, "20260718_0006")
    engine = create_engine(database_url)
    metadata = MetaData()
    scenes = Table("scenes", metadata, autoload_with=engine)
    with engine.connect() as connection:
        document = connection.execute(
            select(scenes.c.payload).where(scenes.c.id == "scene-with-copy")
        ).scalar_one()
    assert "matchBinding" not in document["payload"]
    engine.dispose()


def test_match_snapshot_cutover_is_immutable_idempotent_and_visible_via_api(
    tmp_path,
    monkeypatch,
) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'canonical-match-cutover.db'}"
    config = _migration_config(database_url)
    command.upgrade(config, "20260718_0007")

    retired_payload = {
        "schemaVersion": 1,
        "matchId": "match-world-cup",
        "event": {
            "name": "Spain vs Belgium",
            "competition": "World Cup",
            "season": "2026",
            "date": "2026-07-10",
            "time": "19:00",
            "status": "Match Finished",
            "home": {"id": "team-spain", "name": "Spain"},
            "away": {"id": "team-belgium", "name": "Belgium"},
            "score": {"home": 2, "away": 1},
        },
        "players": [
            {
                "id": "player-eight",
                "name": "Player Eight",
                "teamId": "team-spain",
                "position": "GK",
                "number": "8",
                "lineupRole": "starter",
            },
            {
                "id": "player-nine",
                "name": "Player Nine",
                "teamId": "team-belgium",
                "position": "FW",
                "number": "9",
                "lineupRole": "starter",
            },
        ],
        "lineup": [],
        "timeline": [
            {
                "id": "event-goal",
                "minute": 51,
                "type": "goal",
                "label": "Goal · Player Eight",
                "playerId": "player-eight",
                "teamId": "team-spain",
            }
        ],
        "substitutions": [
            {
                "id": "substitution-one",
                "minute": 70,
                "teamId": "team-spain",
                "playerOutId": "player-eight",
                "playerInId": "player-ten",
                "label": "Substitution",
            }
        ],
        "rosterQuality": {
            "status": "partial",
            "playerCount": 2,
            "homePlayerCount": 1,
            "awayPlayerCount": 1,
            "automaticIdentityEligible": False,
            "manualIdentityEligible": True,
            "reasons": ["test-partial"],
        },
        "warnings": ["Historical provider warning"],
    }
    retired_hash = canonical_payload_hash(retired_payload)
    engine = create_engine(database_url)
    metadata = MetaData()
    matches = Table("matches", metadata, autoload_with=engine)
    projects = Table("projects", metadata, autoload_with=engine)
    snapshots = Table("match_snapshots", metadata, autoload_with=engine)
    with engine.begin() as connection:
        connection.execute(
            matches.insert(),
            {
                "id": "match-world-cup",
                "sport": "football",
                "name": "Spain vs Belgium",
                "competition": "World Cup",
                "season": "2026",
                "kickoff_at": "2026-07-10 19:00",
                "status": "Match Finished",
                "home_team_name": "Spain",
                "away_team_name": "Belgium",
                "metadata": {"score": {"home": 2, "away": 1}},
            },
        )
        connection.execute(
            projects.insert(),
            {
                "id": "project-live",
                "title": "World Cup moment",
                "status": "active",
                "revision": 4,
                "match_id": "match-world-cup",
                "current_match_snapshot_id": "snapshot-retired",
                "metadata": {},
            },
        )
        connection.execute(
            snapshots.insert(),
            {
                "id": "snapshot-retired",
                "project_id": "project-live",
                "match_id": "match-world-cup",
                "provider": "canonical",
                "external_event_id": "event-retired-backfill",
                "schema_version": 1,
                "fetched_at": "2026-07-17T12:00:00+00:00",
                "content_hash": retired_hash,
                "is_current": True,
                "payload": retired_payload,
            },
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    metadata = MetaData()
    projects = Table("projects", metadata, autoload_with=engine)
    snapshots = Table("match_snapshots", metadata, autoload_with=engine)
    with engine.connect() as connection:
        project = connection.execute(
            select(
                projects.c.current_match_snapshot_id,
                projects.c.match_id,
                projects.c.revision,
            ).where(projects.c.id == "project-live")
        ).one()
        rows = {
            row.id: row
            for row in connection.execute(
                select(
                    snapshots.c.id,
                    snapshots.c.match_id,
                    snapshots.c.content_hash,
                    snapshots.c.is_current,
                    snapshots.c.payload,
                ).where(snapshots.c.project_id == "project-live")
            )
        }

    assert project.match_id == "match-world-cup"
    assert project.revision == 5
    assert project.current_match_snapshot_id != "snapshot-retired"
    assert len(rows) == 2
    retired = rows["snapshot-retired"]
    assert retired.payload == retired_payload
    assert retired.content_hash == retired_hash
    assert retired.is_current is False

    replacement = rows[project.current_match_snapshot_id]
    assert replacement.is_current is True
    assert replacement.match_id == "match-world-cup"
    assert replacement.payload["id"] == "match-world-cup"
    assert replacement.payload["homeTeam"]["name"] == "Spain"
    assert replacement.payload["awayTeam"]["name"] == "Belgium"
    assert replacement.payload["roster"] == retired_payload["players"]
    assert replacement.payload["events"] == retired_payload["timeline"]
    assert replacement.payload["sync"] == {
        "state": "partial",
        "syncedAt": "2026-07-17T12:00:00+00:00",
        "stale": False,
        "warnings": ["Historical provider warning"],
    }
    assert replacement.content_hash == canonical_payload_hash(replacement.payload)
    assert replacement.id == stable_identifier(
        "snapshot",
        "project-live",
        replacement.content_hash,
        length=32,
    )

    store = ProjectStore(sessionmaker(bind=engine, expire_on_commit=False))
    match_repository = ProjectMatchRepository(
        sessionmaker(bind=engine, expire_on_commit=False)
    )
    monkeypatch.setattr(project_match_routes, "project_store", store)
    monkeypatch.setattr(project_match_routes, "project_matches", match_repository)
    application = FastAPI()
    application.include_router(project_routes.router)
    response = _request(application, "GET", "/api/projects/project-live/match")
    assert response.status_code == 200
    match = response.json()
    assert match["name"] == "Spain vs Belgium"
    assert match["competition"] == "World Cup"
    assert match["score"] == {"home": 2, "away": 1}
    assert match["homeTeam"]["name"] == "Spain"
    assert match["awayTeam"]["name"] == "Belgium"
    assert [player["name"] for player in match["roster"]] == [
        "Player Eight",
        "Player Nine",
    ]
    assert match["events"][0]["label"] == "Goal · Player Eight"
    assert match["substitutions"][0]["playerOutId"] == "player-eight"
    engine.dispose()

    # Re-run the data migration itself. Downgrade intentionally keeps the
    # canonical replacement selected, so a second upgrade must be a no-op.
    command.downgrade(config, "20260718_0007")
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    metadata = MetaData()
    projects = Table("projects", metadata, autoload_with=engine)
    snapshots = Table("match_snapshots", metadata, autoload_with=engine)
    with engine.connect() as connection:
        repeated_project = connection.execute(
            select(
                projects.c.current_match_snapshot_id,
                projects.c.revision,
            ).where(projects.c.id == "project-live")
        ).one()
        repeated_count = len(
            connection.execute(
                select(snapshots.c.id).where(
                    snapshots.c.project_id == "project-live"
                )
            ).all()
        )
    assert repeated_project.current_match_snapshot_id == replacement.id
    assert repeated_project.revision == 5
    assert repeated_count == 2
    engine.dispose()
