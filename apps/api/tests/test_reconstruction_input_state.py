from __future__ import annotations

from copy import deepcopy

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, SceneRow, VideoAssetRow
from app.project_match import match_snapshot_reference
from app.project_match_repository import ProjectMatchRepository
from app.project_lifecycle_contract import ProjectCreate
from app.project_match_persistence_contract import MatchSnapshotCreate, MatchUpsert
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore
from app.sample import make_video_scene
from app.scene_document import (
    annotate_reconstruction_input_state,
    reconstruction_input_fingerprint,
)
from app.scene_repository import SceneRepository


def _canonical(roster_name: str, *, fetched_at: str) -> dict:
    return {
        "schemaVersion": 1,
        "id": "match-input-state",
        "name": "Spain vs Belgium",
        "competition": "World Cup",
        "season": "2026",
        "date": "2026-07-10",
        "time": "19:00",
        "status": "finished",
        "score": {"home": 2, "away": 1},
        "homeTeam": {"id": "team-home", "name": "Spain"},
        "awayTeam": {"id": "team-away", "name": "Belgium"},
        "roster": [
            {
                "id": "player-home-8",
                "name": roster_name,
                "teamId": "team-home",
                "teamName": "Spain",
                "number": "8",
                "lineupRole": "starter",
            }
        ],
        "lineup": [],
        "events": [],
        "substitutions": [],
        "rosterQuality": {
            "status": "partial",
            "playerCount": 1,
            "homePlayerCount": 1,
            "awayPlayerCount": 0,
            "automaticIdentityEligible": False,
            "manualIdentityEligible": True,
            "reasons": ["canonical-roster-incomplete"],
        },
        "sync": {"state": "partial", "syncedAt": fetched_at, "warnings": []},
    }


def test_project_boundary_derives_stale_state_without_implicit_scene_hydration() -> None:
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
    scenes = SceneRepository(sessions)
    scene = make_video_scene(
        scene_id="scene-input-state",
        title="1-A",
        duration=8.0,
        video_asset={
            "id": "asset-input-state",
            "filename": "source.mp4",
            "fps": 25.0,
            "analysisFps": 10.0,
            "frameCount": 80,
            "processingState": "ready",
            "selectedSegmentId": "shot-01",
            "reconstruction": {
                "status": "ready",
                "model": "yolo26m.pt",
            },
        },
    )
    with sessions.begin() as session:
        session.add(
            VideoAssetRow(
                id="asset-input-state",
                filename="source.mp4",
                original_name="source.mp4",
                content_type="video/mp4",
                status="ready",
                stage="Ready",
                progress=100,
                scene_id=scene["id"],
            )
        )
    projects.create_project(ProjectCreate(id="project-input-state", title="Match"))
    resources.link_video_asset("project-input-state", "asset-input-state")
    first_snapshot = matches.publish(
        "project-input-state",
        MatchUpsert(id="match-input-state", name="Spain vs Belgium"),
        MatchSnapshotCreate(
            provider="provider-private",
            external_event_id="event-1",
            fetched_at="2026-07-17T10:00:00+00:00",
            payload=_canonical("Player Eight", fetched_at="2026-07-17T10:00:00+00:00"),
        ),
    ).snapshot
    first_ref = match_snapshot_reference(first_snapshot)
    reconstruction = scene["payload"]["videoAsset"]["reconstruction"]
    reconstruction["matchSnapshotRef"] = first_ref
    queued_input = reconstruction_input_fingerprint(scene)
    reconstruction["inputFingerprint"] = queued_input
    scenes.put(scene)
    resources.link_scene("project-input-state", scene["id"], role="segment")

    statements: list[str] = []
    event.listen(
        engine,
        "before_cursor_execute",
        lambda _connection, _cursor, statement, _parameters, _context, _many: statements.append(
            statement.lower()
        ),
    )
    current = scenes.get(scene["id"])
    assert current is not None
    assert "matchBinding" not in current["payload"]
    current_reconstruction = current["payload"]["videoAsset"]["reconstruction"]
    assert "inputState" not in current_reconstruction
    assert not any("match_snapshots" in statement for statement in statements)
    assert not any("projects" in statement for statement in statements)

    annotate_reconstruction_input_state(current, first_ref)
    assert current_reconstruction["inputState"] == "current"
    assert current_reconstruction["currentInputFingerprint"] == queued_input

    second_snapshot = matches.publish(
        "project-input-state",
        MatchUpsert(id="match-input-state", name="Spain vs Belgium"),
        MatchSnapshotCreate(
            provider="provider-private",
            external_event_id="event-1",
            fetched_at="2026-07-17T11:00:00+00:00",
            payload=_canonical("Corrected Player Eight", fetched_at="2026-07-17T11:00:00+00:00"),
        ),
    ).snapshot

    stale = scenes.get(scene["id"])
    assert stale is not None
    annotate_reconstruction_input_state(
        stale,
        match_snapshot_reference(second_snapshot),
    )
    stale_reconstruction = stale["payload"]["videoAsset"]["reconstruction"]
    assert stale_reconstruction["status"] == "ready"
    assert stale_reconstruction["inputFingerprint"] == queued_input
    assert stale_reconstruction["inputState"] == "stale"
    assert stale_reconstruction["inputStateReason"] == "reconstruction-input-changed"
    assert stale_reconstruction["currentInputFingerprint"] != queued_input

    # Reading the new project snapshot only annotates the old result. It does
    # not queue work, replace tracks, or persist derived comparison fields.
    with sessions() as session:
        persisted = session.get(SceneRow, scene["id"]).payload
    persisted_reconstruction = persisted["payload"]["videoAsset"]["reconstruction"]
    assert persisted_reconstruction["status"] == "ready"
    assert "inputState" not in persisted_reconstruction
    assert "currentInputFingerprint" not in persisted_reconstruction

    scenes.put(stale)
    with sessions() as session:
        persisted_after_put = session.get(SceneRow, scene["id"]).payload
    persisted_after_reconstruction = persisted_after_put["payload"]["videoAsset"][
        "reconstruction"
    ]
    assert "inputState" not in persisted_after_reconstruction
    assert "inputStateReason" not in persisted_after_reconstruction
    assert "currentInputFingerprint" not in persisted_after_reconstruction

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_scene_without_run_fingerprint_has_no_implicit_input_state() -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    scenes = SceneRepository(sessions)
    legacy = make_video_scene(
        scene_id="legacy-input-state",
        title="Legacy",
        duration=4.0,
        video_asset={
            "id": "legacy-asset",
            "filename": "legacy.mp4",
            "reconstruction": {"status": "ready", "model": "yolo26m.pt"},
        },
    )
    scenes.put(legacy)

    loaded = scenes.get(legacy["id"])
    assert loaded is not None
    reconstruction = loaded["payload"]["videoAsset"]["reconstruction"]
    assert "inputState" not in reconstruction
    assert "currentInputFingerprint" not in reconstruction
    annotate_reconstruction_input_state(loaded, None)
    assert reconstruction["inputState"] == "unknown"
    assert reconstruction["currentInputFingerprint"].startswith("sha256:")
    scenes.put(loaded)

    with sessions() as session:
        persisted = session.get(SceneRow, legacy["id"]).payload
    persisted_reconstruction = persisted["payload"]["videoAsset"]["reconstruction"]
    assert "inputState" not in persisted_reconstruction
    assert "currentInputFingerprint" not in persisted_reconstruction

    Base.metadata.drop_all(engine)
    engine.dispose()
