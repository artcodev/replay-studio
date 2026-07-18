from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main_module
from app.canonical_match import (
    canonicalize_event_bundle,
)
from app.canonical_match_persistence import persist_canonical_match
from app.database import Base, SceneRow
from app.external_reference_repository import ExternalReferenceRepository
from app.project_models import ExternalReferenceRow, MatchRow, MatchSnapshotRow
from app.project_match_repository import ProjectMatchRepository
from app.project_lifecycle_contract import ProjectCreate
from app.project_store import ProjectStore
from app.match_contracts import (
    EventBundle,
    ExternalEvent,
    ExternalLineupEntry,
    ExternalPlayer,
    ExternalRosterQuality,
    ExternalSubstitution,
    ExternalTeam,
    ExternalTimelineEvent,
)


def _bundle() -> EventBundle:
    return EventBundle(
        source="provider-secret-name",
        event=ExternalEvent(
            id="upstream-match-77",
            provider="provider-secret-name",
            name="Spain vs Belgium",
            date="2026-07-10",
            league="World Cup",
            home=ExternalTeam(id="upstream-team-home", name="Spain"),
            away=ExternalTeam(id="upstream-team-away", name="Belgium"),
            home_score=2,
            away_score=1,
        ),
        players=[
            ExternalPlayer(
                id="upstream-player-8",
                name="Player Eight",
                team_id="upstream-team-home",
                team_name="Spain",
                number="8",
                lineup_role="starter",
            ),
            ExternalPlayer(
                id="upstream-player-9",
                name="Player Nine",
                team_id="upstream-team-away",
                team_name="Belgium",
                number="9",
                lineup_role="starter",
            ),
        ],
        lineup=[
            ExternalLineupEntry(
                id="upstream-lineup-1",
                player_id="upstream-player-8",
                player_name="Player Eight",
                team_id="upstream-team-home",
                team_name="Spain",
                side="home",
                number="8",
                role="starter",
                order=0,
            )
        ],
        timeline=[
            ExternalTimelineEvent(
                id="upstream-event-1",
                minute=51,
                type="Goal",
                detail="Normal Goal",
                label="Goal · Player Eight",
                player_id="upstream-player-8",
                player_name="Player Eight",
                team_id="upstream-team-home",
                team_name="Spain",
            )
        ],
        substitutions=[
            ExternalSubstitution(
                id="upstream-sub-1",
                minute=70,
                team_id="upstream-team-home",
                team_name="Spain",
                player_out_id="upstream-player-8",
                player_out_name="Player Eight",
                player_in_id="upstream-player-10",
                player_in_name="Player Ten",
                label="Substitution",
            )
        ],
        roster_quality=ExternalRosterQuality(
            status="partial",
            player_count=2,
            home_player_count=1,
            away_player_count=1,
            automatic_identity_eligible=False,
            manual_identity_eligible=True,
            reasons=["test-partial"],
        ),
        fetched_at="2026-07-17T12:00:00+00:00",
    )


def test_canonical_match_hides_provider_and_upstream_identifiers() -> None:
    canonical = canonicalize_event_bundle(_bundle())

    assert canonical["id"].startswith("match-")
    assert canonical["homeTeam"]["id"].startswith("team-")
    assert canonical["roster"][0]["id"].startswith("player-")
    assert canonical["events"][0]["id"].startswith("event-")
    assert canonical["events"][0]["type"] == "goal"

    serialized = str(canonical)
    assert "provider-secret-name" not in serialized
    assert "upstream-match-77" not in serialized
    assert "upstream-player-8" not in serialized


def test_canonical_ids_are_stable_for_the_same_provider_bundle() -> None:
    first = canonicalize_event_bundle(_bundle())
    second = canonicalize_event_bundle(_bundle())

    assert first == second


def test_canonical_initialization_and_repeated_api_startup_do_not_mint_matches(
    monkeypatch,
) -> None:
    """API startup is operational only; Match state changes through explicit sync."""

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
    store.create_project(ProjectCreate(id="project-canonical", title="World Cup moment"))

    first = persist_canonical_match(
        "project-canonical",
        _bundle(),
        matches=matches,
        references=references,
    )
    second = persist_canonical_match(
        "project-canonical",
        _bundle(),
        matches=matches,
        references=references,
    )
    assert first == second

    monkeypatch.setattr(main_module, "init_database", lambda: None)

    async def start_twice() -> None:
        async with main_module.lifespan(main_module.app):
            pass
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(start_twice())

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(MatchRow)) == 1
        assert session.scalar(select(func.count()).select_from(MatchSnapshotRow)) == 1
        assert session.scalar(select(func.count()).select_from(SceneRow)) == 0
    project = store.get_project("project-canonical")
    assert project is not None
    assert project.match_id == first["id"]
    assert matches.current_payload(project.id) == first

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_canonical_publication_rolls_back_every_table_on_reference_failure() -> None:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    projects = ProjectStore(sessions)
    matches = ProjectMatchRepository(sessions)

    class FailingReferences(ExternalReferenceRepository):
        def upsert_many_in_transaction(self, session, requests):
            del session, requests
            raise RuntimeError("reference write failed")

    projects.create_project(ProjectCreate(id="project-rollback", title="Rollback"))
    with pytest.raises(RuntimeError, match="reference write failed"):
        persist_canonical_match(
            "project-rollback",
            _bundle(),
            matches=matches,
            references=FailingReferences(sessions),
        )

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(MatchRow)) == 0
        assert session.scalar(select(func.count()).select_from(MatchSnapshotRow)) == 0
        assert (
            session.scalar(select(func.count()).select_from(ExternalReferenceRow))
            == 0
        )
    project = projects.get_project("project-rollback")
    assert project is not None
    assert project.match_id is None
    assert project.current_match_snapshot_id is None

    Base.metadata.drop_all(engine)
    engine.dispose()
