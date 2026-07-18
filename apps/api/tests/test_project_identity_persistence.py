from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Index, MetaData, Table, create_engine, event, func, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, SceneRow
from app.project_identity import sync_project_identities_from_scene
from app.project_identity_repository import (
    ProjectIdentityConflict,
    ProjectIdentityRepository,
)
from app.project_models import ProjectPersonMembershipRow, ProjectPersonRow
from app.project_match_repository import ProjectMatchRepository
from app.project_resource_repository import ProjectResourceRepository
from app.project_identity_contract import ProjectPersonSyncItem
from app.project_lifecycle_contract import ProjectCreate
from app.project_match_persistence_contract import MatchSnapshotCreate, MatchUpsert
from app.project_store import ProjectStore


@pytest.fixture
def identity_persistence():
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
    identities = ProjectIdentityRepository(sessions)
    with sessions.begin() as session:
        for scene_id in ("scene-a", "scene-b"):
            session.add(
                SceneRow(
                    id=scene_id,
                    title=scene_id,
                    payload={"id": scene_id, "payload": {"canonicalPeople": []}},
                )
            )
    store.create_project(ProjectCreate(id="project-1", title="Test match"))
    resources.link_scene("project-1", "scene-a")
    resources.link_scene("project-1", "scene-b")
    matches.publish(
        "project-1",
        MatchUpsert(id="match-canonical", name="Test match"),
        MatchSnapshotCreate(
            provider="private-test-provider",
            external_event_id="private-event-42",
            payload={
                "schemaVersion": 1,
                "id": "match-canonical",
                "roster": [
                    {
                        "id": "player-canonical-8",
                        "name": "Home Eight",
                        "teamId": "team-home",
                        "position": "Midfielder",
                        "number": "8",
                    }
                ],
            },
        ),
    )
    yield store, resources, matches, identities, sessions
    Base.metadata.drop_all(engine)
    engine.dispose()


def _scene(scene_id: str, people: list[dict]) -> dict:
    return {"id": scene_id, "payload": {"canonicalPeople": people}}


def _person(
    canonical_id: str,
    *,
    roster_id: str | None = None,
    label: str | None = None,
) -> dict:
    return {
        "canonicalPersonId": canonical_id,
        "displayName": label or canonical_id,
        "teamId": "home",
        "role": "player",
        "identityStatus": "resolved",
        "identityConfidence": 0.91,
        "externalPlayerId": roster_id,
        "observations": [{"frameIndex": 1}, {"frameIndex": 2}],
    }


def test_project_store_has_no_identity_persistence_authority() -> None:
    removed_methods = {
        "list_project_people",
        "get_project_person",
        "assign_project_person_membership",
        "sync_project_people",
    }

    assert removed_methods.isdisjoint(vars(ProjectStore))


def test_project_identity_list_batches_membership_projection(
    identity_persistence,
) -> None:
    _store, _resources, _matches, identities, sessions = identity_persistence
    identities.sync_scene_people(
        "project-1",
        "scene-a",
        [
            ProjectPersonSyncItem(scene_person_id=f"local-{index}", display_name=f"P{index}")
            for index in range(3)
        ],
    )
    statements: list[str] = []
    engine = sessions.kw["bind"]

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        people = identities.list_for_project("project-1")
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert len(people) == 3
    assert sum(" from project_person_memberships" in item for item in statements) == 1


def test_sync_merges_only_verified_roster_people_and_is_idempotent(
    identity_persistence,
) -> None:
    store, resources, matches, identities, sessions = identity_persistence
    first = sync_project_identities_from_scene(
        _scene(
            "scene-a",
            [
                _person("canonical-bound-a", roster_id="player-canonical-8"),
                _person("canonical-local", label="Local A"),
            ],
        ),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )
    second = sync_project_identities_from_scene(
        _scene(
            "scene-b",
            [
                _person("canonical-bound-b", roster_id="player-canonical-8"),
                # Equal scene-local ids across scenes are deliberately not a
                # project-level merge signal.
                _person("canonical-local", label="Local B"),
                _person(
                    "canonical-private-upstream",
                    roster_id="provider-player-999",
                    label="Unverified",
                ),
            ],
        ),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )

    assert first.people_created == 2
    assert second.people_created == 2
    assert second.unverified_roster_binding_count == 1
    people = identities.list_for_project("project-1")
    assert len(people) == 4
    bound = next(item for item in people if item.roster_person_id)
    assert bound.roster_person_id == "player-canonical-8"
    assert bound.display_name == "Home Eight"
    assert len(bound.memberships) == 2
    local_memberships = [
        membership
        for person in people
        for membership in person.memberships
        if membership.scene_person_id == "canonical-local"
    ]
    assert len(local_memberships) == 2
    assert len({item.project_person_id for item in local_memberships}) == 2

    # Neither integration provenance nor an unverified upstream player id can
    # escape through the public project identity document.
    serialized = json.dumps(
        [item.model_dump(by_alias=True, mode="json") for item in people],
        sort_keys=True,
    )
    assert "private-test-provider" not in serialized
    assert "private-event-42" not in serialized
    assert "provider-player-999" not in serialized
    assert "externalPlayerId" not in serialized

    repeated = sync_project_identities_from_scene(
        _scene(
            "scene-a",
            [
                _person("canonical-bound-a", roster_id="player-canonical-8"),
                _person("canonical-local", label="Local A"),
            ],
        ),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )
    assert repeated.people_created == 0
    assert repeated.memberships_created == 0
    assert repeated.memberships_updated == 0
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(ProjectPersonRow)) == 4
        assert (
            session.scalar(
                select(func.count()).select_from(ProjectPersonMembershipRow)
            )
            == 5
        )


def test_explicit_membership_survives_automatic_resync(identity_persistence) -> None:
    store, resources, matches, identities, _sessions = identity_persistence
    for scene_id, label in (("scene-a", "A"), ("scene-b", "B")):
        sync_project_identities_from_scene(
            _scene(scene_id, [_person(f"canonical-{label.lower()}", label=label)]),
            project_id="project-1",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )
    people = identities.list_for_project("project-1")
    target = next(
        person
        for person in people
        if person.memberships[0].scene_id == "scene-a"
    )
    identities.assign_membership(
        "project-1",
        target.id,
        "scene-b",
        "canonical-b",
    )

    report = sync_project_identities_from_scene(
        _scene("scene-b", [_person("canonical-b", label="B rebuilt")]),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )
    merged = next(item for item in report.people if item.id == target.id)
    assert len(merged.memberships) == 2
    assert any(
        membership.assignment_source == "explicit"
        for membership in merged.memberships
    )
    assert len(report.people) == 1


def test_explicit_membership_rejects_conflicting_accepted_roster_identity(
    identity_persistence,
) -> None:
    store, resources, matches, identities, _sessions = identity_persistence
    sync_project_identities_from_scene(
        _scene("scene-a", [_person("bound", roster_id="player-canonical-8")]),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )
    sync_project_identities_from_scene(
        _scene("scene-b", [_person("local")]),
        project_id="project-1",
        projects=store,
        resources=resources,
        matches=matches,
        identities=identities,
    )
    people = identities.list_for_project("project-1")
    local = next(person for person in people if person.roster_person_id is None)
    identities.assign_membership(
        "project-1",
        local.id,
        "scene-b",
        "local",
    )
    with pytest.raises(ProjectIdentityConflict, match="conflicts with explicit"):
        sync_project_identities_from_scene(
            _scene("scene-b", [_person("local", roster_id="player-canonical-8")]),
            project_id="project-1",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )


def test_identity_repository_rejects_noncanonical_roster_id(
    identity_persistence,
) -> None:
    _store, _resources, _matches, identities, _sessions = identity_persistence
    with pytest.raises(ProjectIdentityConflict, match="canonical match snapshot"):
        identities.sync_scene_people(
            "project-1",
            "scene-a",
            [
                ProjectPersonSyncItem(
                    scene_person_id="canonical-a",
                    roster_person_id="raw-provider-player-id",
                    display_name="Unsafe",
                )
            ],
        )


def test_identity_sync_requires_explicit_project_and_exact_scene_owner(
    identity_persistence,
) -> None:
    store, resources, matches, identities, sessions = identity_persistence
    scene = _scene("scene-a", [_person("canonical-a")])

    with pytest.raises(TypeError):
        sync_project_identities_from_scene(
            scene,
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )
    with pytest.raises(ProjectIdentityConflict, match="Project id is required"):
        sync_project_identities_from_scene(
            scene,
            project_id="",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )
    with pytest.raises(
        ProjectIdentityConflict,
        match="Project project-missing was not found",
    ):
        sync_project_identities_from_scene(
            scene,
            project_id="project-missing",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )

    with sessions.begin() as session:
        session.add(
            SceneRow(
                id="scene-other",
                title="scene-other",
                payload={
                    "id": "scene-other",
                    "payload": {"canonicalPeople": []},
                },
            )
        )
    store.create_project(ProjectCreate(id="project-other", title="Other match"))
    resources.link_scene("project-other", "scene-other")

    with pytest.raises(
        ProjectIdentityConflict,
        match="Scene scene-a is not owned by project project-other",
    ):
        sync_project_identities_from_scene(
            scene,
            project_id="project-other",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )
    with pytest.raises(
        ProjectIdentityConflict,
        match="Scene scene-other is not owned by project project-1",
    ):
        sync_project_identities_from_scene(
            _scene("scene-other", [_person("canonical-other")]),
            project_id="project-1",
            projects=store,
            resources=resources,
            matches=matches,
            identities=identities,
        )

    assert identities.list_for_project("project-1") == []
    assert identities.list_for_project("project-other") == []


def test_identity_migration_is_additive_and_restart_safe(tmp_path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'identity.sqlite3'}")
    with engine.begin() as connection:
        Base.metadata.tables["scenes"].create(connection)
        Base.metadata.tables["matches"].create(connection)
        Base.metadata.tables["projects"].create(connection)

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
                Index(
                    name,
                    *(table.c[column] for column in columns),
                    unique=unique,
                ).create(connection)

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
                / "20260717_0002_project_identity.py"
            )
            spec = importlib.util.spec_from_file_location(
                "project_identity_migration",
                revision_path,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.upgrade()
            module.upgrade()
        finally:
            if previous is None:
                sys.modules.pop("alembic", None)
            else:
                sys.modules["alembic"] = previous

        assert {
            "project_people",
            "project_person_memberships",
        } <= set(inspect(connection).get_table_names())
    engine.dispose()
