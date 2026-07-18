from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.project_routes as project_routes
import app.project_identity_routes as project_identity_routes
from app.database import Base, SceneRow
from app.project_identity_repository import ProjectIdentityRepository
from app.project_identity_contract import ProjectPersonSyncItem
from app.project_lifecycle_contract import ProjectCreate
from app.project_resource_repository import ProjectResourceRepository
from app.project_store import ProjectStore


async def _async_request(application: FastAPI, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _request(application: FastAPI, method: str, path: str, **kwargs):
    return asyncio.run(_async_request(application, method, path, **kwargs))


def test_project_identity_list_and_explicit_membership_are_provider_neutral(
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
    identities = ProjectIdentityRepository(sessions)
    with sessions.begin() as session:
        session.add(
            SceneRow(
                id="scene-1",
                title="1-A",
                payload={"id": "scene-1", "payload": {"canonicalPeople": []}},
            )
        )
    store.create_project(ProjectCreate(id="project-1", title="World Cup"))
    resources.link_scene("project-1", "scene-1")
    identities.sync_scene_people(
        "project-1",
        "scene-1",
        [
            ProjectPersonSyncItem(
                scene_person_id="canonical-a",
                display_name="Anonymous A",
                team_id="home",
                observation_count=4,
            ),
            ProjectPersonSyncItem(
                scene_person_id="canonical-b",
                display_name="Anonymous B",
                team_id="home",
                observation_count=3,
            ),
        ],
    )
    monkeypatch.setattr(project_identity_routes, "project_identities", identities)
    app = FastAPI()
    app.include_router(project_routes.router)

    loaded = _request(app, "GET", "/api/projects/project-1/identities")
    assert loaded.status_code == 200
    identities = loaded.json()
    assert len(identities) == 2
    target = next(
        identity
        for identity in identities
        if identity["memberships"][0]["scenePersonId"] == "canonical-a"
    )
    serialized = str(identities)
    assert "provider" not in serialized.lower()
    assert "externalPlayerId" not in serialized

    assigned = _request(
        app,
        "POST",
        f"/api/projects/project-1/identities/{target['id']}/memberships",
        json={"sceneId": "scene-1", "scenePersonId": "canonical-b"},
    )
    assert assigned.status_code == 200
    assert assigned.json()["assignmentSource"] == "explicit"
    assert assigned.json()["projectPersonId"] == target["id"]

    merged = _request(app, "GET", "/api/projects/project-1/identities")
    assert merged.status_code == 200
    assert len(merged.json()) == 1
    assert len(merged.json()[0]["memberships"]) == 2

    missing = _request(app, "GET", "/api/projects/missing/identities")
    assert missing.status_code == 404

    Base.metadata.drop_all(engine)
    engine.dispose()
