from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.main import app
import app.match_import_routes as match_import_routes
from app.manual_match_import import build_manual_match_bundle
from app.external_reference_repository import ExternalReferenceRepository
from app.project_match_repository import ProjectMatchRepository
from app.project_lifecycle_contract import ProjectCreate
from app.project_store import ProjectStore
from app.match_contracts import ManualMatchImportRequest


def _request_body(*, player_count: int = 22) -> dict:
    players = []
    for index in range(player_count):
        home = index < (player_count + 1) // 2
        players.append(
            {
                "id": f"player-{index}",
                "name": f"Player {index}",
                "team_id": "team-home" if home else "team-away",
                "number": str(
                    index + 1
                    if home
                    else index + 1 - (player_count + 1) // 2
                ),
                "position": "Midfielder",
                "lineup_role": "starter",
            }
        )
    return {
        "event": {
            "id": "manual-event-1",
            "name": "Manual Home vs Manual Away",
            "date": "2026-07-17",
            "league": "Community Cup",
        },
        "teams": {
            "home": {"id": "team-home", "name": "Manual Home"},
            "away": {"id": "team-away", "name": "Manual Away"},
        },
        "players": players,
        "provenance": {
            "label": "Official match sheet",
            "reference": "local://match-sheet.json",
            "capturedAt": "2026-07-17T12:30:00Z",
            "notes": "Noncommercial research import",
        },
    }


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_async_request(method, path, **kwargs))


@pytest.fixture
def match_persistence(monkeypatch):
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
    store.create_project(ProjectCreate(id="project-manual", title="Manual match"))
    monkeypatch.setattr(match_import_routes, "project_store", store)
    monkeypatch.setattr(match_import_routes, "project_matches", matches)
    monkeypatch.setattr(match_import_routes, "external_references", references)
    yield store, matches
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_manual_json_import_updates_only_the_canonical_project_match(
    match_persistence,
) -> None:
    _projects, matches = match_persistence
    body = _request_body()
    body["timeline"] = [
        {
            "id": "goal-1",
            "minute": 61,
            "type": "goal",
            "label": "Goal",
            "player_id": "player-0",
        }
    ]

    response = _request(
        "POST",
        "/api/projects/project-manual/match/import",
        json=body,
    )

    assert response.status_code == 200, response.text
    match = response.json()
    assert match["sync"]["state"] == "manual"
    assert match["homeTeam"]["name"] == "Manual Home"
    assert match["awayTeam"]["name"] == "Manual Away"
    assert len(match["roster"]) == 22
    assert len(match["events"]) == 1
    assert match["events"][0]["label"] == "Goal"
    # Provider/upstream ids stay behind the canonical Project API boundary.
    assert match["homeTeam"]["id"] != "team-home"
    assert match["roster"][0]["id"] != "player-0"
    assert "provider" not in str(match).lower()

    snapshot = matches.current_payload("project-manual")
    assert snapshot is not None
    assert snapshot["id"] == match["id"]
    assert len(snapshot["roster"]) == 22


def test_partial_manual_roster_remains_available_for_manual_identity(
    match_persistence,
) -> None:
    _projects, matches = match_persistence
    response = _request(
        "POST",
        "/api/projects/project-manual/match/import",
        json=_request_body(player_count=2),
    )

    assert response.status_code == 200, response.text
    assert len(response.json()["roster"]) == 2
    snapshot = matches.current_payload("project-manual")
    assert snapshot is not None
    assert snapshot["rosterQuality"]["automaticIdentityEligible"] is False
    assert snapshot["rosterQuality"]["manualIdentityEligible"] is True


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (lambda body: body.update({"rosterQuality": {}}), "extra_forbidden"),
        (
            lambda body: body["players"][0].update({"unexpected": True}),
            "extra_forbidden",
        ),
        (
            lambda body: body["players"][1].update({"id": "player-0"}),
            "Duplicate player id",
        ),
        (
            lambda body: body["players"][0].update({"team_id": "unknown-team"}),
            "references an unknown team",
        ),
    ],
)
def test_invalid_manual_import_never_changes_the_current_project_snapshot(
    match_persistence,
    mutation,
    detail: str,
) -> None:
    projects, _matches = match_persistence
    before = projects.get_project("project-manual")
    assert before is not None
    body = _request_body()
    mutation(body)

    response = _request(
        "POST",
        "/api/projects/project-manual/match/import",
        json=body,
    )

    assert response.status_code == 422
    assert detail in response.text
    after = projects.get_project("project-manual")
    assert after is not None
    assert after.revision == before.revision
    assert after.current_match_snapshot_id is None


def test_bundled_spain_belgium_roster_stays_strict_and_automatic_ready() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "matches"
        / "spain-belgium-2026-qf.json"
    )
    request = ManualMatchImportRequest.model_validate(
        json.loads(fixture_path.read_text(encoding="utf-8"))
    )

    bundle, provenance = build_manual_match_bundle(request)

    assert len(bundle.players) == 52
    assert len(bundle.lineup) == 52
    assert len(bundle.timeline) == 7
    assert len(bundle.substitutions) == 9
    assert bundle.roster_quality.automatic_identity_eligible is True
    assert provenance["kind"] == "manual-json"
