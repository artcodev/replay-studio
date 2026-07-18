from __future__ import annotations

import asyncio
from types import SimpleNamespace

import app.project_match_routes as project_match_routes
from app.match_contracts import ExternalEvent, ExternalTeam


def test_provider_neutral_date_search_records_actual_fallback_provider(
    monkeypatch,
) -> None:
    event = ExternalEvent(
        id="sportsdb-42",
        provider="thesportsdb",
        name="Spain vs Belgium",
        home=ExternalTeam(id="home", name="Spain"),
        away=ExternalTeam(id="away", name="Belgium"),
    )

    async def events_by_date_with_fallback(date: str):
        assert date == "2026-07-10"
        return "thesportsdb", [event]

    captured: dict[str, object] = {}

    def remember_candidates(events, *, provider: str):
        captured["events"] = events
        captured["provider"] = provider
        return ["mapped-candidate"]

    monkeypatch.setattr(
        project_match_routes,
        "sports_provider",
        SimpleNamespace(
            events_by_date_with_fallback=events_by_date_with_fallback,
        ),
    )
    monkeypatch.setattr(
        project_match_routes,
        "remember_match_candidates",
        remember_candidates,
    )
    monkeypatch.setattr(
        project_match_routes,
        "project_store",
        SimpleNamespace(project_exists=lambda project_id: project_id == "project-1"),
    )

    result = asyncio.run(
        project_match_routes.search_matches("project-1", q=None, date="2026-07-10")
    )

    assert result == ["mapped-candidate"]
    assert captured == {"events": [event], "provider": "thesportsdb"}
