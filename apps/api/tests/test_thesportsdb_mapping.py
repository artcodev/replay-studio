from __future__ import annotations

from app.providers.thesportsdb_mapping import map_event, map_lineup


def test_maps_event_contract() -> None:
    event = map_event(
        {
            "idEvent": "42",
            "strEvent": "Aurora vs Atlas",
            "dateEvent": "2026-07-11",
            "idHomeTeam": "1",
            "strHomeTeam": "Aurora",
            "idAwayTeam": "2",
            "strAwayTeam": "Atlas",
            "intHomeScore": "2",
            "intAwayScore": "1",
        }
    )

    assert event.id == "42"
    assert event.home.name == "Aurora"
    assert event.away_score == 1


def test_maps_lineup_side_to_match_national_team() -> None:
    event = map_event(
        {
            "idEvent": "2519345",
            "strEvent": "Spain vs Belgium",
            "idHomeTeam": "133909",
            "strHomeTeam": "Spain",
            "idAwayTeam": "134515",
            "strAwayTeam": "Belgium",
        }
    )
    source = {
        "idPlayer": "34146306",
        "strPlayer": "Thibaut Courtois",
        "strHome": "No",
        "idTeam": "133738",
        "strTeam": "Real Madrid",
    }

    players, entries = map_lineup([source], event)

    assert entries[0].team_id == "134515"
    assert players[0].team_id == "134515"
    assert players[0].team_name == "Belgium"
